// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * Battery and charger meter for WonderMedia WM8505 netbooks (e.g. Sylvania
 * SYNET07526).
 *
 * These boards have no fuel gauge, no I2C battery IC and no on-SoC ADC.  The
 * battery voltage is instead sensed with an RC-slope "poor man's ADC" on a
 * GPIO: a capacitor is charged through a battery-derived voltage and the time
 * for the GPIO input to cross its logic threshold is inversely related to the
 * battery voltage.  This mirrors the vendor WinCE battery driver for this board
 * variant (board-type 5, sense channel EXTGPIO4).
 *
 * One voltage sample:
 *   1. drive the sense GPIO low as an output for WM8505_BATT_DISCHARGE_MS to
 *      drain the capacitor (>= 5*RC, RC ~= 180 ms);
 *   2. release the GPIO to an input;
 *   3. time how long until the input reads logic-high.
 * A shorter charge time means a higher battery voltage.  Several samples are
 * median-filtered and mapped to a voltage through a board-specific calibration
 * curve measured on a bench supply, then to a percentage through the vendor's
 * open-circuit voltage table.
 *
 * Three further status lines (all optional) reproduce the rest of the vendor
 * driver:
 *   - a battery-low / under-voltage warning input that trips at ~6.8 V;
 *   - an AC-adapter-present input, exposed as a separate mains power supply;
 *   - a charger STATUS input (charging vs. charge-complete), only meaningful
 *     while the adapter is present.
 * The WM8505 GPIO controller (gpio-wmt) has no interrupt support, so all of
 * these inputs are sampled by the periodic poll rather than driven by IRQs.
 *
 * Copyright (C) 2026 Brant Martin
 */

#include <linux/delay.h>
#include <linux/err.h>
#include <linux/gpio/consumer.h>
#include <linux/ktime.h>
#include <linux/math64.h>
#include <linux/mod_devicetable.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/platform_device.h>
#include <linux/power_supply.h>
#include <linux/property.h>
#include <linux/slab.h>
#include <linux/sort.h>
#include <linux/workqueue.h>

/*
 * Timing.  RC ~= 180 ms on this board, so the 300 ms discharge is >= 5*RC and
 * fully drains the sense capacitor; the worst-case charge time in the
 * calibration curve below is 230 ms (at 6.0 V), so a 500 ms charge timeout
 * leaves comfortable margin while still bounding a stuck / disconnected line.
 * The vendor firmware uses a ~1e6-tick busy-loop timeout (WinCE 0xf4241,
 * u-boot "ThreshholdCnt 570000"); those are raw perf-counter budgets for the
 * same physical ~0.5 s ceiling.
 */
#define WM8505_BATT_DISCHARGE_MS	300	/* hold low >= 5*RC (RC ~= 180 ms) */
#define WM8505_BATT_CHARGE_TMO_MS	500	/* longest plausible charge time */
#define WM8505_BATT_SAMPLES		5	/* median-filtered per poll */
#define WM8505_BATT_POLL_MS		5000	/* refresh interval */

/*
 * Calibration curve, measured with a current-limited bench supply on the
 * battery terminals (300 ms discharge, median charge time).  Voltage decreases
 * as charge time increases.  Entries are sorted by ascending charge time.
 *
 * The high-voltage end is far more log-compressed than the original bench
 * sweep captured: an in-situ measurement against a meter found the old top
 * bench points (127/140/150 ms) badly under-reading.  The top anchor is
 * therefore a field measurement taken at the charger's fully-saturated
 * constant-voltage terminal (metered 8.70 V, median 148 ms); the 7.0 V and
 * below points are the bench sweep, which still reads correct in situ (a
 * separate 6.8 V <-> 177 ms check landed dead-on).  Only one anchor is used at
 * the very top on purpose: dt/dV collapses to ~20 ms/V up here, so a second
 * nearby point would sit inside the ~5 ms sample noise and manufacture a
 * spurious slope rather than refine it.
 */
struct wm8505_batt_point {
	unsigned int charge_ms;
	unsigned int uv;
};

static const struct wm8505_batt_point wm8505_batt_curve[] = {
	{ 148, 8700000 },	/* metered at the saturated CV charge terminal */
	{ 167, 7000000 },
	{ 192, 6500000 },
	{ 230, 6000000 },
};

/*
 * State-of-charge comes from an open-circuit-voltage table supplied by the
 * device tree "monitored-battery" node (a "simple-battery"), interpolated by
 * the power-supply core.  The reference board provisions the vendor's own
 * breakpoints (the u-boot "battvoltlist": a 2S Li-ion curve, 0..100 % in 10 %
 * steps) via ocv-capacity-table-0.  Keeping the profile in DT means a board
 * with a different pack can override it without touching the driver.  There is
 * no thermistor, so a nominal temperature is used for the lookup.
 */
#define WM8505_BATT_OCV_TEMP		20	/* deg C, nominal (no sensor) */

struct wm8505_batt {
	struct device *dev;
	struct gpio_desc *sense_gpio;
	struct gpio_desc *low_bat_gpio;
	struct gpio_desc *ac_gpio;
	struct gpio_desc *stat_gpio;
	struct power_supply *psy;	/* battery */
	struct power_supply *ac_psy;	/* mains, only if ac_gpio present */
	struct power_supply_battery_info *info;	/* OCV table, design voltages */
	struct delayed_work poll_work;

	struct mutex lock;	/* protects the cached readings below */
	bool present;
	bool low_bat;		/* UVLO asserted */
	bool ac_online;		/* AC adapter present (for edge detection) */
	int voltage_uv;
	int capacity;		/* percent, or -1 if unknown */
	int last_status;	/* last logged status (for edge detection) */
};

/* Take one RC-slope sample: returns the charge time in ms, or -ETIMEDOUT. */
static int wm8505_batt_sample(struct wm8505_batt *batt)
{
	ktime_t start;
	s64 elapsed_ms;

	/* Drain the capacitor. */
	gpiod_direction_output(batt->sense_gpio, 0);
	msleep(WM8505_BATT_DISCHARGE_MS);

	/*
	 * Release and busy-poll the charge up to the logic threshold.  The
	 * charge time is only ~130-230 ms and small differences map to large
	 * voltage differences, so we spin rather than sleep: this SoC's timers
	 * are coarse enough that even usleep_range(500, 1500) overshoots the
	 * crossing by tens of milliseconds, which would badly skew the reading.
	 * The loop is fully preemptible (this runs from a workqueue), and the
	 * vendor firmware times the same way.
	 */
	gpiod_direction_input(batt->sense_gpio);
	start = ktime_get();
	for (;;) {
		if (gpiod_get_value(batt->sense_gpio))
			return (int)ktime_to_ms(ktime_sub(ktime_get(), start));

		elapsed_ms = ktime_to_ms(ktime_sub(ktime_get(), start));
		if (elapsed_ms >= WM8505_BATT_CHARGE_TMO_MS)
			return -ETIMEDOUT;

		cpu_relax();
	}
}

static int wm8505_batt_cmp(const void *a, const void *b)
{
	return *(const int *)a - *(const int *)b;
}

/*
 * Interpolate the calibration curve: charge time (ms) -> microvolts.
 *
 * Beyond either end of the curve the terminal segment's slope is extrapolated
 * rather than clamped, so voltage_now is not pinned at the calibrated limits.
 * This matters most at the top: the charger on this board does not reliably
 * terminate its constant-voltage phase, and has been observed pushing the pack
 * well above the 8.70 V top anchor (8.8 V+).  A clamped reading would hide that
 * over-voltage entirely; extrapolating lets voltage_now climb so the runaway is
 * visible.  The trade-off is noise: dt/dV collapses to ~20 ms/V up here, so the
 * ~5 ms sample jitter becomes ~0.2-0.4 V of jitter on the extrapolated reading,
 * and readings outside the 6.0-8.7 V bench-verified range are directional, not
 * precise.  The UVLO line (low-bat) remains the authoritative low-end trip.
 */
static int wm8505_batt_uv_from_ms(unsigned int ms)
{
	const struct wm8505_batt_point *c = wm8505_batt_curve;
	size_t n = ARRAY_SIZE(wm8505_batt_curve);
	s64 span_ms, span_uv, pos;
	size_t i;

	/*
	 * Select the segment whose slope applies.  Inside the curve this is the
	 * bracketing segment; below c[0] it stays at segment 0 and above
	 * c[n-1] it stops at the last segment (n-2), so the same point-slope
	 * formula extrapolates past both ends.
	 */
	for (i = 0; i + 2 < n && ms > c[i + 1].charge_ms; i++)
		;

	span_ms = (s64)c[i + 1].charge_ms - c[i].charge_ms;
	span_uv = (s64)c[i + 1].uv - c[i].uv;	/* negative: V falls as ms rises */
	pos = (s64)ms - c[i].charge_ms;

	return c[i].uv + (int)div_s64(span_uv * pos, span_ms);
}

/* Caller must hold batt->lock. */
static int wm8505_batt_level(struct wm8505_batt *batt)
{
	if (batt->low_bat)
		return POWER_SUPPLY_CAPACITY_LEVEL_CRITICAL;
	if (!batt->present || batt->capacity < 0)
		return POWER_SUPPLY_CAPACITY_LEVEL_UNKNOWN;
	if (batt->capacity <= 5)
		return POWER_SUPPLY_CAPACITY_LEVEL_CRITICAL;
	if (batt->capacity <= 20)
		return POWER_SUPPLY_CAPACITY_LEVEL_LOW;
	if (batt->capacity < 95)
		return POWER_SUPPLY_CAPACITY_LEVEL_NORMAL;
	return POWER_SUPPLY_CAPACITY_LEVEL_FULL;
}

/* Live charge status, per the vendor battdrvr logic.  Caller holds the lock. */
static int wm8505_batt_status(struct wm8505_batt *batt)
{
	if (!batt->present)
		return POWER_SUPPLY_STATUS_UNKNOWN;

	/* No adapter-detect line: cannot distinguish charge state. */
	if (!batt->ac_gpio)
		return POWER_SUPPLY_STATUS_UNKNOWN;

	if (!gpiod_get_value(batt->ac_gpio))
		return POWER_SUPPLY_STATUS_DISCHARGING;

	/* Adapter present: the STAT line means charging vs. charge-complete. */
	if (batt->stat_gpio && gpiod_get_value(batt->stat_gpio))
		return POWER_SUPPLY_STATUS_CHARGING;

	return POWER_SUPPLY_STATUS_FULL;
}

static void wm8505_batt_poll(struct work_struct *work)
{
	struct wm8505_batt *batt = container_of(to_delayed_work(work),
						struct wm8505_batt, poll_work);
	int samples[WM8505_BATT_SAMPLES];
	bool present, low_bat, low_bat_prev, ac_online;
	bool batt_changed, ac_changed, status_changed;
	int old_level, new_level, status;
	int i, n = 0, uv = 0, capacity = -1;

	for (i = 0; i < WM8505_BATT_SAMPLES; i++) {
		int ms = wm8505_batt_sample(batt);

		if (ms >= 0)
			samples[n++] = ms;
	}

	/*
	 * PRESENT is best-effort: a charge timeout means the line never rose,
	 * which we treat as "no pack".  Note the converse is not reliable -- a
	 * disconnected pack can still leave the sense node floating high enough
	 * to RC-charge and read a bogus ~8 V, so PRESENT can false-positive.
	 * The vendor driver has the same limitation; don't overclaim here.
	 */
	present = n > 0;
	if (present) {
		sort(samples, n, sizeof(samples[0]), wm8505_batt_cmp, NULL);
		uv = wm8505_batt_uv_from_ms(samples[n / 2]);
		if (batt->info) {
			capacity = power_supply_batinfo_ocv2cap(batt->info, uv,
								WM8505_BATT_OCV_TEMP);
			if (capacity < 0)
				capacity = -1;
		}
		dev_dbg(batt->dev,
			"raw charge ms: min=%d median=%d max=%d (n=%d) -> %d uV, %d%%\n",
			samples[0], samples[n / 2], samples[n - 1], n,
			uv, capacity);
	}

	/* Sample the status lines (no GPIO IRQs on this SoC). */
	low_bat = batt->low_bat_gpio && gpiod_get_value(batt->low_bat_gpio);
	ac_online = batt->ac_gpio && gpiod_get_value(batt->ac_gpio);

	mutex_lock(&batt->lock);
	old_level = wm8505_batt_level(batt);
	low_bat_prev = batt->low_bat;
	batt->present = present;
	batt->low_bat = low_bat;
	batt->voltage_uv = uv;
	batt->capacity = capacity;
	new_level = wm8505_batt_level(batt);
	ac_changed = batt->ac_online != ac_online;
	batt->ac_online = ac_online;
	status = wm8505_batt_status(batt);
	status_changed = status != batt->last_status;
	batt->last_status = status;
	/* STATUS follows the AC line, so an AC edge also changes the battery. */
	batt_changed = old_level != new_level || ac_changed;
	mutex_unlock(&batt->lock);

	/* Announce the two calibration transitions (and their complements). */
	if (low_bat && !low_bat_prev)
		dev_warn(batt->dev, "battery-low / UVLO asserted (~%d mV)\n",
			 uv / 1000);
	else if (!low_bat && low_bat_prev)
		dev_info(batt->dev, "battery-low / UVLO cleared (~%d mV)\n",
			 uv / 1000);

	if (status_changed) {
		if (status == POWER_SUPPLY_STATUS_FULL)
			dev_dbg(batt->dev, "charge complete (STAT deasserted, ~%d mV)\n",
				uv / 1000);
		else if (status == POWER_SUPPLY_STATUS_CHARGING)
			dev_dbg(batt->dev, "charging (~%d mV)\n", uv / 1000);
		else if (status == POWER_SUPPLY_STATUS_DISCHARGING)
			dev_dbg(batt->dev, "discharging / on battery (~%d mV)\n",
				uv / 1000);
	}

	if (batt_changed)
		power_supply_changed(batt->psy);
	if (ac_changed && batt->ac_psy)
		power_supply_changed(batt->ac_psy);

	schedule_delayed_work(&batt->poll_work,
			      msecs_to_jiffies(WM8505_BATT_POLL_MS));
}

static int wm8505_batt_get_property(struct power_supply *psy,
				    enum power_supply_property psp,
				    union power_supply_propval *val)
{
	struct wm8505_batt *batt = power_supply_get_drvdata(psy);
	int ret = 0;

	mutex_lock(&batt->lock);
	switch (psp) {
	case POWER_SUPPLY_PROP_STATUS:
		val->intval = wm8505_batt_status(batt);
		break;
	case POWER_SUPPLY_PROP_PRESENT:
		val->intval = batt->present;
		break;
	case POWER_SUPPLY_PROP_HEALTH:
		val->intval = batt->low_bat ? POWER_SUPPLY_HEALTH_DEAD :
					      POWER_SUPPLY_HEALTH_GOOD;
		break;
	case POWER_SUPPLY_PROP_TECHNOLOGY:
		val->intval = POWER_SUPPLY_TECHNOLOGY_LION;
		break;
	case POWER_SUPPLY_PROP_VOLTAGE_NOW:
		if (!batt->present)
			ret = -ENODATA;
		else
			val->intval = batt->voltage_uv;
		break;
	case POWER_SUPPLY_PROP_CAPACITY:
		if (!batt->present || batt->capacity < 0)
			ret = -ENODATA;
		else
			val->intval = batt->capacity;
		break;
	case POWER_SUPPLY_PROP_CAPACITY_LEVEL:
		val->intval = wm8505_batt_level(batt);
		break;
	case POWER_SUPPLY_PROP_VOLTAGE_MIN_DESIGN:
		if (batt->info && batt->info->voltage_min_design_uv > 0)
			val->intval = batt->info->voltage_min_design_uv;
		else
			ret = -ENODATA;
		break;
	case POWER_SUPPLY_PROP_VOLTAGE_MAX_DESIGN:
		if (batt->info && batt->info->voltage_max_design_uv > 0)
			val->intval = batt->info->voltage_max_design_uv;
		else
			ret = -ENODATA;
		break;
	case POWER_SUPPLY_PROP_SCOPE:
		val->intval = POWER_SUPPLY_SCOPE_SYSTEM;
		break;
	default:
		ret = -EINVAL;
		break;
	}
	mutex_unlock(&batt->lock);

	return ret;
}

static enum power_supply_property wm8505_batt_props[] = {
	POWER_SUPPLY_PROP_STATUS,
	POWER_SUPPLY_PROP_PRESENT,
	POWER_SUPPLY_PROP_HEALTH,
	POWER_SUPPLY_PROP_TECHNOLOGY,
	POWER_SUPPLY_PROP_VOLTAGE_NOW,
	POWER_SUPPLY_PROP_VOLTAGE_MIN_DESIGN,
	POWER_SUPPLY_PROP_VOLTAGE_MAX_DESIGN,
	POWER_SUPPLY_PROP_CAPACITY,
	POWER_SUPPLY_PROP_CAPACITY_LEVEL,
	POWER_SUPPLY_PROP_SCOPE,
};

static const struct power_supply_desc wm8505_batt_desc = {
	.name		= "wm8505-battery",
	.type		= POWER_SUPPLY_TYPE_BATTERY,
	.properties	= wm8505_batt_props,
	.num_properties	= ARRAY_SIZE(wm8505_batt_props),
	.get_property	= wm8505_batt_get_property,
};

static int wm8505_ac_get_property(struct power_supply *psy,
				  enum power_supply_property psp,
				  union power_supply_propval *val)
{
	struct wm8505_batt *batt = power_supply_get_drvdata(psy);

	switch (psp) {
	case POWER_SUPPLY_PROP_ONLINE:
		val->intval = batt->ac_gpio && gpiod_get_value(batt->ac_gpio);
		break;
	default:
		return -EINVAL;
	}

	return 0;
}

static enum power_supply_property wm8505_ac_props[] = {
	POWER_SUPPLY_PROP_ONLINE,
};

static const struct power_supply_desc wm8505_ac_desc = {
	.name		= "wm8505-ac",
	.type		= POWER_SUPPLY_TYPE_MAINS,
	.properties	= wm8505_ac_props,
	.num_properties	= ARRAY_SIZE(wm8505_ac_props),
	.get_property	= wm8505_ac_get_property,
};

/* The mains supply feeds the battery; expose the link for power_supply core. */
static char *wm8505_ac_supplied_to[] = {
	"wm8505-battery",
};

static int wm8505_batt_probe(struct platform_device *pdev)
{
	struct device *dev = &pdev->dev;
	struct power_supply_config cfg = {};
	struct wm8505_batt *batt;
	int ret;

	batt = devm_kzalloc(dev, sizeof(*batt), GFP_KERNEL);
	if (!batt)
		return -ENOMEM;

	batt->dev = dev;
	batt->capacity = -1;
	batt->last_status = -1;	/* force the first poll to log the status */
	mutex_init(&batt->lock);
	INIT_DELAYED_WORK(&batt->poll_work, wm8505_batt_poll);

	batt->sense_gpio = devm_gpiod_get(dev, "sense", GPIOD_IN);
	if (IS_ERR(batt->sense_gpio))
		return dev_err_probe(dev, PTR_ERR(batt->sense_gpio),
				     "failed to get sense GPIO\n");

	batt->low_bat_gpio = devm_gpiod_get_optional(dev, "low-bat", GPIOD_IN);
	if (IS_ERR(batt->low_bat_gpio))
		return dev_err_probe(dev, PTR_ERR(batt->low_bat_gpio),
				     "failed to get low-bat GPIO\n");

	batt->ac_gpio = devm_gpiod_get_optional(dev, "ac-detect", GPIOD_IN);
	if (IS_ERR(batt->ac_gpio))
		return dev_err_probe(dev, PTR_ERR(batt->ac_gpio),
				     "failed to get ac-detect GPIO\n");

	batt->stat_gpio = devm_gpiod_get_optional(dev, "charge-status", GPIOD_IN);
	if (IS_ERR(batt->stat_gpio))
		return dev_err_probe(dev, PTR_ERR(batt->stat_gpio),
				     "failed to get charge-status GPIO\n");

	/* Latch the initial UVLO state before the supply becomes visible. */
	batt->low_bat = batt->low_bat_gpio &&
			gpiod_get_value(batt->low_bat_gpio);

	cfg.drv_data = batt;
	cfg.fwnode = dev_fwnode(dev);

	/* Register the mains supply first so the battery link resolves. */
	if (batt->ac_gpio) {
		struct power_supply_config ac_cfg = {
			.drv_data = batt,
			.supplied_to = wm8505_ac_supplied_to,
			.num_supplicants = ARRAY_SIZE(wm8505_ac_supplied_to),
		};

		batt->ac_psy = devm_power_supply_register(dev, &wm8505_ac_desc,
							  &ac_cfg);
		if (IS_ERR(batt->ac_psy))
			return dev_err_probe(dev, PTR_ERR(batt->ac_psy),
					     "failed to register mains supply\n");
	}

	batt->psy = devm_power_supply_register(dev, &wm8505_batt_desc, &cfg);
	if (IS_ERR(batt->psy))
		return dev_err_probe(dev, PTR_ERR(batt->psy),
				     "failed to register battery supply\n");

	/*
	 * Fetch the OCV table and design voltages from the "monitored-battery"
	 * DT node.  It is optional: without it voltage/status still work, only
	 * the capacity and design-voltage properties become unavailable.  The
	 * info is devm-allocated against the power supply, so it needs no
	 * explicit free.
	 */
	ret = power_supply_get_battery_info(batt->psy, &batt->info);
	if (ret == -EPROBE_DEFER)
		return ret;
	if (ret) {
		dev_warn(dev, "no battery info (%d); capacity unavailable\n", ret);
		batt->info = NULL;
	}

	platform_set_drvdata(pdev, batt);

	/* Kick off an initial reading immediately, then poll periodically. */
	schedule_delayed_work(&batt->poll_work, 0);

	return 0;
}

static void wm8505_batt_remove(struct platform_device *pdev)
{
	struct wm8505_batt *batt = platform_get_drvdata(pdev);

	/*
	 * The poll work re-arms itself, so disable it (rather than a plain
	 * cancel) to guarantee it cannot requeue during teardown.
	 */
	disable_delayed_work_sync(&batt->poll_work);
}

static const struct of_device_id wm8505_batt_of_match[] = {
	{ .compatible = "wm,wm8505-battery" },
	{}
};
MODULE_DEVICE_TABLE(of, wm8505_batt_of_match);

static struct platform_driver wm8505_batt_driver = {
	.driver = {
		.name		= "wm8505-battery",
		.of_match_table	= wm8505_batt_of_match,
	},
	.probe	= wm8505_batt_probe,
	.remove	= wm8505_batt_remove,
};
module_platform_driver(wm8505_batt_driver);

MODULE_AUTHOR("Brant Martin");
MODULE_DESCRIPTION("WonderMedia WM8505 GPIO RC-slope battery and charger meter");
MODULE_LICENSE("GPL");

"""Synthetic sensor logs for each fault, plus a summary the agent can read.

The logs are invented for testing (clearly labelled synthetic in the app).
One row per minute: temperatures in C, pressures in psig, currents in amps.
"""

from __future__ import annotations

import csv
import io
import random
import statistics

from . import knowledge

CHANNELS = [
    "return_air_c", "supply_air_c", "outdoor_air_c", "suction_psig",
    "discharge_psig", "compressor_amps", "outdoor_fan_amps", "room_sensor_c",
]

# (low, high) per channel while running in each condition. Anything not listed uses NORMAL.
NORMAL = {
    "return_air_c": (25.5, 27.5), "supply_air_c": (12.5, 15.0), "outdoor_air_c": (33, 37),
    "suction_psig": (122, 140), "discharge_psig": (350, 410), "compressor_amps": (5.5, 7.5),
    "outdoor_fan_amps": (0.45, 0.7),
}

PROFILES = {
    "normal": {},
    "clogged_air_filter": {
        "supply_air_c": (6.0, 9.0), "suction_psig": (92, 108), "discharge_psig": (310, 345),
        "compressor_amps": (5.0, 6.0),
    },
    "condenser_fan_failure": {
        "outdoor_fan_amps": (0.0, 0.04), "discharge_psig": (500, 585), "compressor_amps": (10.5, 13.0),
        "supply_air_c": (17.5, 21.0), "return_air_c": (28.0, 30.0),
    },
    "low_refrigerant": {
        "suction_psig": (62, 85), "discharge_psig": (225, 280), "compressor_amps": (3.4, 4.4),
        "supply_air_c": (18.0, 21.0), "return_air_c": (27.5, 29.5),
    },
    "dirty_condenser_coil": {
        "discharge_psig": (460, 520), "compressor_amps": (9.0, 10.5), "supply_air_c": (15.0, 17.5),
    },
    "failed_compressor_capacitor": {
        "compressor_amps": (0.0, 0.0), "suction_psig": (285, 305), "discharge_psig": (285, 305),
        "return_air_c": (29.0, 31.0), "supply_air_c": (28.5, 30.5),
    },
    "faulty_room_thermistor": {
        "compressor_amps": (0.0, 0.0), "outdoor_fan_amps": (0.0, 0.02), "suction_psig": (285, 305),
        "discharge_psig": (285, 305), "return_air_c": (29.0, 31.0), "supply_air_c": (28.5, 30.5),
    },
}


def _blend(normal: tuple, fault: tuple, severity: float) -> tuple:
    """Move a range from normal towards the fault range. severity 1 = full fault."""
    return tuple(n + severity * (f - n) for n, f in zip(normal, fault))


def generate(fault: str, minutes: int = 30, seed: int | None = None, severity: float = 1.0) -> list[dict]:
    """One reading per minute. `severity` (0-1) makes mild faults look closer to normal."""
    rng = random.Random(seed)
    ranges = dict(NORMAL)
    for ch, fault_range in PROFILES[fault].items():
        ranges[ch] = _blend(NORMAL[ch], fault_range, severity)

    # Unit-to-unit variation: no two installations read the same when healthy
    # (pipe length, charge, sensor placement, how hot the day is).
    outdoor_shift = rng.gauss(0, 3.0)
    for ch, (lo, hi) in list(ranges.items()):
        n_lo, n_hi = NORMAL[ch]
        shift = rng.gauss(0, 0.55 * (n_hi - n_lo))
        if ch in ("discharge_psig",) and hi > 0:
            shift += outdoor_shift * 7       # hotter outdoors -> higher head pressure
        if ch == "outdoor_air_c":
            shift += outdoor_shift
        if hi == lo == 0:                    # a component that is off stays off
            continue
        ranges[ch] = (max(lo + shift, 0), max(hi + shift, 0))
    rows = []
    for m in range(minutes):
        row = {"minute": m}
        for ch in CHANNELS:
            if ch == "room_sensor_c":
                continue
            lo, hi = ranges[ch]
            row[ch] = round(rng.uniform(lo, hi), 2)
        # Controller's own room reading: normally tracks return air.
        if fault == "faulty_room_thermistor":
            offset = -(3 + 7 * severity) + rng.uniform(-1, 1)
        else:
            offset = rng.uniform(-0.8, 0.8)
        row["room_sensor_c"] = round(row["return_air_c"] + offset, 2)
        # Locked-rotor start attempts: big current spike every ~5 minutes.
        if fault == "failed_compressor_capacitor" and m % 5 == 0:
            row["compressor_amps"] = round(rng.uniform(24, 30), 1)
        # High-pressure cutout: compressor drops out for a couple of minutes now and then.
        if fault == "condenser_fan_failure" and m % 8 in (6, 7):
            row["compressor_amps"] = 0.0
        rows.append(row)
    return rows


def to_csv(rows: list[dict]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def from_csv(text: str) -> list[dict]:
    rows = []
    for raw in csv.DictReader(io.StringIO(text)):
        row = {}
        for k, v in raw.items():
            if v in ("", None):
                continue
            try:
                row[k] = float(v)
            except ValueError:
                pass
        rows.append(row)
    return rows


def summarize(rows: list[dict]) -> dict:
    """Median/min/max per channel, derived values, and flags against normal ranges."""
    ranges = knowledge.load()["normal_ranges"]
    present = [ch for ch in CHANNELS if any(ch in r for r in rows)]
    stats = {}
    for ch in present:
        vals = [r[ch] for r in rows if ch in r]
        stats[ch] = {"median": round(statistics.median(vals), 2), "min": min(vals), "max": max(vals)}

    derived = {}
    if "return_air_c" in stats and "supply_air_c" in stats:
        derived["delta_t_c"] = round(stats["return_air_c"]["median"] - stats["supply_air_c"]["median"], 2)
    if "room_sensor_c" in stats and "return_air_c" in stats:
        derived["room_sensor_offset_c"] = round(
            stats["room_sensor_c"]["median"] - stats["return_air_c"]["median"], 2
        )
    if "compressor_amps" in stats:
        amps = [r["compressor_amps"] for r in rows if "compressor_amps" in r]
        derived["compressor_off_minutes"] = sum(1 for a in amps if a < 0.5)
        derived["compressor_spikes_over_20a"] = sum(1 for a in amps if a > 20)

    flags = {}
    for key, (lo, hi) in ranges.items():
        value = derived.get(key, stats.get(key, {}).get("median"))
        if value is None:
            continue
        if value < lo:
            flags[key] = f"LOW ({value} vs normal {lo}-{hi})"
        elif value > hi:
            flags[key] = f"HIGH ({value} vs normal {lo}-{hi})"

    return {
        "minutes": len(rows),
        "channels_present": present,
        "channels_missing": [ch for ch in CHANNELS if ch not in present],
        "stats": stats,
        "derived": derived,
        "out_of_range": flags,
    }


def rule_scores(summary: dict) -> dict[str, float]:
    """Simple hand-written scoring from sensors alone. Used as the comparison baseline
    and as the offline (mock) answer. Not part of the agent's reasoning."""
    s = {k: v["median"] for k, v in summary["stats"].items()}
    d = summary["derived"]
    score = {f: 0.0 for f in knowledge.fault_ids()}

    def has(*keys):
        return all(k in s or k in d for k in keys)

    if has("delta_t_c") and d["delta_t_c"] > 16:
        score["clogged_air_filter"] += 2
    if has("suction_psig") and 88 <= s["suction_psig"] < 112:
        score["clogged_air_filter"] += 1
    if has("outdoor_fan_amps") and s["outdoor_fan_amps"] < 0.1 and d.get("compressor_off_minutes", 0) < 20:
        score["condenser_fan_failure"] += 3
    if has("discharge_psig") and s["discharge_psig"] > 450:
        score["condenser_fan_failure"] += 1
        score["dirty_condenser_coil"] += 1
    if has("outdoor_fan_amps", "discharge_psig") and s["outdoor_fan_amps"] > 0.3 and s["discharge_psig"] > 450:
        score["dirty_condenser_coil"] += 2
    if has("suction_psig") and s["suction_psig"] < 88:
        score["low_refrigerant"] += 2
    if has("discharge_psig") and s["discharge_psig"] < 290:
        score["low_refrigerant"] += 1
    if d.get("compressor_spikes_over_20a", 0) >= 2:
        score["failed_compressor_capacitor"] += 3
    if has("room_sensor_offset_c") and d["room_sensor_offset_c"] < -4:
        score["faulty_room_thermistor"] += 3

    total = sum(score.values())
    if total == 0:
        # Nothing out of place: call it normal if the profile has that class.
        if "normal" in score:
            return {f: (1.0 if f == "normal" else 0.0) for f in score}
        return {f: round(1 / len(score), 3) for f in score}
    return {f: round(v / total, 3) for f, v in score.items()}

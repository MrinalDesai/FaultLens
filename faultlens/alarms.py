"""Alarm history: the unit's own fault log, as a technician reads it from the display.

Real split ACs keep a short history of protection trips and error codes. Each manufacturer
uses its own codes, so FaultLens uses a small demo code table (below), labelled as such.
Every unit's log is SIMULATED from its fault and severity, like the other signals.

What makes the log useful, and realistic, is that it is indirect evidence: a high-pressure
trip says "the outdoor side is too hot", not "the fan is broken" (a dirty coil does the same),
and some faults (a sensor that has slipped onto the coil, an early leak) leave no code at all.
"""

from __future__ import annotations

import csv
import io
import random
from datetime import datetime, timedelta

# Demo code table. Not any manufacturer's codes.
CODES = {
    "P1": "Indoor coil freeze protection: compressor stopped because the indoor coil was below 0 C",
    "E4": "Outdoor high-temperature / high-pressure protection: compressor stopped because the outdoor coil was too hot",
    "E5": "Compressor start failure: current too high at start, compressor stopped and will retry",
    "E6": "Low-pressure protection: suction pressure too low",
    "F1": "Room temperature sensor open or short circuit",
    "L0": "Filter cleaning reminder (timer, every 250 running hours, not a fault)",
}
CODE_TABLE_NOTE = "FaultLens demo codes. Each manufacturer uses its own codes; check the unit's service manual."
END = datetime(2026, 10, 28, 18, 0)


def _times(rng: random.Random, n: int, hours: tuple[int, int], days: int = 7) -> list[datetime]:
    out = []
    for _ in range(n):
        day = END - timedelta(days=rng.randrange(days))
        out.append(day.replace(hour=rng.randint(*hours), minute=rng.randrange(60)))
    return out


def generate(condition: str, seed: int = 0, severity: float = 1.0, variant: str = "", days: int = 7) -> list[dict]:
    """The last `days` days of the unit's fault log, oldest first."""
    rng = random.Random(seed * 7 + 11)
    s = max(0.0, min(1.0, severity))
    events: list[tuple[datetime, str]] = []

    def add(code, n, hours=(8, 22)):
        events.extend((t, code) for t in _times(rng, n, hours, days))

    if rng.random() < 0.4:                       # harmless reminder on any unit
        add("L0", 1)
    if condition == "clogged_air_filter":
        add("P1", round(1 + 6 * s))
        if rng.random() < 0.8:
            add("L0", 1)                             # the reminder was ignored
    elif condition == "condenser_fan_failure":
        add("E4", round(3 + 12 * s), (11, 18))       # hot hours: trips every few minutes once the coil heats up
    elif condition == "dirty_condenser_coil":
        add("E4", round(1 + 4 * s * s), (13, 16))    # only on the hottest afternoons, and only when bad
    elif condition == "low_refrigerant":
        if s > 0.6:
            add("E6", round(1 + 4 * (s - 0.6) / 0.4))
        if rng.random() < 0.5 * s:
            add("P1", 1)                             # the starved inlet can freeze too
    elif condition == "failed_compressor_capacitor":
        # Each start attempt fails; the unit retries every few minutes while it is switched on.
        start = END - timedelta(hours=rng.randint(2, 30))
        for k in range(round(6 + 14 * s)):
            events.append((start + timedelta(minutes=3 * k + rng.randrange(2)), "E5"))
    elif condition == "faulty_room_thermistor":
        if variant != "loose":                       # a slipped sensor reads a real (wrong) temperature: no code
            add("F1", 1 + rng.randrange(3))
    events.sort()
    return [{"time": t.strftime("%Y-%m-%d %H:%M"), "code": c, "meaning": CODES[c]} for t, c in events]


def summarize(rows: list[dict]) -> dict:
    """Counts per code, first and last time, plus the code table note."""
    by = {}
    for r in rows:
        e = by.setdefault(r["code"], {"code": r["code"], "meaning": CODES.get(r["code"], r.get("meaning", "")),
                                      "count": 0, "first": r["time"], "last": r["time"]})
        e["count"] += 1
        e["last"] = r["time"]
    hours = {}
    for r in rows:
        if r["code"] != "L0":
            h = int(r["time"][11:13])
            hours[h] = hours.get(h, 0) + 1
    return {"codes": sorted(by.values(), key=lambda e: -e["count"]),
            "total_fault_codes": sum(e["count"] for e in by.values() if e["code"] != "L0"),
            "busiest_hours": sorted(hours, key=lambda h: -hours[h])[:3],
            "note": CODE_TABLE_NOTE + " The log is a unit's own protection record: it shows what the unit "
                    "protected itself from, not which part failed. No fault codes does not mean no fault."}


def describe(rows: list[dict]) -> str:
    """One plain-language line for the game log."""
    sm = summarize(rows)
    if not rows:
        return "The unit's fault history is empty: no codes in the last 7 days."
    parts = [f"{e['code']} x{e['count']} (last {e['last'][5:]})" for e in sm["codes"]]
    text = "Fault history, last 7 days: " + ", ".join(parts) + "."
    if not sm["total_fault_codes"]:
        text += " Only the filter reminder, no fault codes."
    return text


def to_csv(rows: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["time", "code", "meaning"])
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def from_csv(text: str) -> list[dict]:
    rows = list(csv.DictReader(io.StringIO(text)))
    out = []
    for r in rows:
        code = (r.get("code") or "").strip().upper()
        if code:
            out.append({"time": (r.get("time") or "").strip(), "code": code,
                        "meaning": CODES.get(code, (r.get("meaning") or "").strip() or "Unknown code")})
    return out


def is_alarm_csv(text: str) -> bool:
    head = text.lstrip("﻿").splitlines()[0].lower() if text.strip() else ""
    return "code" in [h.strip() for h in head.split(",")]


if __name__ == "__main__":
    for c in ["normal", "clogged_air_filter", "condenser_fan_failure", "low_refrigerant", "dirty_condenser_coil",
              "failed_compressor_capacitor", "faulty_room_thermistor"]:
        for sev in (0.3, 1.0):
            print(f"{c:28s} sev {sev}: {describe(generate(c, seed=5, severity=sev))}")

"""How well each conclusion cites its evidence.

For every piece of evidence FaultLens gives (evidence_for and evidence_against, every cause),
check automatically that it refers only to what FaultLens actually received:

  numbers    every number it quotes must exist in what it was given (a reading at some minute,
             a median, min or max, a derived value, a normal range, an alarm count or a number in
             the technician's notes), within rounding
  channels   it must not quote a sensor channel that was held back from it
  signals    it must not describe a thermal image, sound, spectrogram or photo it was not shown
  codes      every alarm code it names must be in the alarm history it was given

Saying that something is missing ("no fan current was provided") is correct handling, not a
violation. The score is the share of evidence items with no violation.
"""

from __future__ import annotations

import re

from faultlens import alarms as alarm_mod
from faultlens import knowledge, sensors

ABSENT_BEFORE = r"\b(no|missing|without|lack of|absent|unavailable|not provided|not available|not shown|not given|cannot see|can't see)\b(\W+\w+){0,4}?\W+"
ABSENT_AFTER = r"(\W+\w+){0,4}?\W+(was|were|is|are)?\s*(not provided|not available|not shown|not given|missing|unavailable|absent|not measured|not logged)\b"


def _says_absent(t: str, word: str) -> bool:
    """True when the text says this thing was not available, rather than quoting it."""
    w = re.escape(word)
    return bool(re.search(ABSENT_BEFORE + w, t) or re.search(w + ABSENT_AFTER, t))
NUMBER = re.compile(r"(?<![A-Za-z_#])-?\d+(?:\.\d+)?")
CODE = re.compile(r"\b([EPFL]\d)\b")

CHANNEL_WORDS = {
    "supply_air_c": ["supply air", "supply_air", "louvre", "outlet air", "air leaving"],
    "return_air_c": ["return air", "return_air", "intake air"],
    "outdoor_air_c": ["outdoor air", "outdoor_air", "ambient", "outside air"],
    "suction_psig": ["suction"],
    "discharge_psig": ["discharge pressure", "discharge_psig", "head pressure", "high side"],
    "compressor_amps": ["compressor amps", "compressor current", "compressor_amps", "compressor draw", "compressor amp"],
    "outdoor_fan_amps": ["fan amps", "fan current", "outdoor_fan_amps", "fan motor current", "fan draw", "fan amp"],
    "room_sensor_c": ["room sensor", "room_sensor", "sensor reading", "sensor reads"],
}
SIGNAL_WORDS = {
    "thermal": ["thermal", "infrared", "hot spot", "heat map"],
    "sound": ["spectrogram", "recording", "tone", "audio"],
    "photo": ["photo", "frame", "picture", "visible in"],
}


def _numbers_given(sensor_rows, notes, alarm_rows, pattern_match) -> list[float]:
    vals = []
    for r in sensor_rows:
        vals += [float(v) for k, v in r.items() if isinstance(v, (int, float))]
    if sensor_rows:
        sm = sensors.summarize(sensor_rows)
        for st in sm["stats"].values():
            vals += [float(v) for v in st.values()]
        vals += [float(v) for v in sm["derived"].values() if isinstance(v, (int, float))]
        for txt in sm.get("out_of_range", {}).values():
            vals += [float(x) for x in NUMBER.findall(txt)]
        vals += [float(sm["minutes"])]
    # Numbers in the fault guides are fair to quote: the agent reads them through its guide tool.
    import json as _json
    vals += [float(x) for x in NUMBER.findall(_json.dumps(knowledge.load()))]
    vals += [float(x) for x in NUMBER.findall(_json.dumps(knowledge.sources()))]
    for rng in (knowledge.load().get("normal_ranges") or {}).values():
        if isinstance(rng, (list, tuple)):
            vals += [float(x) for x in rng if isinstance(x, (int, float))]
    for n in notes:
        vals += [float(x) for x in NUMBER.findall(n)]
    if alarm_rows:
        sm = alarm_mod.summarize(alarm_rows)
        vals += [float(c["count"]) for c in sm["codes"]] + [float(sm["total_fault_codes"]), 7.0]
        for r in alarm_rows:
            vals += [float(x) for x in NUMBER.findall(r["time"])]
    if pattern_match:
        for f in pattern_match.get("fused", []):
            vals += [f["probability"], f["probability"] * 100]
        for s in pattern_match.get("signals", {}).values():
            vals += [s["confidence"], s["confidence"] * 100]
            for k in s.get("key_measurements", []):
                vals += [float(k["value"]), float(k["healthy_typical"])]
    return vals


def _known(x: float, vals: list[float]) -> bool:
    return any(abs(x - v) <= max(0.6, 0.03 * abs(v)) for v in vals)


def check_item(text: str, given: dict) -> list[str]:
    """Violations in one evidence item (empty list = grounded)."""
    t = text.lower()
    out = []
    for x in NUMBER.findall(text):
        v = float(x)
        if abs(v) in (0, 1, 2, 3) and "." not in x:   # small counts ("2 of 3 signals") carry no reading
            continue
        if not _known(v, given["numbers"]):
            out.append(f"number {x} not in the evidence")
    for ch in given["hidden_channels"]:
        hits = [w for w in CHANNEL_WORDS.get(ch, []) if w in t]
        if hits and not any(_says_absent(t, w) for w in hits):
            out.append(f"quotes {ch}, which was held back")
    for sig, words in SIGNAL_WORDS.items():
        hits = [w for w in words if w in t]
        if not given["signals"].get(sig) and hits and not any(_says_absent(t, w) for w in hits):
            out.append(f"describes a {sig} it was not shown")
    for code in CODE.findall(text):
        if code not in given["codes"] and not _says_absent(t, code.lower()):
            out.append(f"alarm code {code} not in the alarm history")
    return out


def check_assessment(assessment: dict, sensor_rows=(), notes=(), alarm_rows=(), alarms_given=False,
                     hidden_channels=(), signals=None, pattern_match=None) -> dict:
    given = {"numbers": _numbers_given(list(sensor_rows), list(notes), list(alarm_rows), pattern_match),
             "hidden_channels": [h for h in hidden_channels if h],
             "signals": signals or {},
             "codes": {r["code"] for r in alarm_rows} if alarms_given else set()}
    items, problems = 0, []
    for h in assessment.get("hypotheses", []):
        for key in ("evidence_for", "evidence_against"):
            for e in h.get(key) or []:
                items += 1
                v = check_item(str(e), given)
                if v:
                    problems.append({"cause": h.get("fault_id"), "evidence": str(e)[:200], "problems": v})
    return {"items": items, "grounded": items - len(problems), "problems": problems}


def summarize(checks: list[dict]) -> dict:
    items = sum(c["items"] for c in checks)
    grounded = sum(c["grounded"] for c in checks)
    return {"evidence_items": items,
            "evidence_grounded": round(grounded / items, 3) if items else None,
            "assessments_fully_grounded": round(sum(not c["problems"] for c in checks) / len(checks), 3) if checks else None}

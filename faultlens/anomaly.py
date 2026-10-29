"""Anomaly lab: inject a fault into a stretch of a healthy signal, then find it.

A healthy unit runs; partway through, a fault creeps in (ramping up, the way real faults start),
stays, and is then repaired. The pattern model slides a window along the signal and labels each
stretch. Nothing is called on an API: this is FaultLens's own trained model.

Scored honestly against the true stretch:
  right fault     a detected stretch with the injected fault overlaps the true one
  coverage        share of the true stretch labelled with the right fault
  delay           how long after the fault started it was first caught (a mild start is hard)
  false alarms    stretches labelled as a fault where there was none, or with the wrong fault
"""

from __future__ import annotations

import random

import numpy as np

from . import features, pattern, render, sensors, simulate

KEY_READING = {   # the sensor channel that shows each fault best
    "normal": "discharge_psig",
    "clogged_air_filter": "supply_air_c",
    "condenser_fan_failure": "discharge_psig",
    "low_refrigerant": "suction_psig",
    "dirty_condenser_coil": "discharge_psig",
    "failed_compressor_capacitor": "compressor_amps",
    "faulty_room_thermistor": "room_sensor_c",
}
SETUP = {   # length, window, step (seconds for sound, minutes for sensors)
    "sound": {"length": 60, "window": 10, "step": 2, "unit": "s", "hold": 2, "sure": 0.7},
    "sensors": {"length": 360, "window": 30, "step": 10, "unit": "min", "hold": 2, "sure": 0.8},
}
# "sure" and "hold" were tuned on separate tuning runs (seeds 5000-5007), not on the reported runs.
# "hold": an alarm (or its clearing) is only raised once the same answer holds for this many
# windows in a row, as alarm systems do, so one stray window does not become an alarm.
CHUNK = 10   # sound is built from back-to-back 10 s recordings, the length the model learned from


def _weights(length: float, rng: random.Random, n: int) -> tuple[np.ndarray, tuple[float, float]]:
    """Fault weight over time: 0 healthy, ramps up, holds, drops to 0 at the repair."""
    start = length * rng.uniform(0.22, 0.35)
    full = start + length * rng.uniform(0.08, 0.14)
    end = length * rng.uniform(0.72, 0.82)
    t = np.linspace(0, length, n, endpoint=False)
    w = np.clip((t - start) / (full - start), 0, 1)
    w[t >= end] = 0.0
    return w, (round(start, 1), round(end, 1))


def build(fault: str, signal: str = "sensors", seed: int = 0, severity: float = 1.0) -> dict:
    """The stream with the fault injected, and the true fault stretch."""
    rng = random.Random(seed * 13 + 5)
    L = SETUP[signal]["length"]
    if signal == "sound":
        # Back-to-back 10 s recordings. One long simulation would include the unit tripping off and
        # going silent, which sound alone cannot tell apart from a unit that never started.
        healthy = np.concatenate([simulate.sound("normal", seconds=CHUNK, seed=seed + k) for k in range(L // CHUNK)])
        faulty = healthy if fault == "normal" else np.concatenate(
            [simulate.sound(fault, seconds=CHUNK, seed=seed + k, severity=severity) for k in range(L // CHUNK)])
        w, truth = _weights(L, rng, len(healthy))
        x = (1 - w) * healthy + w * faulty
        stream = {"samples": x}
    else:
        healthy = sensors.generate("normal", minutes=L, seed=seed)
        faulty = healthy if fault == "normal" else sensors.generate(fault, minutes=L, seed=seed, severity=severity)
        w, truth = _weights(L, rng, L)
        rows = []
        for m, (a, b) in enumerate(zip(healthy, faulty)):
            rows.append({"minute": m, **{k: round((1 - w[m]) * a[k] + w[m] * b[k], 2) for k in a if k != "minute"}})
        stream = {"rows": rows}
    return {"fault": fault, "signal": signal, "seed": seed, "truth": None if fault == "normal" else truth,
            "length": L, "weights": w, **stream}


def detect(stream: dict) -> dict:
    """Slide the window, classify each position, merge neighbours with the same answer."""
    model = pattern.get_model()
    if model is None:
        raise RuntimeError("Train the pattern model first: python -m faultlens.pattern")
    sig, cfg = stream["signal"], SETUP[stream["signal"]]
    W, S, L = cfg["window"], cfg["step"], stream["length"]
    starts = list(np.arange(0, L - W + 1e-9, S))
    if sig == "sound":
        sr = simulate.SR
        feats = [features.sound_features(stream["samples"][int(a * sr):int((a + W) * sr)]) for a in starts]
        P = model.proba_batch("sound", feats)
    else:
        feats = [features.sensor_features(stream["rows"][int(a):int(a + W)]) for a in starts]
        P = model.proba_batch("sensors", feats)
    windows = []
    for a, p in zip(starts, P):
        k = int(np.argmax(p))
        label = pattern.CLASSES[k]
        is_fault = label != "normal" and p[k] >= cfg["sure"]
        # Each window speaks for the middle of its span, one step wide, so stretches tile the timeline.
        mid = a + W / 2
        windows.append({"start": round(max(0.0, mid - S / 2), 1), "end": round(min(L, mid + S / 2), 1),
                        "label": label if is_fault else "normal", "top": label, "confidence": round(float(p[k]), 3)})
    windows[0]["start"], windows[-1]["end"] = 0.0, float(L)
    # Hold: a change of answer only counts once it lasts `hold` windows; shorter runs keep the
    # answer before them.
    hold = cfg["hold"]
    labels = [wdw["label"] for wdw in windows]
    settled, current, i = [], "normal", 0
    while i < len(labels):
        j = i
        while j < len(labels) and labels[j] == labels[i]:
            j += 1
        run = labels[i] if (j - i >= hold or labels[i] == current) else current
        settled += [run] * (j - i)
        current = run
        i = j
    for wdw, lab in zip(windows, settled):
        wdw["raw_label"], wdw["label"] = wdw["label"], lab
    segments = []
    for wdw in windows:
        if segments and segments[-1]["label"] == wdw["label"]:
            seg = segments[-1]
            seg["end"] = wdw["end"]
            seg["confidences"].append(wdw["confidence"])
        else:
            segments.append({"start": wdw["start"], "end": wdw["end"], "label": wdw["label"], "confidences": [wdw["confidence"]]})
    for seg in segments:
        c = seg.pop("confidences")
        seg["confidence"] = round(float(np.mean(c)), 3)
        seg["windows"] = len(c)
    return {"windows": windows, "segments": segments, "threshold": cfg["sure"], "hold": hold}


def score(stream: dict, found: dict) -> dict:
    fault, truth, L = stream["fault"], stream["truth"], stream["length"]
    faults = [x for x in found["segments"] if x["label"] != "normal"]

    def overlap(seg, a, b):
        return max(0.0, min(seg["end"], b) - max(seg["start"], a))

    if truth is None:   # control run: anything flagged is a false alarm
        return {"right_fault": None, "coverage": None, "delay": None, "false_alarms": len(faults),
                "false_alarm_time": round(sum(x["end"] - x["start"] for x in faults), 1),
                "wrong_label_time": 0.0, "slow_to_clear": 0.0, "wrong_labels": sorted({x["label"] for x in faults})}
    a, b = truth
    right = [x for x in faults if x["label"] == fault and overlap(x, a, b) > 0]
    covered = sum(overlap(x, a, b) for x in faults if x["label"] == fault)
    W = SETUP[stream["signal"]]["window"]
    # After the repair, each window still holds up to one window-length of the fault, so an alarm
    # that runs on into [b, b + W] is slow to clear, not a false alarm. Anything else outside the
    # true stretch is a false alarm. Inside it but naming the wrong fault: wrong label.
    lag_time = sum(overlap(x, b, b + W) for x in faults if x["start"] <= b + W and x["end"] > b and x["start"] < b + W
                   and (x["start"] < b or any(y["end"] >= x["start"] - 1e-6 and y["start"] < b for y in faults)))
    false_parts = []
    for x in faults:
        for lo, hi in ((0.0, a), (b + W, float(L))):
            o = overlap(x, lo, hi)
            if o > 0:
                false_parts.append(o)
        # between b and b+W but not attached to the fault stretch: also false
        if x["start"] >= b and x["start"] < b + W and not any(y["start"] < b and y["end"] >= x["start"] - 1e-6 for y in faults):
            false_parts.append(overlap(x, b, b + W))
    wrong_inside = sum(overlap(x, a, b) for x in faults if x["label"] != fault)
    delay = round(min(x["start"] for x in right) - a, 1) if right else None
    step = SETUP[stream["signal"]]["step"]
    return {"right_fault": bool(right), "coverage": round(covered / (b - a), 3), "delay": delay,
            "false_alarms": len([o for o in false_parts if o >= step]),
            "false_alarm_time": round(sum(false_parts), 1), "slow_to_clear": round(lag_time, 1),
            "wrong_label_time": round(wrong_inside, 1),
            "wrong_labels": sorted({x["label"] for x in faults if x["label"] != fault})}


CHANNEL_NAMES = {"supply_air_c": "Air leaving the indoor unit (C)", "discharge_psig": "Discharge (high-side) pressure (psi)",
                 "suction_psig": "Suction (low-side) pressure (psi)", "compressor_amps": "Compressor current (A)",
                 "room_sensor_c": "Room temperature: the unit's sensor vs a reference thermometer (C)"}


def series(stream: dict) -> dict:
    """What to draw: a spectrogram for sound, the reading that shows the fault best for sensors."""
    if stream["signal"] == "sound":
        return {"kind": "spectrogram", "image": render.spectrogram(stream["samples"], width=1200, height=220)}
    from . import knowledge
    ch = KEY_READING.get(stream["fault"], "discharge_psig")
    rng = knowledge.load().get("normal_ranges", {}).get(ch)
    out = {"kind": "line", "channel": ch, "name": CHANNEL_NAMES.get(ch, ch),
           "values": [r[ch] for r in stream["rows"]], "normal_range": rng}
    if ch == "room_sensor_c":   # the fault is the gap between the two
        out["compare"] = {"name": "Reference thermometer (return air)", "values": [r["return_air_c"] for r in stream["rows"]]}
    return out


def run(fault: str, signal: str = "sensors", seed: int | None = None, severity: float = 1.0) -> dict:
    seed = random.randrange(10**6) if seed is None else seed
    stream = build(fault, signal, seed, severity)
    found = detect(stream)
    return {"stream": stream, "found": found, "score": score(stream, found), "series": series(stream)}


def measure(runs: int = 20) -> dict:
    """Over many runs per fault (seeds 1000+, never used for tuning). Saved to models/anomaly_report.json."""
    import json
    rep = {"runs_per_fault": runs, "note": "Seeds 1000+; alarm settings were tuned on seeds 5000-5007.", "signals": {}}
    for signal in ("sensors", "sound"):
        out = {}
        for f in simulate.CONDITIONS:
            res = [run(f, signal, seed=1000 + i)["score"] for i in range(runs)]
            d = [r["delay"] for r in res if r["delay"] is not None]
            out[f] = {"runs": runs, "runs_with_false_alarms": sum(r["false_alarms"] > 0 for r in res)}
            if f != "normal":
                out[f].update(right_fault=sum(r["right_fault"] for r in res),
                              coverage=round(float(np.mean([r["coverage"] for r in res])), 3),
                              median_delay=round(float(np.median(d)), 1) if d else None,
                              wrong_label=round(float(np.mean([r["wrong_label_time"] for r in res])), 1),
                              slow_to_clear=round(float(np.mean([r["slow_to_clear"] for r in res])), 1))
        rep["signals"][signal] = {"unit": SETUP[signal]["unit"], "window": SETUP[signal]["window"],
                                  "sure": SETUP[signal]["sure"], "hold": SETUP[signal]["hold"], "faults": out}
    path = pattern.ROOT / "models" / "anomaly_report.json"
    path.write_text(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":   # measure it over many runs
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20)
    args = ap.parse_args()
    for signal in ("sensors", "sound"):
        print(f"\n{signal}: {args.runs} runs per fault")
        for f in simulate.CONDITIONS:
            res = [run(f, signal, seed=1000 + i)["score"] for i in range(args.runs)]
            if f == "normal":
                print(f"  {f:28s} false alarms in {sum(r['false_alarms'] > 0 for r in res)}/{len(res)} runs")
                continue
            right = sum(r["right_fault"] for r in res)
            cov = np.mean([r["coverage"] for r in res])
            delays = [r["delay"] for r in res if r["delay"] is not None]
            fa = sum(r["false_alarms"] > 0 for r in res)
            wl = np.mean([r["wrong_label_time"] for r in res])
            lag = np.mean([r["slow_to_clear"] for r in res])
            print(f"  {f:28s} right {right}/{len(res)}  coverage {cov:.0%}  median delay "
                  f"{np.median(delays) if delays else float('nan'):.0f}{SETUP[signal]['unit']}  false alarms in {fa}/{len(res)} runs"
                  f"  wrong label {wl:.0f}{SETUP[signal]['unit']} avg  slow to clear {lag:.0f}{SETUP[signal]['unit']} avg")


# ----------------------------------------------------------------------------- explanation by Muse Spark

EXPLAIN_PROMPT = """You are FaultLens, explaining to a technician what happened in one anomaly-lab run on a {equipment}.
A fault was injected into a stretch of a healthy signal ({signal}), fading in, holding, then repaired. FaultLens's
own pattern model slid a window along the signal and labelled each stretch. You get the true stretch, what the model
called, its confidence per window, the score, and the readings (or the spectrogram picture for sound).

In 4-6 plain sentences, no jargon:
- say what changed in the signal and when, quoting actual numbers and times from the data given,
- say when and why the model caught it, and explain any wrong label or delay (a fault that starts mildly looks
  like a neighbouring fault; each window looks back {window} {unit}, so alarms clear late),
- say what the run shows about this signal for this fault, honestly, including weaknesses.
Only use numbers that appear in the data. Do not invent readings."""


def explain_data(r: dict) -> dict:
    """A compact, checkable summary of one run for the model."""
    st, found, sc = r["stream"], r["found"], r["score"]
    cfg = SETUP[st["signal"]]
    data = {"signal": st["signal"], "injected_fault": st["fault"], "true_fault_stretch": st["truth"], "length": st["length"],
            "unit": cfg["unit"], "window": cfg["window"], "step": cfg["step"],
            "alarm_rule": f"at least {int(cfg['sure'] * 100)}% sure for {cfg['hold']} windows in a row",
            "model_called": [{k: s[k] for k in ("start", "end", "label", "confidence")} for s in found["segments"]],
            "windows": [{"centre": round((w["start"] + w["end"]) / 2, 1), "top": w["top"], "confidence": w["confidence"],
                         "alarm": w["label"]} for w in found["windows"]],
            "score": sc}
    if st["signal"] == "sensors":
        ch = KEY_READING.get(st["fault"], "discharge_psig")
        vals = [row[ch] for row in st["rows"]]
        data["key_reading"] = {"name": CHANNEL_NAMES.get(ch, ch),
                               "every_10_min": [{"minute": m, "value": vals[m]} for m in range(0, len(vals), 10)]}
        if ch == "room_sensor_c":
            data["key_reading"]["reference_every_10_min"] = [{"minute": m, "value": st["rows"][m]["return_air_c"]}
                                                            for m in range(0, len(vals), 10)]
    return data


def _mock_explanation(d: dict) -> str:
    u, s = d["unit"], d["score"]
    if not d["true_fault_stretch"]:
        return (f"(offline mode) Nothing was injected. The model raised {s['false_alarms']} false alarm(s) "
                f"over {d['length']} {u}.")
    a, b = d["true_fault_stretch"]
    calls = ", ".join(f"{x['label']} {x['start']}-{x['end']} {u} ({int(x['confidence'] * 100)}%)" for x in d["model_called"] if x["label"] != "normal")
    return (f"(offline mode, no model call) The fault ran from {a} to {b} {u}. The model called: {calls or 'nothing'}. "
            f"It was first caught {s['delay']} {u} after it began and covered {int((s['coverage'] or 0) * 100)}% of the stretch; "
            f"each window looks back {d['window']} {u}, so the alarm cleared {s['slow_to_clear']} {u} after the repair.")


def explain(fault: str, signal: str, seed: int) -> dict:
    """Muse Spark explains one run in plain words. One call."""
    import json as _json
    import time as _time
    from . import config, knowledge
    from .meta_client import client, image_part, text_part
    t0 = _time.time()
    r = run(fault, signal, seed)
    d = explain_data(r)
    if config.MOCK:
        return {"text": _mock_explanation(d), "model": "offline mode", "seconds": 0.0, "usage": {}}
    cfg = SETUP[signal]
    content = [text_part("Run data:\n" + _json.dumps(d))]
    if signal == "sound":
        content += [text_part("Spectrogram of the whole 60 s recording, time left to right, pitch bottom to top:"),
                    image_part(r["series"]["image"])]
    resp = client().responses.create(model=config.AGENT_MODEL, input=[
        {"role": "system", "content": EXPLAIN_PROMPT.format(equipment=knowledge.load()["equipment"], signal=signal,
                                                            window=cfg["window"], unit=cfg["unit"])},
        {"role": "user", "content": content}])
    usage = {"input_tokens": getattr(resp.usage, "input_tokens", 0), "output_tokens": getattr(resp.usage, "output_tokens", 0)}
    return {"text": (resp.output_text or "").strip(), "model": f"Muse Spark ({config.AGENT_MODEL})",
            "seconds": round(_time.time() - t0, 1), "usage": usage}

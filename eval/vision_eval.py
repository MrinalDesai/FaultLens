"""How well does Muse Spark diagnose from the pictures alone?

For freshly generated units it is shown only the two thermal images and the sound spectrogram
(no sensor log, no pattern model, no notes) and must pick the condition. The same units are
scored by the pattern model using the same three signals, and by both combined.

This measures a capability of Meta's model directly: reading thermal images and a spectrogram.

Run:  python -m eval.vision_eval --per-condition 3                       (Muse Spark on Meta's API)
      python -m eval.vision_eval --per-condition 3 --provider fireworks  (Muse Spark Open on Fireworks)
Both use the same seeds, so they see exactly the same units.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import re

import numpy as np
from openai import OpenAI

from faultlens import config, features, knowledge, pattern, render, simulate
from faultlens.meta_client import client, image_part, image_to_data_url, text_part

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "models" / "vision_report.json"
REPORT_OPEN = ROOT / "models" / "vision_report_open.json"

PROMPT = """You are looking at evidence from a split air conditioner with one condition from the list below.
You get three images: a thermal image of the indoor coil, a thermal image of the outdoor unit (coil on the left,
compressor round body and capacitor small block on the right), and a spectrogram of the outdoor-unit sound
(10 s left to right, pitch bottom to top, brighter is louder). Readings printed on the thermal images are real.

Conditions:
{menu}

Look at the images and call the answer tool with a probability for every condition and one sentence on what you saw."""


def _tool() -> dict:
    ids = knowledge.fault_ids()
    return {"type": "function", "name": "answer", "description": "Your reading of the images.",
            "parameters": {"type": "object", "properties": {
                "probabilities": {"type": "object", "properties": {i: {"type": "number"} for i in ids},
                                  "required": ids},
                "what_i_saw": {"type": "string"}}, "required": ["probabilities", "what_i_saw"]}}


def ask_model(thermal_in, thermal_out, sound) -> tuple[dict, str, dict]:
    menu = "\n".join(f"- {f['fault_id']}: {f['title']}" for f in knowledge.fault_menu())
    content = [text_part(PROMPT.format(menu=menu)),
               text_part("Indoor coil:"), image_part(render.thermal_view(thermal_in)),
               text_part("Outdoor unit:"), image_part(render.thermal_view(thermal_out)),
               text_part("Spectrogram:"), image_part(render.spectrogram(sound))]
    kwargs = dict(model=config.AGENT_MODEL, input=[{"role": "user", "content": content}], tools=[_tool()])
    try:
        resp = client().responses.create(reasoning={"effort": config.COACH_REASONING}, **kwargs)
    except TypeError:
        resp = client().responses.create(**kwargs)
    usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
    for item in resp.output:
        if getattr(item, "type", "") == "function_call" and item.name == "answer":
            a = json.loads(item.arguments or "{}")
            return a.get("probabilities", {}), a.get("what_i_saw", ""), usage
    return {}, (getattr(resp, "output_text", "") or "")[:200], usage


def ask_open_model(thermal_in, thermal_out, sound) -> tuple[dict, str, dict]:
    """Same question to Muse Spark Open on Fireworks (Chat Completions, JSON answer)."""
    menu = "\n".join(f"- {f['fault_id']}: {f['title']}" for f in knowledge.fault_menu())
    ask = PROMPT.format(menu=menu).replace(
        "call the answer tool with a probability for every condition and one sentence on what you saw.",
        'reply with JSON only: {"probabilities": {"<condition id>": number, ...}, "what_i_saw": "one sentence"}.')
    content = [{"type": "text", "text": ask}]
    for caption, img in (("Indoor coil:", render.thermal_view(thermal_in)),
                         ("Outdoor unit:", render.thermal_view(thermal_out)),
                         ("Spectrogram:", render.spectrogram(sound))):
        content += [{"type": "text", "text": caption},
                    {"type": "image_url", "image_url": {"url": image_to_data_url(img)}}]
    fw = OpenAI(base_url=config.FIREWORKS_BASE_URL, api_key=config.FIREWORKS_API_KEY)
    resp = fw.chat.completions.create(model=config.FIREWORKS_MODEL, messages=[{"role": "user", "content": content}],
                                      max_tokens=4000)
    usage = {"input_tokens": resp.usage.prompt_tokens, "output_tokens": resp.usage.completion_tokens}
    text = resp.choices[0].message.content or ""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}, text[:200], usage
    try:
        a = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}, text[:200], usage
    return a.get("probabilities", {}), a.get("what_i_saw", ""), usage


def _norm(p: dict) -> np.ndarray:
    v = np.array([max(float(p.get(c, 0) or 0), 0) for c in pattern.CLASSES])
    return v / v.sum() if v.sum() > 0 else np.full(len(v), 1 / len(v))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-condition", type=int, default=3)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--provider", choices=["meta", "fireworks"], default="meta")
    args = ap.parse_args()
    open_model = args.provider == "fireworks"
    if open_model and (not config.FIREWORKS_API_KEY or config.FIREWORKS_API_KEY.startswith("your-")):
        raise SystemExit("Add FIREWORKS_API_KEY to .env first.")
    reader = ask_open_model if open_model else ask_model
    model_name = config.FIREWORKS_MODEL if open_model else config.AGENT_MODEL
    model = pattern.get_model()
    if model is None:
        raise SystemExit("Train the pattern model first: python -m faultlens.pattern")
    rng = random.Random(args.seed)
    rows, usage = [], {"input_tokens": 0, "output_tokens": 0}
    t0 = time.time()
    for cond in pattern.CLASSES:
        for _ in range(args.per_condition):
            seed = rng.randint(0, 10**8)
            sev = 1.0 if cond == "normal" else rng.uniform(0.3, 1.0)
            ti = simulate.thermal_indoor(cond, seed=seed, severity=sev)
            to = simulate.thermal_outdoor(cond, seed=seed, severity=sev)
            snd = simulate.sound(cond, seed=seed, severity=sev)
            feats = {"sound": features.sound_features(snd),
                     "thermal_indoor": features.thermal_indoor_features(ti),
                     "thermal_outdoor": features.thermal_outdoor_features(to)}
            p_pattern = model.fuse({m: model.proba(m, f) for m, f in feats.items()})
            try:
                probs, saw, u = reader(ti, to, snd)
                usage = {k: usage[k] + u[k] for k in usage}
            except Exception as e:
                probs, saw = {}, f"error: {e}"
            p_vision = _norm(probs)
            p_both = model.fuse({"pattern": p_pattern, "vision": p_vision}, calibrated=False)
            row = {"condition": cond, "severity": round(sev, 2),
                   "muse_spark": pattern.CLASSES[int(np.argmax(p_vision))] if probs else None,
                   "pattern_model": pattern.CLASSES[int(np.argmax(p_pattern))],
                   "combined": pattern.CLASSES[int(np.argmax(p_both))], "what_muse_spark_saw": saw}
            rows.append(row)
            print(f"{cond:28s} sev {sev:.2f} | Muse Spark: {row['muse_spark']} | pattern: {row['pattern_model']} | both: {row['combined']}")

    def acc(key, subset=None):
        rs = [r for r in rows if r[key] is not None and (subset is None or subset(r))]
        return round(sum(r[key] == r["condition"] for r in rs) / len(rs), 3) if rs else None

    mild = lambda r: r["condition"] != "normal" and r["severity"] < 0.55  # noqa: E731
    report = {
        "what": "Muse Spark diagnosing from two thermal images and a spectrogram only (no sensor log, no notes). "
                "Units are simulated and generated fresh with new seeds.",
        "units": len(rows), "model": model_name, "provider": args.provider,
        "reasoning": None if open_model else config.COACH_REASONING,
        "accuracy": {"muse_spark_images_only": acc("muse_spark"), "pattern_model_same_signals": acc("pattern_model"),
                     "both_combined": acc("combined")},
        "accuracy_mild_faults": {"muse_spark_images_only": acc("muse_spark", mild),
                                 "pattern_model_same_signals": acc("pattern_model", mild),
                                 "both_combined": acc("combined", mild)},
        "chance": round(1 / len(pattern.CLASSES), 3),
        "usage": usage, "seconds": round(time.time() - t0, 1), "rows": rows,
    }
    out = REPORT_OPEN if open_model else REPORT
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"Saved {out.relative_to(ROOT)}")
    print(json.dumps({k: report[k] for k in ("accuracy", "accuracy_mild_faults", "chance", "usage", "seconds")}, indent=1))


if __name__ == "__main__":
    main()

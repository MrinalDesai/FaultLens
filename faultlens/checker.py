"""Independent checker: a second model reviews whether the evidence supports the diagnosis.

By default it runs on Muse Spark through Meta's API with a separate, skeptical prompt.
Set CHECKER_BASE_URL, CHECKER_API_KEY and CHECKER_MODEL to run it on Muse Spark Open
through a partner (Baseten, Modal or Fireworks), which is what the open-weights
bonus prize requires. Partners use the Chat Completions format.
"""

from __future__ import annotations

import json
import re

from openai import OpenAI

from . import config, render
from .agent import Evidence
from .meta_client import image_to_data_url

CHECKER_PROMPT = """You are a strict reviewer checking another assistant's diagnosis of a cooling-equipment failure.
You do not propose your own diagnosis. You only judge whether the evidence actually supports the top-ranked cause.

You are shown the same images the assistant saw (thermal images, sound spectrogram, photos) and the
numbers printed on the thermal images. A claim about what an image shows counts as evidence if you can
see it in that image or in its printed readings. Do not reject a claim only because it is not in the text.

Reject if any of these are true:
- A piece of evidence_for is not actually present in the evidence below (invented or misread).
- The evidence equally supports the second-ranked cause and nothing separates them.
- Something in the evidence clearly contradicts the top cause and is not addressed.
- The status is "confident" but a key reading needed to confirm the cause is missing.
- The pattern match's signals disagree about the top cause and the assessment does not explain why.

Reply with JSON only:
{"verdict": "accept" or "reject", "reasons": ["..."], "missing_evidence": "one specific observation, or empty"}
"""


def _images(ev: Evidence) -> list[tuple[str, object]]:
    """(caption, image) pairs: exactly what the assistant was shown."""
    out = []
    if ev.thermal_indoor is not None:
        out.append(("Thermal image, indoor coil", render.thermal_view(ev.thermal_indoor)))
    if ev.thermal_outdoor is not None:
        out.append(("Thermal image, outdoor unit (coil left, compressor round body right, capacitor small block "
                    "top right)", render.thermal_view(ev.thermal_outdoor)))
    if ev.sound is not None:
        out.append(("Spectrogram of the outdoor-unit sound, 10 s left to right, pitch bottom to top",
                    render.spectrogram(ev.sound)))
    for i, frame in enumerate(ev.frames[:4]):
        out.append((f"Photo {i}", frame))
    return out


def _evidence_text(ev: Evidence, assessment: dict) -> str:
    readings = {}
    if ev.thermal_indoor is not None:
        readings["thermal_indoor"] = render.thermal_readings(ev.thermal_indoor)
    if ev.thermal_outdoor is not None:
        readings["thermal_outdoor"] = render.thermal_readings(ev.thermal_outdoor)
    return json.dumps({
        "sensor_summary": ev.sensor_summary(),
        "technician_notes": ev.notes,
        "parts_detected": ev.parts_summary(),
        "pattern_match": ev.pattern_summary(),
        "images_provided": {"photos": len(ev.frames), "thermal_indoor": ev.thermal_indoor is not None,
                            "thermal_outdoor": ev.thermal_outdoor is not None, "spectrogram": ev.sound is not None},
        "numbers_printed_on_thermal_images": readings,
        "assessment_to_review": {k: assessment.get(k) for k in ("status", "hypotheses", "summary")},
    }, indent=1)


def _parse(text: str) -> dict:
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return {"verdict": "reject", "reasons": ["Checker reply was not valid JSON."], "missing_evidence": ""}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"verdict": "reject", "reasons": ["Checker reply was not valid JSON."], "missing_evidence": ""}
    data["verdict"] = "accept" if str(data.get("verdict", "")).lower() == "accept" else "reject"
    data.setdefault("reasons", [])
    data.setdefault("missing_evidence", "")
    return data


def check(ev: Evidence, assessment: dict) -> dict:
    if config.MOCK:
        top = (assessment.get("hypotheses") or [{}])[0].get("probability", 0)
        ok = top >= 0.6
        return {"verdict": "accept" if ok else "reject", "checker_model": "mock",
                "reasons": ["(mock) top cause clearly ahead" if ok else "(mock) top two causes too close"],
                "missing_evidence": ""}

    body = _evidence_text(ev, assessment)
    images = _images(ev)
    saw_images = bool(images)
    if config.CHECKER_BASE_URL:
        partner = OpenAI(base_url=config.CHECKER_BASE_URL, api_key=config.CHECKER_API_KEY)
        content = [{"type": "text", "text": body}]
        for caption, img in images:
            content.append({"type": "text", "text": caption + ":"})
            content.append({"type": "image_url", "image_url": {"url": image_to_data_url(img)}})
        try:
            resp = partner.chat.completions.create(
                model=config.CHECKER_MODEL,
                messages=[{"role": "system", "content": CHECKER_PROMPT}, {"role": "user", "content": content}],
            )
        except Exception:
            # If the partner refuses images, fall back to text plus the printed thermal readings.
            saw_images = False
            resp = partner.chat.completions.create(
                model=config.CHECKER_MODEL,
                messages=[{"role": "system", "content": CHECKER_PROMPT}, {"role": "user", "content": body}],
            )
        text, model = resp.choices[0].message.content, config.CHECKER_MODEL
    else:
        from .meta_client import client, text_part
        parts = [text_part(body)]
        for caption, img in images:
            parts += [text_part(caption + ":"), {"type": "input_image", "image_url": image_to_data_url(img)}]
        resp = client().responses.create(
            model=config.AGENT_MODEL,
            input=[{"role": "system", "content": CHECKER_PROMPT}, {"role": "user", "content": parts}],
        )
        text, model = resp.output_text, config.AGENT_MODEL
    return {**_parse(text), "checker_model": model, "checker_saw_images": saw_images}

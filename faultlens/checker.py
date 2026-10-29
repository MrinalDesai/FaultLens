"""Independent checker: a second model reviews whether the evidence supports the diagnosis.

By default it runs on Muse Spark through Meta's API with a separate, skeptical prompt.
Set CHECKER_BASE_URL, CHECKER_API_KEY and CHECKER_MODEL to run it on Muse Spark Open
through a partner (Baseten, Modal or Fireworks), which is what the open-weights
bonus prize requires. Partners use the Chat Completions format.
"""

from __future__ import annotations

import base64
import json
import re

from openai import OpenAI

from . import alarms, config, render
from .agent import Evidence
from .meta_client import image_to_data_url

CHECKER_PROMPT = """You are a strict reviewer checking another assistant's diagnosis of a cooling-equipment failure.
You do not propose your own diagnosis. You only judge whether the evidence actually supports the top-ranked cause.

You are shown the same images the assistant saw (thermal images, sound spectrogram, photos) and the
numbers printed on the thermal images. When a sound recording of the outdoor unit is attached, listen to
it yourself: a missing fan tone, a hum followed by a click, rattles or hiss are evidence too. A claim about what an image shows counts as evidence if you can
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


# The recording is only sent once `python -m faultlens.checker` has shown that a format really
# reaches the model (CHECKER_AUDIO_FORMAT in .env). A request can succeed while the audio is
# silently dropped, so "no error" is not proof. Set to False after a refusal this session.
_PARTNER_TAKES_AUDIO = True
AUDIO_FORMATS = ("audio_url", "input_audio")


def _audio_part(ev: Evidence, fmt: str | None = None) -> list[dict]:
    """The outdoor-unit recording as a WAV in the given format."""
    fmt = fmt or config.CHECKER_AUDIO_FORMAT
    if ev.sound is None or fmt not in AUDIO_FORMATS:
        return []
    wav = base64.b64encode(render.write_wav(ev.sound)).decode()
    part = ({"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64," + wav}} if fmt == "audio_url"
            else {"type": "input_audio", "input_audio": {"data": wav, "format": "wav"}})
    secs = len(ev.sound) / 16000
    return [{"type": "text", "text": f"Sound recording of the outdoor unit ({secs:.0f} s, 16 kHz), the same recording the spectrogram shows:"},
            part]


def _evidence_text(ev: Evidence, assessment: dict) -> str:
    readings = {}
    if ev.thermal_indoor is not None:
        readings["thermal_indoor"] = render.thermal_readings(ev.thermal_indoor)
    if ev.thermal_outdoor is not None:
        readings["thermal_outdoor"] = render.thermal_readings(ev.thermal_outdoor)
    return json.dumps({
        "sensor_summary": ev.sensor_summary(),
        "technician_notes": ev.notes,
        "alarm_history": alarms.summarize(ev.alarms) if ev.alarms_given else "not provided",
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
    heard_sound = False
    if config.CHECKER_BASE_URL:
        global _PARTNER_TAKES_AUDIO
        partner = OpenAI(base_url=config.CHECKER_BASE_URL, api_key=config.CHECKER_API_KEY)
        content = [{"type": "text", "text": body}]
        for caption, img in images:
            content.append({"type": "text", "text": caption + ":"})
            content.append({"type": "image_url", "image_url": {"url": image_to_data_url(img)}})
        audio = _audio_part(ev) if _PARTNER_TAKES_AUDIO else []

        def ask(user_content):
            return partner.chat.completions.create(
                model=config.CHECKER_MODEL,
                messages=[{"role": "system", "content": CHECKER_PROMPT}, {"role": "user", "content": user_content}])

        # Try images + sound, then images only, then text only. A rejected request is not billed as a
        # completion, and once audio is refused it is not tried again in this session.
        resp = None
        if audio:
            try:
                resp, heard_sound = ask(content + audio), True
            except Exception:
                _PARTNER_TAKES_AUDIO = False
        if resp is None:
            try:
                resp = ask(content)
            except Exception:
                saw_images = False
                resp = ask(body)
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
    return {**_parse(text), "checker_model": model, "checker_saw_images": saw_images, "checker_heard_sound": heard_sound}


def test_audio() -> None:
    """Find a way of sending sound that really reaches the model. One small call per format.
    Proof is the token count: 5 s of audio adds hundreds of input tokens; if it barely moves,
    the audio was dropped even though the call succeeded."""
    from . import simulate
    if not config.CHECKER_BASE_URL:
        raise SystemExit("No partner checker set in .env (CHECKER_BASE_URL).")
    ev = Evidence()
    ev.sound = simulate.sound("failed_compressor_capacitor", seconds=5, seed=7, severity=1.0)
    partner = OpenAI(base_url=config.CHECKER_BASE_URL, api_key=config.CHECKER_API_KEY)
    prompt = {"type": "text", "text": "Describe this 5-second sound recording in one sentence."}
    baseline = partner.chat.completions.create(model=config.CHECKER_MODEL, max_tokens=1,
                                               messages=[{"role": "user", "content": [prompt]}])
    base_in = baseline.usage.prompt_tokens
    print(f"Without sound: {base_in} input tokens")
    working = None
    for fmt in AUDIO_FORMATS:
        try:
            resp = partner.chat.completions.create(
                model=config.CHECKER_MODEL, max_tokens=800,
                messages=[{"role": "user", "content": [prompt] + _audio_part(ev, fmt)[1:]}])
        except Exception as e:
            print(f"{fmt}: refused ({str(e)[:120]})")
            continue
        extra = resp.usage.prompt_tokens - base_in
        heard = (resp.choices[0].message.content or "").strip()
        reached = extra > 50
        print(f"{fmt}: {resp.usage.prompt_tokens} input tokens (+{extra}) -> "
              f"{'SOUND REACHED THE MODEL' if reached else 'sound was dropped'}")
        print(f"   it said: {heard[:200]}")
        if reached and working is None:
            working = fmt
    if working:
        print(f"\nAdd this line to .env, then restart the app:\nCHECKER_AUDIO_FORMAT={working}")
    else:
        print("\nNo format reached the model. Leave CHECKER_AUDIO_FORMAT out of .env: "
              "the checker keeps using the spectrogram picture, and never claims it listened.")


if __name__ == "__main__":
    test_audio()

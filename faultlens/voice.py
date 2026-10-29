"""Muse Voice Transcribe for recorded notes.

Known from Meta's docs: recordings go to POST {base}/asr/transcribe as a
multipart upload, Bearer auth, mono 16-bit PCM WAV at 16 or 24 kHz only,
up to 32 MB and 10 minutes.

NOT yet confirmed: the exact form field names and the response JSON shape.
The guesses below ("file", "model", response "text" or "turns") must be checked
against the API reference on day one. Typed notes always work as a fallback.
"""

from __future__ import annotations

import io
import wave

import requests

from . import config


def check_wav(data: bytes) -> str | None:
    """Return a problem description, or None if the file meets the format rules."""
    try:
        with wave.open(io.BytesIO(data)) as w:
            if w.getnchannels() != 1:
                return "Audio must be mono."
            if w.getsampwidth() != 2:
                return "Audio must be 16-bit."
            if w.getframerate() not in (16000, 24000):
                return "Audio must be 16 kHz or 24 kHz."
    except (wave.Error, EOFError):
        return "Audio must be a WAV file."
    return None


def transcribe(data: bytes) -> str:
    if config.MOCK:
        return "Outdoor fan is not spinning, compressor is humming, the unit is blowing warm air."

    problem = check_wav(data)
    if problem:
        raise ValueError(problem + " Convert with: ffmpeg -i in.m4a -ac 1 -ar 16000 out.wav")

    resp = requests.post(
        f"{config.META_BASE_URL}/asr/transcribe",
        headers={"Authorization": f"Bearer {config.MODEL_API_KEY}"},
        files={"file": ("note.wav", data, "audio/wav")},  # field name to verify
        data={"model": config.VOICE_MODEL},  # field name to verify
        timeout=120,
    )
    resp.raise_for_status()
    body = resp.json()
    if isinstance(body.get("text"), str):
        return body["text"]
    turns = body.get("turns") or body.get("segments") or []
    return " ".join(t.get("text", "") for t in turns).strip()

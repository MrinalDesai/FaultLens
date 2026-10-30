"""Muse Voice Transcribe for recorded notes.

Known from Meta's docs: recordings go to POST {base}/asr/transcribe as a
multipart upload, Bearer auth, mono 16-bit PCM WAV at 16 or 24 kHz only,
up to 32 MB and 10 minutes.

Request: a "request" part (JSON: mode, model, audioEncoding) and an "audio" part (the WAV).
Response: {"transcript": ..., "turns": [...]}. Typed notes always work as a fallback.
"""

from __future__ import annotations

import io
import json
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

    # Format from Meta's speech-to-text docs (dev.meta.ai/docs/speech-to-text): two multipart parts,
    # "request" (JSON settings) and "audio" (the WAV file); the answer's "transcript" is the full text.
    settings = json.dumps({"mode": "PUSH_TO_TALK", "model": config.VOICE_MODEL, "audioEncoding": "WAV"})
    resp = requests.post(
        f"{config.META_BASE_URL}/asr/transcribe",
        headers={"Authorization": f"Bearer {config.MODEL_API_KEY}"},
        files={"request": (None, settings, "application/json"),
               "audio": ("note.wav", data, "audio/wav")},
        timeout=120,
    )
    if not resp.ok:
        raise RuntimeError(f"{resp.status_code} from Muse Voice Transcribe: {resp.text[:300]}")
    body = resp.json()
    if isinstance(body.get("transcript"), str) and body["transcript"].strip():
        return body["transcript"].strip()
    return " ".join(t.get("transcript", "") for t in body.get("turns") or []).strip()

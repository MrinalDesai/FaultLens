"""Read a handwritten job card: full transcription plus organised fields.

Runs on Muse Spark Open through Fireworks (Meta's open-weights model). Falls back to
Muse Spark on Meta's API if no Fireworks key is set. For library cards the true contents
are known, so every field can be scored.
"""

from __future__ import annotations

import io
import json
import re
import time

from openai import OpenAI
from PIL import Image

from . import config, jobcards
from .meta_client import client, image_to_data_url, text_part

PROMPT = """This is a scanned, handwritten air-conditioner service job card.

1. Transcribe all the writing on the card, printed and handwritten, line by line.
2. Extract the handwritten values into these fields: {fields}.
   Copy values exactly as written. Use null for a field that is empty or that you cannot read. Do not guess.

Reply with JSON only:
{{"transcription": "...", "fields": {{"job_no": ..., "date": ..., ...}}}}"""


def load_upload(data: bytes, filename: str) -> Image.Image:
    """First page of a PDF, or an image file."""
    if filename.lower().endswith(".pdf") or data[:4] == b"%PDF":
        import pymupdf
        doc = pymupdf.open(stream=data, filetype="pdf")
        pix = doc[0].get_pixmap(dpi=150)
        return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    return Image.open(io.BytesIO(data)).convert("RGB")


def _parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return {"transcription": (text or "").strip(), "fields": {}}
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"transcription": (text or "").strip(), "fields": {}}
    fields = data.get("fields") or {}
    return {"transcription": data.get("transcription", ""),
            "fields": {k: fields.get(k) for k in jobcards.FIELDS}}


def _fireworks_ready() -> bool:
    return bool(config.FIREWORKS_API_KEY) and not config.FIREWORKS_API_KEY.startswith("your-")


def read(img: Image.Image, card_id: str | None = None) -> dict:
    """Returns {transcription, fields, model, seconds, usage}."""
    t0 = time.time()
    prompt = PROMPT.format(fields=", ".join(jobcards.FIELDS))
    big = image_to_data_url(img, max_side=1800)   # keep handwriting legible
    if config.MOCK:
        truth = jobcards.truth_for(card_id) if card_id else None
        fields = truth or {k: None for k in jobcards.FIELDS}
        return {"transcription": "(offline mode: true card contents shown, no model call)", "fields": fields,
                "model": "offline mode", "seconds": 0.0, "usage": {}}
    if _fireworks_ready():
        fw = OpenAI(base_url=config.FIREWORKS_BASE_URL, api_key=config.FIREWORKS_API_KEY)
        resp = fw.chat.completions.create(
            model=config.FIREWORKS_MODEL, max_tokens=4000,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt},
                                                   {"type": "image_url", "image_url": {"url": big}}]}])
        out = _parse(resp.choices[0].message.content)
        model = f"Muse Spark Open ({config.FIREWORKS_MODEL.split('/')[-1]}, Fireworks)"
        usage = {"input_tokens": resp.usage.prompt_tokens, "output_tokens": resp.usage.completion_tokens}
    else:
        resp = client().responses.create(
            model=config.AGENT_MODEL,
            input=[{"role": "user", "content": [text_part(prompt), {"type": "input_image", "image_url": big}]}])
        out = _parse(resp.output_text)
        model = f"Muse Spark ({config.AGENT_MODEL})"
        usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
    return {**out, "model": model, "seconds": round(time.time() - t0, 1), "usage": usage}


def _norm(v) -> str:
    s = str(v or "").lower().strip()
    s = re.sub(r"[\s\.,;:]+", " ", s)
    return s.replace("µ", "u").strip()


def score(fields: dict, truth: dict) -> dict:
    """Field-by-field comparison with the true card contents."""
    rows, right = [], 0
    for k in jobcards.FIELDS:
        t, r = truth.get(k) or "", fields.get(k) or ""
        ok = _norm(t) == _norm(r)
        if not ok:
            try:  # numbers: 7.6 vs 7.60
                ok = abs(float(_norm(t)) - float(_norm(r))) < 1e-6
            except ValueError:
                pass
        right += ok
        rows.append({"field": k, "label": jobcards.LABELS[k], "read": r, "on_card": t, "match": ok})
    return {"fields_right": right, "fields_total": len(jobcards.FIELDS), "rows": rows}


def history_summary(fields: dict) -> str:
    """One line for the game log and the coach."""
    bits = [f"{fields.get('date') or 'undated'}"]
    for k in ("complaint", "work_done", "parts_used", "remarks"):
        if fields.get(k):
            bits.append(f"{jobcards.LABELS[k]}: {fields[k]}")
    return "Previous visit " + "; ".join(bits) + "."

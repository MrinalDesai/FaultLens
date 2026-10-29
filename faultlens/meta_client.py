"""Thin wrapper around Meta Model API (OpenAI-compatible Responses API)."""

from __future__ import annotations

import base64
import io
import mimetypes
from functools import lru_cache

from openai import OpenAI
from PIL import Image

from . import config


@lru_cache(maxsize=1)
def client() -> OpenAI:
    if not config.MODEL_API_KEY:
        raise RuntimeError("MODEL_API_KEY is not set. Copy .env.example to .env and add your key.")
    # The OpenAI SDK does not read MODEL_API_KEY on its own, so pass it explicitly.
    return OpenAI(base_url=config.META_BASE_URL, api_key=config.MODEL_API_KEY)


def image_to_data_url(image: Image.Image | str | bytes, max_side: int = 1280) -> str:
    """Return a base64 data URL. Large images are shrunk to save tokens."""
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, bytes):
        image = Image.open(io.BytesIO(image))
    image = image.convert("RGB")
    image.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def file_to_data_url(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()


def text_part(text: str) -> dict:
    return {"type": "input_text", "text": text}


def image_part(image) -> dict:
    return {"type": "input_image", "image_url": image_to_data_url(image)}

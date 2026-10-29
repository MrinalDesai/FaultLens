"""SAM 3.1 on Meta Model API: find named parts in a frame.

Per Meta's docs, SAM 3.1 runs through the Responses API with model "sam-3.1".
The prompt is one short noun phrase per call. The answer is streamed as
special-token text, one record per match, for example:

    <0f>0<|box;x1=181;y1=365;x2=556;y2=486;w=1516;h=1600|><mask token>

<0f> is the frame number, the next number is the object ordinal, then the box.
We only read the boxes here; masks are skipped for now.
"""

from __future__ import annotations

import re

from PIL import Image

from . import config
from .meta_client import client, image_to_data_url

BOX_RE = re.compile(r"<(\d+)f>(\d+)<\|box;([^|]*)\|>")


def parse_boxes(raw: str, label: str) -> list[dict]:
    detections = []
    for frame, ordinal, fields in BOX_RE.findall(raw):
        values = {}
        for pair in fields.split(";"):
            if "=" in pair:
                k, v = pair.split("=", 1)
                try:
                    values[k] = float(v)
                except ValueError:
                    pass
        if not {"x1", "y1", "x2", "y2"} <= values.keys():
            continue
        detections.append(
            {
                "label": label,
                "frame": int(frame),
                "object": int(ordinal),
                "x1": values["x1"],
                "y1": values["y1"],
                "x2": values["x2"],
                "y2": values["y2"],
                "w": values.get("w"),
                "h": values.get("h"),
                "conf": values.get("c"),
            }
        )
    return detections


def segment_image(image: Image.Image, phrase: str) -> list[dict]:
    """Return boxes for every match of `phrase` (for example 'condenser fan')."""
    if config.MOCK:
        return _mock_boxes(image, phrase)

    stream = client().responses.create(
        model=config.SAM_MODEL,
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": phrase},
                    {"type": "input_image", "image_url": image_to_data_url(image)},
                ],
            }
        ],
        metadata={"include_confidence": "true", "mask_encoding": "one_bit"},
        stream=True,
    )
    raw = []
    for event in stream:
        if getattr(event, "type", "") == "response.output_text.delta":
            raw.append(event.delta)
    # No matches is a normal result: an empty answer, not an error.
    return parse_boxes("".join(raw), phrase)


def find_parts(image: Image.Image, phrases: list[str]) -> list[dict]:
    found = []
    for phrase in phrases:
        found.extend(segment_image(image, phrase))
    return found


def _mock_boxes(image: Image.Image, phrase: str) -> list[dict]:
    w, h = image.size
    seed = sum(map(ord, phrase)) % 5
    x1 = (0.08 + 0.15 * seed) * w
    y1 = (0.15 + 0.1 * seed) * h
    return [
        {
            "label": phrase, "frame": 0, "object": 0,
            "x1": x1, "y1": y1, "x2": min(x1 + 0.25 * w, w - 1), "y2": min(y1 + 0.25 * h, h - 1),
            "w": w, "h": h, "conf": 0.5,
        }
    ]

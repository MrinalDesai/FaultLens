"""Video and image helpers: keyframe extraction and overlay drawing."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def extract_keyframes(video_path: str, count: int = 4) -> list[Image.Image]:
    """Pick `count` evenly spaced, reasonably sharp frames from a short clip."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        cap.release()
        raise ValueError(f"Could not read frames from {video_path}")

    frames: list[Image.Image] = []
    # Sample a small window around each target position and keep the sharpest frame,
    # so motion blur from a handheld phone does not ruin the evidence.
    targets = np.linspace(0, total - 1, count + 2)[1:-1].astype(int)
    for t in targets:
        best, best_score = None, -1.0
        for offset in (-4, 0, 4):
            idx = int(min(max(t + offset, 0), total - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            score = cv2.Laplacian(gray, cv2.CV_64F).var()
            if score > best_score:
                best, best_score = frame, score
        if best is not None:
            frames.append(Image.fromarray(cv2.cvtColor(best, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


PALETTE = ["#E4572E", "#17BEBB", "#FFC914", "#76B041", "#2E86AB", "#A23B72", "#F18F01"]


def draw_boxes(image: Image.Image, detections: list[dict]) -> Image.Image:
    """Draw labelled boxes. Each detection: {label, x1, y1, x2, y2, w, h, conf?}."""
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    labels = sorted({d["label"] for d in detections})
    size = max(14, out.width // 45)
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:  # older Pillow without sized default font
        font = ImageFont.load_default()
    for d in detections:
        color = PALETTE[labels.index(d["label"]) % len(PALETTE)]
        # SAM reports boxes against the image size it saw; rescale to this image.
        sx = out.width / d.get("w", out.width)
        sy = out.height / d.get("h", out.height)
        box = [d["x1"] * sx, d["y1"] * sy, d["x2"] * sx, d["y2"] * sy]
        draw.rectangle(box, outline=color, width=max(2, out.width // 300))
        tag = d["label"] + (f" {d['conf']:.2f}" if d.get("conf") is not None else "")
        left, top, right, bottom = draw.textbbox((0, 0), tag, font=font)
        tw, th = right - left, bottom - top
        y = box[1] - th - 8 if box[1] - th - 8 >= 0 else box[1]
        draw.rectangle([box[0], y, box[0] + tw + 8, y + th + 8], fill=color)
        draw.text((box[0] + 4, y + 4 - top), tag, fill="black", font=font)
    return out

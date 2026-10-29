"""Turn signals into pictures for the console: thermal false colour and a sound spectrogram."""

from __future__ import annotations

import io
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import simulate

# Inferno-like colour ramp, from cold (black-purple) to hot (pale yellow).
_STOPS = np.array([
    [0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99],
    [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164],
], dtype=np.float64)
_LUT = np.stack([np.interp(np.linspace(0, 1, 256), np.linspace(0, 1, len(_STOPS)), _STOPS[:, k])
                 for k in range(3)], axis=1).astype(np.uint8)


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def thermal_view(temps: np.ndarray, scale: int = 4, label: str = "") -> Image.Image:
    """False-colour thermal image with a temperature bar, hottest and coldest marked."""
    lo, hi = np.percentile(temps, 1), np.percentile(temps, 99.7)
    norm = np.clip((temps - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = _LUT[(norm * 255).astype(np.uint8)]
    img = Image.fromarray(rgb).resize((temps.shape[1] * scale, temps.shape[0] * scale), Image.NEAREST)

    bar_w, pad = 26, 8
    canvas = Image.new("RGB", (img.width + bar_w + 70, img.height), (22, 34, 43))
    canvas.paste(img, (0, 0))
    bar = np.linspace(255, 0, img.height).astype(np.uint8)
    canvas.paste(Image.fromarray(np.repeat(_LUT[bar][:, None, :], bar_w, axis=1)), (img.width + pad, 0))
    d = ImageDraw.Draw(canvas)
    f = _font(max(13, img.height // 28))
    d.text((img.width + pad + bar_w + 6, 2), f"{hi:.0f} C", fill="white", font=f)
    d.text((img.width + pad + bar_w + 6, img.height - 20), f"{lo:.0f} C", fill="white", font=f)

    # Mark hottest and coldest point.
    for fn, color in ((np.argmax, (255, 255, 255)), (np.argmin, (120, 200, 255))):
        r, c = np.unravel_index(fn(temps), temps.shape)
        x, y = c * scale + scale // 2, r * scale + scale // 2
        d.ellipse([x - 7, y - 7, x + 7, y + 7], outline=color, width=2)
        d.text((x + 9, y - 8), f"{temps[r, c]:.1f}", fill=color, font=f)
    if label:
        d.text((6, 4), label, fill="white", font=f)
    return canvas


def thermal_readings(temps: np.ndarray) -> dict:
    """The numbers printed on the thermal view, as text, so a reader without the image can check them."""
    hot = np.unravel_index(np.argmax(temps), temps.shape)
    cold = np.unravel_index(np.argmin(temps), temps.shape)
    rows, cols = temps.shape

    def where(rc):
        r, c = rc
        v = "top" if r < rows / 3 else "bottom" if r > 2 * rows / 3 else "middle"
        h = "left" if c < cols / 3 else "right" if c > 2 * cols / 3 else "centre"
        return f"{v} {h}"

    return {
        "scale_c": [round(float(np.percentile(temps, 1)), 1), round(float(np.percentile(temps, 99.7)), 1)],
        "hottest_point_c": round(float(temps[hot]), 1), "hottest_point_at": where(hot),
        "coldest_point_c": round(float(temps[cold]), 1), "coldest_point_at": where(cold),
    }


def spectrogram(x: np.ndarray, width: int = 640, height: int = 220, max_hz: float = 4000) -> Image.Image:
    """Time left to right, pitch bottom to top, loudness as colour."""
    n_fft, hop = 1024, 256
    x = x / (np.abs(x).max() + 1e-9)
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop]
    spec = np.abs(np.fft.rfft(frames * np.hanning(n_fft), axis=1)).T
    freqs = np.fft.rfftfreq(n_fft, 1 / simulate.SR)
    spec = spec[freqs <= max_hz]
    db = 20 * np.log10(spec + 1e-9)
    db = np.clip((db - (db.max() - 70)) / 70, 0, 1)[::-1]
    img = Image.fromarray(_LUT[(db * 255).astype(np.uint8)]).resize((width, height), Image.BILINEAR)
    d = ImageDraw.Draw(img)
    f = _font(12)
    for hz in (500, 1000, 2000, 3000):
        y = int(height * (1 - hz / max_hz))
        d.line([(0, y), (6, y)], fill="white")
        d.text((9, y - 7), f"{hz // 1000 if hz >= 1000 else hz}{'k' if hz >= 1000 else ''} Hz", fill="white", font=f)
    return img


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def read_wav(data: bytes) -> np.ndarray:
    """Mono float samples at 16 kHz. Stereo is averaged; other rates are resampled."""
    with wave.open(io.BytesIO(data)) as w:
        ch, width, rate, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if width != 2:
        raise ValueError("Sound must be 16-bit WAV.")
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if rate != simulate.SR:
        t_old = np.arange(len(x)) / rate
        t_new = np.arange(int(len(x) * simulate.SR / rate)) / simulate.SR
        x = np.interp(t_new, t_old, x)
    return x


def write_wav(x: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(simulate.SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


def read_thermal_png(data: bytes) -> np.ndarray:
    """Radiometric 16-bit PNG -> temperatures in C. Rejects ordinary photos."""
    im = Image.open(io.BytesIO(data))
    if im.mode not in ("I;16", "I;16B", "I"):
        raise ValueError("Not a radiometric thermal image (expected a 16-bit greyscale PNG).")
    return simulate.png16_to_thermal(np.array(im))


def write_thermal_png(temps: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(simulate.thermal_to_png16(temps)).save(buf, format="PNG")
    return buf.getvalue()

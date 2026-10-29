"""Simulated reference signatures for a split AC, one per condition.

Everything here is GENERATED from simple physical models of how each fault changes
the unit. It is labelled "simulated" wherever it appears. Real recordings or photos
can be added to the library alongside it; they are labelled with their own source.

Signals:
  sound            16 kHz mono, recorded about 1 m from the outdoor unit
  thermal_indoor   radiometric thermal image of the indoor coil (front panel open)
  thermal_outdoor  radiometric thermal image of the outdoor unit, side panel off
                   (coil face on the left, compressor bay with capacitor on the right)

Thermal images are stored as 16-bit PNG where pixel = (temperature_C + 40) * 100,
the same idea as a real radiometric camera export: every pixel is a temperature.
"""

from __future__ import annotations

import numpy as np

SR = 16000
CONDITIONS = [
    "normal",
    "clogged_air_filter",
    "condenser_fan_failure",
    "low_refrigerant",
    "dirty_condenser_coil",
    "failed_compressor_capacitor",
    "faulty_room_thermistor",
]
THERMAL_SHAPE = (120, 160)  # rows, cols: a common low-cost thermal sensor resolution
THERMAL_OFFSET = 40.0
THERMAL_SCALE = 100.0


# ----------------------------------------------------------------------------- helpers

def _shaped_noise(n: int, rng: np.random.Generator, lo: float, hi: float, tilt: float = 1.0) -> np.ndarray:
    """Noise with a smooth band shape (no hard filter edges) and a 1/f^tilt slope."""
    spectrum = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    f_safe = np.maximum(f, 1.0)
    slope = f_safe ** (-tilt / 2)
    # Smooth roll-on below `lo` and roll-off above `hi` (2nd-order-like skirts).
    shape = slope / (1 + (lo / f_safe) ** 4) / (1 + (f_safe / hi) ** 4)
    out = np.fft.irfft(spectrum * shape, n)
    return out / (np.std(out) + 1e-12)


def _tone_series(t: np.ndarray, phase: np.ndarray, amps: list[float], rng) -> np.ndarray:
    out = np.zeros_like(t)
    for k, a in enumerate(amps, start=1):
        out += a * np.sin(k * phase + rng.uniform(0, 2 * np.pi))
    return out


def _envelope(n: int, on_from: float, on_to: float, ramp: float = 0.15) -> np.ndarray:
    """1 between on_from and on_to seconds, smooth ramps at the edges, 0 elsewhere."""
    t = np.arange(n) / SR
    rise = np.clip((t - on_from) / ramp, 0, 1)
    fall = np.clip((on_to - t) / ramp, 0, 1)
    return np.minimum(rise, fall)


def _click(n: int, at: float, rng, level: float = 0.6) -> np.ndarray:
    """A relay/overload click: short broadband burst with fast decay."""
    out = np.zeros(n)
    i = int(at * SR)
    length = int(0.012 * SR)
    if 0 <= i < n - length:
        burst = rng.standard_normal(length) * np.exp(-np.arange(length) / (0.0025 * SR))
        out[i:i + length] = level * burst
    return out


# ----------------------------------------------------------------------------- sound

def sound(condition: str, seconds: float = 10.0, seed: int = 0, severity: float = 1.0) -> np.ndarray:
    """Outdoor-unit sound for one condition. severity 0-1 (ignored for 'normal')."""
    rng = np.random.default_rng(seed)
    s = 0.0 if condition == "normal" else float(np.clip(severity, 0, 1))
    n = int(seconds * SR)
    t = np.arange(n) / SR
    mains = 50.0

    # Unit-to-unit variation.
    comp_hz = rng.uniform(46.5, 48.5)          # rotary compressor shaft speed (2-pole, with slip)
    fan_rpm = rng.uniform(780, 900)
    blades = int(rng.choice([3, 4]))
    comp_level = rng.uniform(0.8, 1.2)
    fan_level = rng.uniform(0.8, 1.2)
    ambient_level = rng.uniform(0.04, 0.1)

    comp_on = np.ones(n)
    fan_on = np.ones(n)
    comp_load = 1.0         # loudness of compressor
    comp_strain = 0.0       # extra upper harmonics when labouring
    fan_speed = 1.0
    extra = np.zeros(n)

    if condition == "clogged_air_filter":
        comp_load = 1 - 0.18 * s           # low suction pressure: compressor less loaded
        comp_hz *= 1 + 0.01 * s
    elif condition == "condenser_fan_failure":
        fan_speed = max(0.0, 1 - 1.15 * s)  # mild: fan slowing; severe: stopped
        comp_load = 1 + 0.45 * s
        comp_strain = 0.8 * s
        comp_hz *= 1 - 0.015 * s
        if s > 0.6:                         # high-pressure cutout during the clip
            stop = rng.uniform(4.5, 7.5)
            comp_on = _envelope(n, -1, stop, ramp=0.4)
            extra += _click(n, stop, rng, 0.5)
    elif condition == "low_refrigerant":
        comp_load = 1 - 0.3 * s
        comp_hz *= 1 + 0.012 * s
        extra += 0.035 * s * _shaped_noise(n, rng, 2500, 7000, tilt=0.2)   # faint hiss at a valve
    elif condition == "dirty_condenser_coil":
        comp_load = 1 + 0.3 * s
        comp_strain = 0.5 * s
        # Air forced through blocked fins: a whistling band.
        extra += 0.05 * s * fan_level * _shaped_noise(n, rng, 1200, 1900, tilt=0.0)
    elif condition == "failed_compressor_capacitor":
        # Locked-rotor attempt: loud magnetic hum, then the overload trips. Fan keeps running.
        start = rng.uniform(0.3, 2.5)
        trip = start + rng.uniform(2.0, 3.5)
        locked = _envelope(n, start, trip, ramp=0.05)
        hum = 0.9 * np.sin(2 * np.pi * mains * t) + 1.2 * np.sin(2 * np.pi * 2 * mains * t) \
            + 0.4 * np.sin(2 * np.pi * 3 * mains * t)
        extra += (0.35 + 0.35 * s) * locked * hum
        extra += _click(n, trip, rng, 0.7)
        comp_on = np.zeros(n)
    elif condition == "faulty_room_thermistor":
        # Indoor controller never asks for cooling: outdoor unit off.
        # Mild: it was still running at the start of the clip and then shut down.
        stop = (1 - s) * seconds * rng.uniform(0.6, 1.0)
        comp_on = _envelope(n, -1, stop, ramp=0.5) if stop > 0.5 else np.zeros(n)
        fan_on = comp_on.copy()
        if stop > 0.5:
            extra += _click(n, stop, rng, 0.4)

    sig = np.zeros(n)

    # Compressor: shaft harmonics, mains-related magnetic hum, broadband mechanical noise.
    comp_phase = 2 * np.pi * np.cumsum(np.full(n, comp_hz)) / SR
    amps = [0.30, 0.42, 0.22, 0.16, 0.10, 0.08, 0.05, 0.04, 0.03, 0.02]
    amps = [a * (1 + comp_strain * (k / 4)) for k, a in enumerate(amps, start=1)]
    comp = _tone_series(t, comp_phase, amps, rng)
    comp += 0.25 * np.sin(2 * np.pi * 2 * mains * t)
    comp += 0.35 * (1 + comp_strain) * _shaped_noise(n, rng, 180, 3200, tilt=0.6)
    sig += comp_level * comp_load * comp_on * comp

    # Outdoor fan: blade-pass tone family plus air rush.
    if fan_speed > 0:
        fan_hz = fan_rpm / 60 * fan_speed
        wobble = 1 + 0.003 * np.sin(2 * np.pi * rng.uniform(0.1, 0.4) * t)
        fan_phase = 2 * np.pi * np.cumsum(fan_hz * wobble) / SR
        fan = _tone_series(t, blades * fan_phase, [0.22, 0.09, 0.04], rng)
        fan += 0.45 * fan_speed ** 1.5 * _shaped_noise(n, rng, 120, 4500, tilt=1.0)
        sig += fan_level * fan_on * fan_speed * fan

    sig += extra

    # Surroundings: traffic and wind rumble, distant sounds.
    sig += ambient_level * _shaped_noise(n, rng, 40, 5000, tilt=1.2)

    # Recording chain: a wall reflection, microphone self-noise, random level.
    delay = int(rng.uniform(0.004, 0.02) * SR)
    sig[delay:] += rng.uniform(0.1, 0.3) * sig[:-delay]
    sig += 0.003 * rng.standard_normal(n)
    sig *= rng.uniform(0.5, 1.0) / (np.sqrt(np.mean(sig ** 2)) * 6 + 1e-9)
    fade = int(0.05 * SR)
    sig[:fade] *= np.linspace(0, 1, fade)
    sig[-fade:] *= np.linspace(1, 0, fade)
    return np.clip(sig, -1, 1)


# ----------------------------------------------------------------------------- thermal

def _fin_texture(shape, rng, horizontal: bool = True) -> np.ndarray:
    """Coil fins and tubes: fine stripes plus tube rows that run slightly different."""
    rows, cols = shape
    yy, xx = np.mgrid[0:rows, 0:cols]
    axis = yy if horizontal else xx
    stripes = 0.25 * np.sin(axis * rng.uniform(1.4, 1.8))
    tubes = 0.6 * (np.sin(axis * 2 * np.pi / rng.uniform(9, 12)) > 0.85)
    return stripes + tubes


def _blobs(shape, rng, count: int, size: tuple[float, float]) -> np.ndarray:
    rows, cols = shape
    yy, xx = np.mgrid[0:rows, 0:cols]
    out = np.zeros(shape)
    for _ in range(count):
        cy, cx = rng.uniform(0, rows), rng.uniform(0, cols)
        r = rng.uniform(*size)
        out = np.maximum(out, np.exp(-(((yy - cy) ** 2) + ((xx - cx) ** 2)) / (2 * r * r)))
    return out


def _lerp(a: float, b: float, s: float) -> float:
    return a + s * (b - a)


def _unit_temps(condition: str, rng) -> tuple[dict, dict]:
    """Healthy and full-fault temperatures for this particular unit (they vary per install)."""
    normal = {k: v + rng.normal(0, 2.5) for k, v in NORMAL_T.items()}
    fault = {k: normal[k] + (v - NORMAL_T[k]) * rng.uniform(0.75, 1.15) for k, v in FAULT_T[condition].items()}
    return normal, {**normal, **fault}


# Target temperatures (C) at full severity. Each fault moves from NORMAL towards these.
NORMAL_T = {"indoor_coil": 11.0, "outdoor_coil": 46.0, "compressor": 62.0, "capacitor": 40.0}
FAULT_T = {
    "normal": {},
    "clogged_air_filter": {"indoor_coil": 3.0, "outdoor_coil": 41.0, "compressor": 58.0},
    "condenser_fan_failure": {"indoor_coil": 19.0, "outdoor_coil": 64.0, "compressor": 88.0, "capacitor": 52.0},
    "low_refrigerant": {"indoor_coil": 22.0, "outdoor_coil": 39.0, "compressor": 82.0},
    "dirty_condenser_coil": {"indoor_coil": 15.0, "outdoor_coil": 57.0, "compressor": 78.0},
    "failed_compressor_capacitor": {"indoor_coil": 26.0, "outdoor_coil": 36.0, "compressor": 70.0, "capacitor": 68.0},
    "faulty_room_thermistor": {"indoor_coil": 27.0, "outdoor_coil": 35.0, "compressor": 36.0, "capacitor": 36.0},
}

# Fixed capture layout, shared by the generator and the feature extractor.
INDOOR_REGIONS = {
    "housing_top": (slice(0, 14), slice(0, 160)),
    "coil": (slice(18, 104), slice(8, 152)),
    "coil_inlet_end": (slice(18, 104), slice(8, 32)),
    "coil_bottom": (slice(76, 104), slice(8, 152)),
    "housing_bottom": (slice(108, 120), slice(0, 160)),
}
OUTDOOR_REGIONS = {
    "wall": (slice(0, 10), slice(0, 160)),
    "coil": (slice(14, 112), slice(4, 100)),
    "coil_top": (slice(14, 50), slice(4, 100)),
    "coil_bottom": (slice(76, 112), slice(4, 100)),
    "bay": (slice(14, 112), slice(106, 156)),
    "compressor": (slice(54, 108), slice(112, 150)),
    "capacitor": (slice(20, 40), slice(118, 140)),
}


def thermal_indoor(condition: str, seed: int = 0, severity: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    s = 0.0 if condition == "normal" else float(np.clip(severity, 0, 1))
    room = rng.uniform(24, 30)
    normal_t, targets = _unit_temps(condition, rng)
    coil_t = _lerp(normal_t["indoor_coil"], targets["indoor_coil"], s) + rng.normal(0, 0.6)

    img = np.full(THERMAL_SHAPE, room) + rng.normal(0, 0.25, THERMAL_SHAPE)
    r, c = INDOOR_REGIONS["coil"]
    h, w = r.stop - r.start, c.stop - c.start
    coil = np.full((h, w), coil_t) + 0.8 * _fin_texture((h, w), rng)
    # Refrigerant enters at the left end: slightly colder there in a healthy coil.
    coil += np.linspace(-0.8, 0.8, w)[None, :]

    if condition == "clogged_air_filter":
        ice = _blobs((h, w), rng, count=int(6 + 10 * s), size=(5, 12))
        ice *= np.clip((np.arange(h)[:, None] / h - 0.35) * 2.5, 0, 1)  # mostly the lower half
        coil = coil * (1 - ice * s) + (-3.0 + rng.normal(0, 0.4)) * ice * s
    elif condition == "low_refrigerant":
        # Starved coil: only the inlet end is cold (sometimes frosted); the rest is warm.
        frac = np.clip(np.arange(w) / w, 0, 1)[None, :]
        inlet_cold = np.exp(-frac / 0.12)
        coil = coil * (1 - inlet_cold) + (_lerp(9, -2, s)) * inlet_cold
    img[r, c] = coil
    img += rng.normal(0, 0.3, THERMAL_SHAPE)       # sensor noise
    img += rng.uniform(-2.5, 2.5)                  # emissivity / calibration offset
    return img


def thermal_outdoor(condition: str, seed: int = 0, severity: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    s = 0.0 if condition == "normal" else float(np.clip(severity, 0, 1))
    ambient = rng.uniform(28, 42)
    normal_t, targets = _unit_temps(condition, rng)

    def t_of(part: str) -> float:
        # Temperatures above ambient scale with how hot the day is.
        base = _lerp(normal_t[part], targets[part], s)
        return base + (ambient - 35) * 0.6 + rng.normal(0, 0.8)

    img = np.full(THERMAL_SHAPE, ambient) + rng.normal(0, 0.3, THERMAL_SHAPE)

    r, c = OUTDOOR_REGIONS["coil"]
    h, w = r.stop - r.start, c.stop - c.start
    coil = np.full((h, w), t_of("outdoor_coil")) + 0.9 * _fin_texture((h, w), rng, horizontal=False)
    if condition == "condenser_fan_failure":
        # No airflow: heat stratifies, top of the coil much hotter than the bottom.
        coil += s * np.linspace(6, -4, h)[:, None]
    if condition == "dirty_condenser_coil":
        dirt = _blobs((h, w), rng, count=int(4 + 8 * s), size=(6, 16))
        coil += 7 * s * dirt
    img[r, c] = coil

    rb, cb = OUTDOOR_REGIONS["bay"]
    img[rb, cb] = ambient + 2 + rng.normal(0, 0.3, (rb.stop - rb.start, cb.stop - cb.start))

    # Compressor: a rounded body, hottest at the top (discharge side).
    rc, cc = OUTDOOR_REGIONS["compressor"]
    h2, w2 = rc.stop - rc.start, cc.stop - cc.start
    yy, xx = np.mgrid[0:h2, 0:w2]
    body = (((xx - w2 / 2) / (w2 / 2)) ** 2 + ((yy - h2 / 2) / (h2 / 2)) ** 2) <= 1
    comp_t = t_of("compressor")
    comp = np.where(body, comp_t + np.linspace(4, -4, h2)[:, None], img[rc, cc])
    img[rc, cc] = comp

    rcap, ccap = OUTDOOR_REGIONS["capacitor"]
    img[rcap, ccap] = t_of("capacitor") + rng.normal(0, 0.4, (rcap.stop - rcap.start, ccap.stop - ccap.start))

    img += rng.normal(0, 0.35, THERMAL_SHAPE)
    img += rng.uniform(-2.5, 2.5)                  # emissivity / calibration offset
    return img


def thermal_to_png16(temps: np.ndarray) -> np.ndarray:
    return np.clip(np.round((temps + THERMAL_OFFSET) * THERMAL_SCALE), 0, 65535).astype(np.uint16)


def png16_to_thermal(raw: np.ndarray) -> np.ndarray:
    return raw.astype(np.float64) / THERMAL_SCALE - THERMAL_OFFSET

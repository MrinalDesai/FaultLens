"""Turn each signal into a fixed set of numbers the pattern model can learn from.

Every feature has a plain name, so the app can show which measurements drove a match.
Missing signals produce NaN features, which the model handles.
"""

from __future__ import annotations

import numpy as np

from . import sensors, simulate

SR = simulate.SR

# --------------------------------------------------------------------- sound

_BANDS = np.geomspace(30, 7800, 19)  # 18 log-spaced bands


def _stft_power(x: np.ndarray, n_fft: int = 1024, hop: int = 512) -> tuple[np.ndarray, np.ndarray]:
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop]
    spec = np.abs(np.fft.rfft(frames * np.hanning(n_fft), axis=1)) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1 / SR)
    return spec, freqs


def sound_features(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x / (np.sqrt(np.mean(x ** 2)) + 1e-9)  # loudness-independent: phone distance varies
    spec, freqs = _stft_power(x)
    total = spec.sum(axis=1) + 1e-12
    feats: dict[str, float] = {}

    band_energy = []
    for lo, hi in zip(_BANDS[:-1], _BANDS[1:]):
        m = (freqs >= lo) & (freqs < hi)
        e = spec[:, m].sum(axis=1) / total            # share of energy in this band, per frame
        band_energy.append(e)
        name = f"sound_band_{int(lo)}_{int(hi)}hz"
        feats[name + "_mean"] = float(np.log10(e.mean() + 1e-9))
        feats[name + "_var"] = float(np.std(np.log10(e + 1e-9)))

    # Tonal peaks: compressor, mains hum, fan blade-pass. Needs ~1 Hz resolution.
    fine, fine_f = _stft_power(x, n_fft=16384, hop=8192)
    full = fine.mean(axis=0)
    floor = np.median(full[(fine_f > 20) & (fine_f < 600)]) + 1e-12

    def peak(lo, hi):
        m = (fine_f >= lo) & (fine_f < hi)
        return float(np.log10(full[m].max() / floor + 1e-9))

    feats["sound_tone_40_55hz"] = peak(40, 55)
    feats["sound_tone_95_105hz"] = peak(95, 105)
    feats["sound_tone_140_160hz"] = peak(140, 160)

    # How the sound changes over time: switching on/off, trips, bursts.
    frame_db = 10 * np.log10(total)
    sec = SR // 512
    per_sec = np.array([frame_db[i:i + sec].mean() for i in range(0, len(frame_db) - sec + 1, sec)])
    feats["sound_level_range_db"] = float(per_sec.max() - per_sec.min()) if len(per_sec) else 0.0
    feats["sound_level_drop_db"] = float(per_sec[: len(per_sec) // 2].mean() - per_sec[len(per_sec) // 2:].mean()) \
        if len(per_sec) >= 2 else 0.0
    low = np.log10(np.stack(band_energy[:5]).sum(axis=0) + 1e-9)
    feats["sound_low_band_change"] = float(np.percentile(low, 95) - np.percentile(low, 5))
    flat = np.exp(np.mean(np.log(spec[:, freqs > 2000] + 1e-12), axis=1)) / (spec[:, freqs > 2000].mean(axis=1) + 1e-12)
    feats["sound_high_flatness"] = float(flat.mean())
    centroid = (spec * freqs).sum(axis=1) / total
    feats["sound_centroid_hz"] = float(np.median(centroid))
    return feats


# --------------------------------------------------------------------- thermal

def _region_stats(img: np.ndarray, regions: dict, prefix: str) -> dict[str, float]:
    out = {}
    for name, (r, c) in regions.items():
        patch = img[r, c]
        out[f"{prefix}_{name}_mean_c"] = float(patch.mean())
        out[f"{prefix}_{name}_min_c"] = float(np.percentile(patch, 2))
        out[f"{prefix}_{name}_max_c"] = float(np.percentile(patch, 98))
        out[f"{prefix}_{name}_spread_c"] = float(patch.std())
    return out


def thermal_indoor_features(img: np.ndarray) -> dict[str, float]:
    f = _region_stats(img, simulate.INDOOR_REGIONS, "indoor")
    room = (f["indoor_housing_top_mean_c"] + f["indoor_housing_bottom_mean_c"]) / 2
    f["indoor_coil_below_room_c"] = room - f["indoor_coil_mean_c"]
    f["indoor_inlet_vs_coil_c"] = f["indoor_coil_mean_c"] - f["indoor_coil_inlet_end_mean_c"]
    f["indoor_cold_spots_c"] = f["indoor_coil_mean_c"] - f["indoor_coil_bottom_min_c"]
    return f


def thermal_outdoor_features(img: np.ndarray) -> dict[str, float]:
    f = _region_stats(img, simulate.OUTDOOR_REGIONS, "outdoor")
    amb = f["outdoor_wall_mean_c"]
    f["outdoor_coil_above_ambient_c"] = f["outdoor_coil_mean_c"] - amb
    f["outdoor_compressor_above_ambient_c"] = f["outdoor_compressor_max_c"] - amb
    f["outdoor_capacitor_above_ambient_c"] = f["outdoor_capacitor_mean_c"] - amb
    f["outdoor_coil_top_minus_bottom_c"] = f["outdoor_coil_top_mean_c"] - f["outdoor_coil_bottom_mean_c"]
    return f


# --------------------------------------------------------------------- sensors

def sensor_features(rows: list[dict]) -> dict[str, float]:
    summary = sensors.summarize(rows)
    f = {}
    for ch in sensors.CHANNELS:
        st = summary["stats"].get(ch)
        f[f"sensor_{ch}_median"] = st["median"] if st else np.nan
        f[f"sensor_{ch}_max"] = st["max"] if st else np.nan
    for k in ("delta_t_c", "room_sensor_offset_c", "compressor_off_minutes", "compressor_spikes_over_20a"):
        f[f"sensor_{k}"] = summary["derived"].get(k, np.nan)
    return f


# --------------------------------------------------------------------- plain-language labels

_LABELS = {
    "sound_high_flatness": "Hiss in the high frequencies",
    "sound_level_range_db": "How much the loudness changes during the clip (dB)",
    "sound_level_drop_db": "Loudness drop from first half to second half (dB)",
    "sound_low_band_change": "Change in low rumble over the clip",
    "sound_centroid_hz": "Overall pitch of the sound (Hz)",
    "sound_tone_40_55hz": "Strength of the 40-55 Hz tone (compressor and fan)",
    "sound_tone_95_105hz": "Strength of the 100 Hz electrical hum",
    "sound_tone_140_160hz": "Strength of the 150 Hz hum",
    "indoor_coil_below_room_c": "Indoor coil, degrees colder than the room",
    "indoor_inlet_vs_coil_c": "Indoor coil, inlet end colder than the rest by (C)",
    "indoor_cold_spots_c": "Indoor coil, coldest spots below the coil average (C)",
    "outdoor_coil_above_ambient_c": "Outdoor coil, degrees above outside air",
    "outdoor_compressor_above_ambient_c": "Compressor, degrees above outside air",
    "outdoor_capacitor_above_ambient_c": "Capacitor, degrees above outside air",
    "outdoor_coil_top_minus_bottom_c": "Outdoor coil, top hotter than bottom by (C)",
    "sensor_delta_t_c": "Air temperature drop across the indoor unit (C)",
    "sensor_room_sensor_offset_c": "Unit's room reading minus actual room air (C)",
    "sensor_compressor_off_minutes": "Minutes the compressor was off",
    "sensor_compressor_spikes_over_20a": "Compressor current spikes over 20 A",
}
_CHANNELS = {
    "return_air_c": "return air temperature (C)", "supply_air_c": "supply air temperature (C)",
    "outdoor_air_c": "outside air temperature (C)", "suction_psig": "suction pressure (psi)",
    "discharge_psig": "discharge pressure (psi)", "compressor_amps": "compressor current (A)",
    "outdoor_fan_amps": "outdoor fan current (A)", "room_sensor_c": "unit's own room reading (C)",
}
_REGIONS = {
    "housing_top": "casing above the coil", "housing_bottom": "casing below the coil", "coil": "whole coil",
    "coil_inlet_end": "inlet end of the coil", "coil_bottom": "bottom of the coil", "coil_top": "top of the coil",
    "wall": "wall behind the unit", "bay": "compressor bay", "compressor": "compressor", "capacitor": "capacitor",
}
_STATS = {"mean": "average", "min": "coldest point", "max": "hottest point", "spread": "unevenness",
          "median": "typical", "var": "variation"}


def label(name: str) -> str:
    """Plain-language name for a feature, for the console and the agent."""
    if name in _LABELS:
        return _LABELS[name]
    if name.startswith("sound_band_"):
        lo, hi = name.split("_")[2], name.split("_")[3].replace("hz", "")
        kind = "Variation" if name.endswith("_var") else "Share"
        return f"{kind} of sound at {lo}-{hi} Hz"
    if name.startswith("sensor_"):
        body, stat = name[len("sensor_"):].rsplit("_", 1)
        stat_word = {"median": "Typical", "max": "Highest"}.get(stat, stat)
        return f"{stat_word} {_CHANNELS.get(body, body.replace('_', ' '))}"
    for prefix, place in (("indoor_", "Indoor coil image"), ("outdoor_", "Outdoor unit image")):
        if name.startswith(prefix):
            body = name[len(prefix):].removesuffix("_c")
            region, stat = body.rsplit("_", 1)
            return f"{place}, {_REGIONS.get(region, region.replace('_', ' '))}: {_STATS.get(stat, stat)} (C)"
    return name.replace("_", " ")


# --------------------------------------------------------------------- names

MODALITIES = ["sound", "thermal_indoor", "thermal_outdoor", "sensors"]
EXTRACTORS = {
    "sound": sound_features,
    "thermal_indoor": thermal_indoor_features,
    "thermal_outdoor": thermal_outdoor_features,
    "sensors": sensor_features,
}

"""Parts of the split AC: the numbered exploded view, a photo of each healthy part,
and which parts each fault affects.

The faulty part is outlined on the exploded view. The outlines come from SAM 3.1 when
`python -m faultlens.parts --sam` has been run once (it saves library/diagram/parts_sam.json);
otherwise from boxes drawn by hand. The app says which.

SAM is given the part name and finds every match on the diagram; we keep the match that best
overlaps the hand-drawn box, so it picks the right one of, say, two capacitors.
"""

from __future__ import annotations

import argparse
import json

from PIL import Image, ImageDraw

from . import library

DIAGRAM = library.LIBRARY / "diagram" / "split_ac_exploded.png"
SAM_BOXES = library.LIBRARY / "diagram" / "parts_sam.json"
PART_DIR = library.LIBRARY / "parts"

# number: (name, unit, image file or None, SAM phrase, hand box x1, y1, x2, y2 on the 1672x941 diagram)
PARTS = {
    1: ("Front panel and grille", "Indoor", None, "front panel", (15, 180, 415, 530)),
    2: ("Air filters", "Indoor", "p02_air_filter.jpg", "air filter", (165, 180, 420, 360)),
    3: ("Evaporator (indoor) coil", "Indoor", "p03_evaporator_coil.jpg", "evaporator coil", (360, 110, 740, 335)),
    4: ("Blower wheel", "Indoor", None, "blower wheel", (430, 235, 740, 435)),
    5: ("Indoor fan motor", "Indoor", None, "small motor", (745, 355, 835, 460)),
    6: ("Room temperature sensor", "Indoor", "p06_room_thermistor.jpg", "sensor with wire", (150, 470, 320, 580)),
    7: ("Indoor control board", "Indoor", None, "circuit board", (680, 480, 780, 560)),
    8: ("Drain pan and hose", "Indoor", None, "drain pan", (405, 435, 880, 690)),
    9: ("Display and remote", "Indoor", "p09_display_remote.jpg", "remote control", (140, 595, 400, 830)),
    10: ("Condenser (outdoor) coil", "Outdoor", "p10_condenser_coil.jpg", "condenser coil", (985, 110, 1340, 450)),
    11: ("Outdoor fan blade", "Outdoor", "p11_fan_blade.jpg", "fan blade", (1005, 295, 1190, 570)),
    12: ("Outdoor fan motor", "Outdoor", "p12_fan_motor.jpg", "electric motor", (1170, 365, 1265, 470)),
    13: ("Fan capacitor (small)", "Outdoor", "p13_fan_capacitor.jpg", "capacitor", (1280, 355, 1325, 430)),
    14: ("Compressor", "Outdoor", None, "compressor", (1335, 425, 1530, 690)),
    15: ("Compressor run capacitor (large)", "Outdoor", "p15_run_capacitor.jpg", "capacitor", (1420, 300, 1490, 410)),
    16: ("Service valves and flare nuts", "Outdoor", "p16_service_valves.jpg", "brass valve", (1570, 560, 1630, 700)),
    17: ("Copper pipes, insulated", "Between units", "p17_copper_lines.jpg", "insulated pipe", (430, 620, 1290, 860)),
    18: ("Contactor (relay)", "Outdoor", None, "relay", (1565, 340, 1622, 415)),
}

# Parts each fault involves (the cause first). Faults with several possible causes list them per cause.
FAULT_PARTS = {
    "normal": {"": []},
    "clogged_air_filter": {"": [2, 3]},
    "condenser_fan_failure": {"bearing": [11, 12], "motor": [12], "capacitor": [13], "": [11, 12, 13]},
    "low_refrigerant": {"flare": [16], "pipe": [17], "hidden": [16, 17], "": [16, 17, 3]},
    "dirty_condenser_coil": {"": [10]},
    "failed_compressor_capacitor": {"": [15, 14]},
    "faulty_room_thermistor": {"": [6]},
}


def parts_for(condition: str, variant: str = "") -> list[int]:
    m = FAULT_PARTS.get(condition, {})
    return m.get(variant) or m.get("", [])


def boxes() -> tuple[dict[int, tuple], str]:
    """Outline per part number, and where the outlines came from."""
    hand = {n: p[4] for n, p in PARTS.items()}
    if SAM_BOXES.exists():
        try:
            sam = {int(k): tuple(v) for k, v in json.loads(SAM_BOXES.read_text())["boxes"].items()}
            return {**hand, **sam}, "SAM 3.1" if len(sam) == len(hand) else f"SAM 3.1 ({len(sam)} parts), rest drawn by hand"
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
    return hand, "drawn by hand"


def listing() -> list[dict]:
    used = {}
    for cond, by_cause in FAULT_PARTS.items():
        for nums in by_cause.values():
            for n in nums:
                used.setdefault(n, set()).add(cond)
    out = []
    for n, (name, unit, img, _, _) in PARTS.items():
        out.append({"no": n, "name": name, "unit": unit,
                    "image": f"/library/parts/{img}" if img and (PART_DIR / img).exists() else None,
                    "faults": sorted(used.get(n, []))})
    return out


def highlight(numbers: list[int], width: int = 1400) -> Image.Image:
    """The exploded view with the given parts outlined and the rest dimmed."""
    base = Image.open(DIAGRAM).convert("RGB")
    bx, _ = boxes()
    if numbers:
        dim = Image.blend(base, Image.new("RGB", base.size, "white"), 0.55)
        mask = Image.new("L", base.size, 0)
        md = ImageDraw.Draw(mask)
        for n in numbers:
            md.rectangle(bx[n], fill=255)
        base = Image.composite(base, dim, mask)
        d = ImageDraw.Draw(base)
        for i, n in enumerate(numbers):
            colour = "#C8372D" if i == 0 else "#E08A2C"
            d.rectangle(bx[n], outline=colour, width=5)
    base.thumbnail((width, width))
    return base


def _iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter + 1e-9)


# Extra names to try per part, after the main one.
ALT_PHRASES = {
    1: ["plastic cover", "panel"], 2: ["mesh filter", "filter"], 3: ["heat exchanger", "radiator"],
    4: ["fan wheel", "cylinder"], 5: ["motor"], 6: ["cable", "sensor"], 7: ["circuit board"],
    8: ["tray", "hose"], 9: ["remote", "display"], 10: ["heat exchanger", "radiator"],
    11: ["propeller", "fan"], 12: ["motor"], 13: ["cylinder"], 14: ["black cylinder", "tank"],
    15: ["cylinder"], 16: ["valve", "fitting"], 17: ["pipe", "hose"], 18: ["switch", "electrical component"],
}


PAIRS = {2, 9, 16, 17}   # parts drawn as two items; SAM's matches are joined into one outline


def _inside(box, area) -> float:
    """Share of `box` that lies inside `area`."""
    x1, y1, x2, y2 = max(box[0], area[0]), max(box[1], area[1]), min(box[2], area[2]), min(box[3], area[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / max(1e-9, (box[2] - box[0]) * (box[3] - box[1]))


def _crop_around(box, W, H, margin=0.35):
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * margin + 20, (y2 - y1) * margin + 20
    return (max(0, int(x1 - mx)), max(0, int(y1 - my)), min(W, int(x2 + mx)), min(H, int(y2 + my)))


def run_sam() -> None:
    """For each part, show SAM a close-up around where the part is and ask for it by name
    (trying a few names). Keep the match that best overlaps the hand-drawn box."""
    from . import sam
    img = Image.open(DIAGRAM).convert("RGB")
    W, H = img.size
    found, log = {}, {}
    for n, (name, _, _, phrase, hand) in PARTS.items():
        crop_box = _crop_around(hand, W, H)
        crop = img.crop(crop_box)
        cw, ch = crop.size
        tried, kept = [], []
        hx1, hy1, hx2, hy2 = hand
        mx, my = (hx2 - hx1) * 0.1, (hy2 - hy1) * 0.1
        area = (hx1 - mx, hy1 - my, hx2 + mx, hy2 + my)   # hand box plus 10%
        for ph in [phrase] + ALT_PHRASES.get(n, []):
            try:
                dets = sam.segment_image(crop, ph)
            except Exception as e:
                tried.append(f"'{ph}': error {e}")
                continue
            ok = 0
            for d in dets:
                sx, sy = cw / (d.get("w") or cw), ch / (d.get("h") or ch)
                box = (crop_box[0] + d["x1"] * sx, crop_box[1] + d["y1"] * sy,
                       crop_box[0] + d["x2"] * sx, crop_box[1] + d["y2"] * sy)
                # Must lie mostly inside the part's area: stops a neighbouring part being taken.
                if _inside(box, area) >= 0.7:
                    kept.append(box)
                    ok += 1
            tried.append(f"'{ph}': {len(dets)} ({ok} on the part)")
            if kept and n not in PAIRS and max(_iou(b, hand) for b in kept) >= 0.5:
                break
        best = None
        if kept:
            if n in PAIRS:   # join the matches into one outline
                best = (min(b[0] for b in kept), min(b[1] for b in kept), max(b[2] for b in kept), max(b[3] for b in kept))
            else:
                best = max(kept, key=lambda b: _iou(b, hand))
        cover = _iou(best, hand) if best else 0.0
        if best and cover >= 0.2:
            found[n] = [round(v) for v in best]
            log[n] = f"found (overlap with hand box {cover:.2f}); tried {', '.join(tried)}"
        else:
            log[n] = f"not found, hand box kept; tried {', '.join(tried)}"
        print(f"#{n:2d} {name}: {log[n]}")
    SAM_BOXES.write_text(json.dumps({"model": "sam-3.1", "method": "close-up crop per part", "boxes": found,
                                     "log": log}, indent=1))
    check = img.copy()
    d = ImageDraw.Draw(check)
    for n, p in PARTS.items():
        d.rectangle(p[4], outline="#C8372D", width=2)
        if n in found:
            d.rectangle(found[n], outline="#1E9E4A", width=4)
    out = SAM_BOXES.with_name("parts_sam_check.jpg")
    check.save(out, quality=85)
    print(f"\nSAM outlined {len(found)} of {len(PARTS)} parts. Saved {SAM_BOXES.name} and {out.name} "
          f"(green = SAM, red = drawn by hand) in library/diagram/.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Outline the parts on the exploded view with SAM 3.1.")
    ap.add_argument("--sam", action="store_true", help="run SAM 3.1 once and save the outlines")
    if ap.parse_args().sam:
        run_sam()
    else:
        ap.print_help()

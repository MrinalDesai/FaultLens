"""Handwritten service job cards: previous visits to each unit, as a technician would have filled them in.

Each card is a printed form (fictional company, fictional customers) filled in by hand,
then "scanned": slight tilt, paper tint, ink variation, noise. The true contents are saved
next to each card so the reader can be scored field by field.

Build:  python -m faultlens.jobcards
Files:  library/jobcards/<condition>/<card_id>.pdf / .jpg / .json
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from . import library

FONT_DIR = Path(__file__).parent / "fonts"
HANDS = ["caveat", "kalam", "nanum-pen-script", "homemade-apple"]
OUT = library.LIBRARY / "jobcards"

FIELDS = ["job_no", "date", "customer", "area", "unit_model", "serial_no", "complaint",
          "suction_psi", "compressor_amps", "supply_air_c", "work_done", "parts_used", "remarks",
          "next_service", "technician"]
LABELS = {
    "job_no": "Job no.", "date": "Date", "customer": "Customer", "area": "Area", "unit_model": "Unit model",
    "serial_no": "Serial no.", "complaint": "Complaint", "suction_psi": "Suction (psi)",
    "compressor_amps": "Compressor (A)", "supply_air_c": "Supply air (C)", "work_done": "Work done",
    "parts_used": "Parts used", "remarks": "Remarks", "next_service": "Next service", "technician": "Technician",
}

# Previous visits. Each hints at what is going on now without giving it away.
VISITS = {
    "normal": [
        dict(complaint="Annual service", suction_psi="132", compressor_amps="6.4", supply_air_c="13",
             work_done="Filters cleaned, coil washed, drain flushed", parts_used="None",
             remarks="All readings normal"),
        dict(complaint="Cooling feels less", suction_psi="128", compressor_amps="6.6", supply_air_c="14",
             work_done="Checked, no fault found. Explained set temp", parts_used="None",
             remarks="Customer had set temp at 26"),
    ],
    "clogged_air_filter": [
        dict(complaint="Weak air flow", suction_psi="104", compressor_amps="5.6", supply_air_c="8",
             work_done="Filters very dusty, cleaned. Ice on coil, defrosted", parts_used="None",
             remarks="Advised filter cleaning every month"),
        dict(complaint="Water dripping indoor", suction_psi="110", compressor_amps="5.8", supply_air_c="9",
             work_done="Drain cleared. Filters dusty, cleaned", parts_used="None",
             remarks="Construction work next door, lot of dust"),
    ],
    "condenser_fan_failure": [
        dict(complaint="Outdoor unit noisy", suction_psi="130", compressor_amps="7.1", supply_air_c="14",
             work_done="Outdoor fan bearing noisy, lubricated", parts_used="None",
             remarks="Fan motor to be replaced if noise returns"),
        dict(complaint="Tripping in afternoon", suction_psi="136", compressor_amps="8.9", supply_air_c="16",
             work_done="Fan slow to start. Tapped blade, started", parts_used="None",
             remarks="Check fan capacitor next visit"),
    ],
    "low_refrigerant": [
        dict(complaint="Not cooling enough", suction_psi="78", compressor_amps="4.1", supply_air_c="19",
             work_done="Low gas. Topped up R32 0.4 kg. Leak not traced", parts_used="R32 0.4 kg",
             remarks="Oil mark at suction valve, check next time"),
        dict(complaint="Slow cooling", suction_psi="84", compressor_amps="4.4", supply_air_c="18",
             work_done="Gas top-up R32 0.3 kg", parts_used="R32 0.3 kg",
             remarks="Second top-up this year, leak likely"),
    ],
    "dirty_condenser_coil": [
        dict(complaint="Poor cooling in afternoon", suction_psi="138", compressor_amps="9.4", supply_air_c="16",
             work_done="Outdoor coil choked with dust. Coil cleaning advised", parts_used="None",
             remarks="Customer declined coil wash"),
        dict(complaint="High bill", suction_psi="134", compressor_amps="9.1", supply_air_c="15",
             work_done="Outdoor unit boxed in by plants. Advised clearance", parts_used="None",
             remarks="Unit in balcony corner, no airflow"),
    ],
    "failed_compressor_capacitor": [
        dict(complaint="AC not starting after power cut", suction_psi="128", compressor_amps="7.9", supply_air_c="14",
             work_done="Voltage low 190 V. Started after reset", parts_used="None",
             remarks="Stabiliser recommended. Compressor hum at start"),
        dict(complaint="Humming sound outside", suction_psi="130", compressor_amps="7.6", supply_air_c="14",
             work_done="Capacitor tested 31 uF (rated 35). Within limit", parts_used="None",
             remarks="Capacitor weak, replace soon"),
    ],
    "faulty_room_thermistor": [
        dict(complaint="AC stops by itself", suction_psi="131", compressor_amps="6.2", supply_air_c="14",
             work_done="Room sensor wire loose, re-clipped", parts_used="Cable tie",
             remarks="Sensor holder broken"),
        dict(complaint="Display shows wrong temp", suction_psi="127", compressor_amps="6.0", supply_air_c="13",
             work_done="Reset unit. Sensor reading checked", parts_used="None",
             remarks="Replace room sensor if it repeats"),
    ],
}
CUSTOMERS = ["A. Kulkarni", "R. Shah", "S. Fernandes", "P. Iyer", "M. Deshpande", "N. Joshi", "K. Rao",
             "V. Pillai", "T. Gokhale", "L. D'Souza", "H. Patil", "J. Menon", "D. Bhosale", "G. Nair"]
AREAS = ["Kothrud", "Baner", "Aundh", "Viman Nagar", "Hadapsar", "Wakad", "Kharadi"]
TECHS = ["Ravi", "Sunil", "Imran", "Joseph", "Anil"]
MONTHS = ["Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]


def _font(name: str, size: int):
    return ImageFont.truetype(str(FONT_DIR / f"{name}.ttf"), size)


def _print_font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _hand(img: Image.Image, xy: tuple[int, int], text: str, font_name: str, size: int, rng: random.Random,
          max_width: int) -> None:
    """Write text word by word with a wandering baseline, slight slant and varying ink."""
    x0, y = xy
    x = x0
    slant = rng.uniform(-2.5, 2.5)
    for word in text.split(" "):
        f = _font(font_name, int(size * rng.uniform(0.92, 1.08)))
        w = int(f.getlength(word + " ")) + 4
        if x + w > x0 + max_width:            # wrap like a person running out of line
            x, y = x0, y + int(size * 1.15)
        ink = (rng.randint(15, 40), rng.randint(35, 70), rng.randint(120, 170), 255)  # blue ballpoint
        tile = Image.new("RGBA", (w + 20, int(size * 1.8)), (0, 0, 0, 0))
        ImageDraw.Draw(tile).text((10, int(size * 0.2)), word, font=f, fill=ink)
        tile = tile.rotate(slant + rng.uniform(-1.5, 1.5), resample=Image.BICUBIC, expand=False)
        img.alpha_composite(tile, (int(x), int(y + rng.uniform(-3, 3))))
        x += w


def render(card: dict, rng: random.Random) -> Image.Image:
    W, H = 1240, 1754                                   # A4 at 150 dpi
    page = Image.new("RGBA", (W, H), (250, 248, 240, 255))
    d = ImageDraw.Draw(page)
    pf, pf_small, pf_big = _print_font(24), _print_font(19), _print_font(40)
    ink = (60, 60, 70)
    d.text((80, 70), "Coolfix Home Services (demo)", font=pf_big, fill=ink)
    d.text((80, 122), "Service job card. Air conditioners. Pune.", font=pf_small, fill=ink)
    d.line([(80, 160), (W - 80, 160)], fill=ink, width=3)

    hand = card["_hand"]
    rows = [("job_no", "date"), ("customer", "area"), ("unit_model", "serial_no"), ("complaint",),
            ("suction_psi", "compressor_amps", "supply_air_c"), ("work_done",), ("parts_used",), ("remarks",),
            ("next_service", "technician")]
    y = 200
    for row in rows:
        tall = len(row) == 1 and row[0] in ("complaint", "work_done", "remarks")
        h = 150 if tall else 95
        cols = len(row)
        cw = (W - 160) // cols
        for i, key in enumerate(row):
            x = 80 + i * cw
            d.rectangle([x, y, x + cw - 10, y + h], outline=(150, 150, 160), width=2)
            d.text((x + 10, y + 8), LABELS[key], font=pf_small, fill=(110, 110, 120))
            value = card[key]
            if value:
                _hand(page, (x + 14, y + 34), value, hand, card["_size"], rng, cw - 40)
        y += h + 14

    # Tick boxes, printed + handwritten ticks
    y += 6
    for i, (label, ticked) in enumerate([("Filters cleaned", "filter" in card["work_done"].lower()),
                                         ("Gas topped up", "R32" in card["parts_used"]),
                                         ("Customer informed", True)]):
        x = 80 + i * 360
        d.rectangle([x, y, x + 30, y + 30], outline=ink, width=2)
        d.text((x + 42, y + 2), label, font=pf, fill=ink)
        if ticked:
            _hand(page, (x - 2, y - 14), "v", hand, 46, rng, 60)
    y += 70
    d.text((80, y), "Customer signature", font=pf_small, fill=(110, 110, 120))
    d.line([(80, y + 70), (520, y + 70)], fill=(150, 150, 160), width=2)
    _hand(page, (90, y + 18), card["customer"].split(" ")[-1], hand, card["_size"] + 6, rng, 400)

    # "Scan": tilt, blur, paper texture, noise, uneven light.
    img = page.convert("RGB").rotate(rng.uniform(-1.4, 1.4), resample=Image.BICUBIC, expand=False,
                                     fillcolor=(235, 233, 225))
    img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.8)))
    a = np.asarray(img).astype(np.float32)
    nprng = np.random.default_rng(rng.randint(0, 10**6))
    a += nprng.normal(0, 4, a.shape)
    yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    light = 1 - 0.08 * (((xx - a.shape[1] * rng.uniform(.3, .7)) / a.shape[1]) ** 2 + ((yy / a.shape[0]) - .5) ** 2)
    a *= light[..., None]
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def _six_months_on(month: str) -> str:
    """Next service six months after the visit, rolling into the next year if needed."""
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    i = names.index(month) + 6
    return f"{names[i % 12]} {2026 + i // 12}"


def build(seed: int = 7) -> None:
    rng = random.Random(seed)
    count = 0
    for cond, visits in VISITS.items():
        folder = OUT / cond
        folder.mkdir(parents=True, exist_ok=True)
        for i, v in enumerate(visits):
            month = MONTHS[rng.randrange(len(MONTHS))]
            card = {
                "job_no": f"CF-{rng.randint(1000, 9999)}",
                "date": f"{rng.randint(1, 28)} {month} 2026",
                "customer": rng.choice(CUSTOMERS), "area": rng.choice(AREAS),
                "unit_model": rng.choice(["Coolwave CS-15F", "Coolwave CS-18F", "Arctiq S12-R32"]),
                "serial_no": f"{rng.choice('ABKMT')}{rng.randint(10, 99)}-{rng.randint(10000, 99999)}",
                **v,
                "next_service": _six_months_on(month) if rng.random() < .8 else "",
                "technician": rng.choice(TECHS),
            }
            card["_hand"] = HANDS[(count + rng.randint(0, 1)) % len(HANDS)]
            card["_size"] = {"caveat": 40, "kalam": 32, "nanum-pen-script": 44, "homemade-apple": 27}[card["_hand"]]
            img = render(card, rng)
            cid = f"{cond}_card_{i + 1}"
            img.save(folder / f"{cid}.jpg", quality=82, optimize=True)
            img.save(folder / f"{cid}.pdf", "PDF", resolution=150, quality=80)
            truth = {k: card[k] for k in FIELDS}
            (folder / f"{cid}.json").write_text(json.dumps({
                "condition": cond, "handwriting_font": card["_hand"], "fields": truth,
                "note": "Generated job card. Fictional company and customers. These are the true contents."},
                indent=1))
            count += 1
    print(f"Wrote {count} job cards to {OUT}")


def listing() -> list[dict]:
    out = []
    for js in sorted(OUT.glob("*/*.json")):
        meta = json.loads(js.read_text())
        cid = js.stem
        cond = js.parent.name
        out.append({"id": cid, "condition": cond, "image": f"/library/jobcards/{cond}/{cid}.jpg",
                    "pdf": f"/library/jobcards/{cond}/{cid}.pdf", "date": meta["fields"]["date"],
                    "complaint": meta["fields"]["complaint"], "handwriting": meta["handwriting_font"]})
    return out


def truth_for(card_id: str) -> dict | None:
    for js in OUT.glob(f"*/{card_id}.json"):
        return json.loads(js.read_text())["fields"]
    return None


def image_for(card_id: str) -> Image.Image | None:
    for jpg in OUT.glob(f"*/{card_id}.jpg"):
        return Image.open(jpg).convert("RGB")
    return None


if __name__ == "__main__":
    build()

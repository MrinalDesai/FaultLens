"""Check a part from a photo.

The technician photographs a part. Muse Spark compares it with the reference photo of the
same part when healthy and with reference photos of that part damaged, then says: which part
it is, whether it is damaged, whether to replace or clean it, and which fault the damage
causes (or whether it is only a symptom of another fault). SAM 3.1 then outlines the part in
the photo.

If the technician does not say which part it is, Muse Spark identifies it from a numbered
sheet of all the healthy part photos.

When a photo from the library is checked, that exact photo is left out of the references,
so the model cannot simply match it to itself, and the known answer is shown next to the result.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from . import config, knowledge, library, parts
from .meta_client import client, image_to_data_url, text_part

FIX = {"replace": "Replace the part", "clean": "Clean it, no replacement needed",
       "no_action": "No action, the part is fine", "check_further": "Check further before deciding"}
ROLE_TEXT = {"cause": "The damage causes the fault", "symptom": "This is a symptom of another fault, not the cause",
             "not_related": "Not related to the fault", "healthy": "The part is healthy"}

SYSTEM = """You are FaultLens, checking one part of a {equipment} from a technician's photo.
You get the technician's photo first, then reference photos. Compare the technician's photo with the
reference photo of the healthy part and with the reference photos of that part damaged.
Judge what you see in the technician's photo. A reference is a guide, not the answer: the technician's
part may be damaged in a way none of the references shows, or may be healthy.
Say plainly, in words a technician uses on site, what you see. Do not guess beyond the photo.
Some damage is the cause of a fault (a burnt motor, a swollen capacitor); some is only a symptom of a
fault elsewhere (frost on the indoor coil is caused by a blocked filter or low refrigerant): say which.
Call the assess_part tool exactly once."""


# ----------------------------------------------------------------------------- references

def _fault_titles() -> dict:
    return {f["fault_id"]: f["title"] for f in knowledge.fault_menu()}


def damaged_references(part_no: int, exclude: Path | None = None) -> list[dict]:
    """Every library photo of this part, from the fault folders, with what it shows."""
    out = []
    titles = _fault_titles()
    for meta_path in sorted(library.LIBRARY.glob("*/photos/photos.json")):
        cond = meta_path.parent.parent.name
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            continue
        for name, e in meta.items():
            path = meta_path.parent / name
            if not isinstance(e, dict) or e.get("part") != part_no or not path.exists():
                continue
            if exclude is not None and path.resolve() == exclude.resolve():
                continue
            if cond == "normal":
                continue
            out.append({"path": path, "caption": e.get("caption", ""), "fault": cond,
                        "fault_title": titles.get(cond, cond), "role": e.get("role", "cause")})
    return out


def healthy_reference(part_no: int, exclude: Path | None = None) -> Path | None:
    img = parts.PARTS[part_no][2]
    p = parts.PART_DIR / img if img else None
    if p is None or not p.exists() or (exclude is not None and p.resolve() == exclude.resolve()):
        return None
    return p


def parts_sheet(exclude: Path | None = None) -> Image.Image:
    """All healthy part photos on one numbered sheet, for identifying an unknown part."""
    tiles = [(n, healthy_reference(n, exclude)) for n in parts.PARTS]
    tiles = [(n, p) for n, p in tiles if p]
    size, cols = 260, 4
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * size, rows * (size + 34)), "white")
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default(size=22)
    except TypeError:
        font = ImageFont.load_default()
    for i, (n, p) in enumerate(tiles):
        im = Image.open(p).convert("RGB")
        im.thumbnail((size - 10, size - 10))
        x, y = (i % cols) * size, (i // cols) * (size + 34)
        sheet.paste(im, (x + 5, y + 5))
        d.text((x + 8, y + size + 4), f"#{n} {parts.PARTS[n][0]}"[:26], fill="black", font=font)
    return sheet


def known_answer(path: Path | None) -> dict | None:
    """For a library photo: what it really shows."""
    if path is None:
        return None
    path = path.resolve()
    if path.parent.resolve() == parts.PART_DIR.resolve():
        for n, p in parts.PARTS.items():
            if p[2] == path.name:
                return {"part_no": n, "condition": "healthy", "caption": f"Healthy {p[0].lower()}", "fault": None,
                        "role": "healthy"}
    meta_path = path.parent / "photos.json"
    if meta_path.exists():
        e = json.loads(meta_path.read_text()).get(path.name)
        if isinstance(e, dict) and e.get("part"):
            cond = path.parent.parent.name
            healthy = cond == "normal"
            return {"part_no": e["part"], "condition": "healthy" if healthy else "damaged", "caption": e.get("caption", ""),
                    "fault": None if healthy else cond, "role": e.get("role", "cause")}
    return None


# ----------------------------------------------------------------------------- the model call

def _tool() -> dict:
    return {"type": "function", "name": "assess_part",
            "description": "Your assessment of the part in the technician's photo.",
            "parameters": {"type": "object", "properties": {
                "part_no": {"type": "integer", "enum": list(parts.PARTS), "description": "Which part this is."},
                "condition": {"type": "string", "enum": ["healthy", "damaged", "unclear"]},
                "what_i_see": {"type": "string", "description": "1-2 plain sentences on what the photo shows."},
                "closest_reference": {"type": "string", "description": "Which reference it looks most like, or 'none'."},
                "action": {"type": "string", "enum": list(FIX)},
                "role": {"type": "string", "enum": list(ROLE_TEXT),
                         "description": "Whether this damage causes a fault, is a symptom of another fault, or is unrelated."},
                "faults": {"type": "array", "items": {"type": "string", "enum": knowledge.fault_ids()},
                           "description": "The faults this finding points to, most likely first."},
                "matches_complaint": {"type": "string", "enum": ["yes", "no", "no complaint given"]},
                "next_check": {"type": "string", "description": "One check that would confirm it, in plain words."},
                "confidence": {"type": "number", "description": "0 to 1."}},
                "required": ["part_no", "condition", "what_i_see", "action", "role", "faults", "confidence"]}}


def identify(photo: Image.Image, exclude: Path | None = None) -> tuple[int | None, dict]:
    """Step 1 when the part is not given: which numbered part is this?"""
    tool = {"type": "function", "name": "identify_part", "description": "Which part is in the photo.",
            "parameters": {"type": "object", "properties": {
                "part_no": {"type": "integer", "enum": list(parts.PARTS)},
                "confidence": {"type": "number"}}, "required": ["part_no"]}}
    names = "; ".join(f"#{n} {p[0]}" for n, p in parts.PARTS.items())
    resp = client().responses.create(model=config.AGENT_MODEL, tools=[tool], input=[{"role": "user", "content": [
        text_part("Which part of a split air conditioner is in this photo? It may be clean or damaged. "
                  f"Parts: {names}. The second image is a numbered sheet of the parts when new. Call identify_part once."),
        {"type": "input_image", "image_url": image_to_data_url(photo, 1024)},
        {"type": "input_image", "image_url": image_to_data_url(parts_sheet(exclude), 1280)}]}])
    usage = {"input_tokens": getattr(resp.usage, "input_tokens", 0), "output_tokens": getattr(resp.usage, "output_tokens", 0)}
    for item in resp.output:
        if getattr(item, "type", "") == "function_call" and item.name == "identify_part":
            n = json.loads(item.arguments or "{}").get("part_no")
            return (n if n in parts.PARTS else None), usage
    return None, usage


def _content(photo: Image.Image, part_no: int | None, complaint: str, exclude: Path | None) -> tuple[list, list[str]]:
    sent = []
    content = [text_part("TECHNICIAN'S PHOTO:"), {"type": "input_image", "image_url": image_to_data_url(photo, 1024)}]
    if part_no is None:
        content += [text_part("The technician did not say which part this is. Identify it from this sheet of healthy "
                              "parts (numbered), then compare."),
                    {"type": "input_image", "image_url": image_to_data_url(parts_sheet(exclude), 1280)}]
        sent.append("numbered sheet of healthy parts")
        candidates = [n for n in parts.PARTS if healthy_reference(n, exclude)]
    else:
        content.append(text_part(f"The technician says this is part #{part_no}: {parts.PARTS[part_no][0]}."))
        candidates = [part_no]
        h = healthy_reference(part_no, exclude)
        if h:
            content += [text_part(f"REFERENCE, healthy #{part_no} {parts.PARTS[part_no][0]}:"),
                        {"type": "input_image", "image_url": image_to_data_url(Image.open(h), 640)}]
            sent.append(f"healthy {parts.PARTS[part_no][0].lower()}")
    refs = [r for n in candidates for r in damaged_references(n, exclude)]
    if part_no is None:
        refs = refs[:8]
    for r in refs:
        content += [text_part(f"REFERENCE, damaged: {r['caption']}. Seen with: {r['fault_title']} "
                              f"({'cause' if r['role'] == 'cause' else r['role']})."),
                    {"type": "input_image", "image_url": image_to_data_url(Image.open(r["path"]), 640)}]
        sent.append(r["caption"])
    content.append(text_part(f"Customer complaint: {complaint or 'not given'}.\n"
                             f"Fault ids: {json.dumps(_fault_titles())}"))
    return content, sent


def _outline(photo: Image.Image, part_no: int) -> tuple[Image.Image | None, str]:
    """SAM 3.1 outlines the part in the technician's photo."""
    from . import sam
    phrase = parts.PARTS[part_no][3]
    try:
        dets = sam.segment_image(photo, phrase)
    except Exception as e:
        return None, f"SAM 3.1 unavailable ({e})"
    if not dets:
        return None, f"SAM 3.1 did not find '{phrase}' in the photo"
    d = max(dets, key=lambda x: (x["x2"] - x["x1"]) * (x["y2"] - x["y1"]))
    out = photo.convert("RGB").copy()
    sx, sy = out.width / (d.get("w") or out.width), out.height / (d.get("h") or out.height)
    ImageDraw.Draw(out).rectangle((d["x1"] * sx, d["y1"] * sy, d["x2"] * sx, d["y2"] * sy),
                                  outline="#C8372D", width=max(4, out.width // 150))
    return out, f"Outlined by SAM 3.1 ('{phrase}')"


def _mock(known: dict | None) -> dict:
    if not known:
        return {"part_no": 2, "condition": "unclear", "what_i_see": "(offline mode) No model call was made.",
                "closest_reference": "none", "action": "check_further", "role": "not_related", "faults": [],
                "matches_complaint": "no complaint given", "next_check": "Run with the Meta API key set.", "confidence": 0.0}
    damaged = known["condition"] == "damaged"
    return {"part_no": known["part_no"], "condition": known["condition"],
            "what_i_see": f"(offline mode: the known answer) {known['caption']}.",
            "closest_reference": known["caption"] if damaged else "healthy part",
            "action": "replace" if damaged and known["role"] == "cause" else ("check_further" if damaged else "no_action"),
            "role": known["role"], "faults": [known["fault"]] if known.get("fault") else [],
            "matches_complaint": "no complaint given", "next_check": "", "confidence": 1.0}


def check(photo: Image.Image, part_no: int | None = None, complaint: str = "",
          library_path: Path | None = None, outline: bool = True) -> dict:
    t0 = time.time()
    known = known_answer(library_path)
    usage = {}
    if config.MOCK:
        result, sent, model = _mock(known), ["(offline mode)"], "offline mode"
    else:
        identified_by = "technician" if part_no else None
        if part_no is None:
            part_no, id_usage = identify(photo, library_path)
            identified_by = "FaultLens (step 1)" if part_no else None
        content, sent = _content(photo, part_no, complaint, library_path)
        profile = knowledge.load()
        kwargs = dict(model=config.AGENT_MODEL, tools=[_tool()],
                      input=[{"role": "system", "content": SYSTEM.format(equipment=profile["equipment"])},
                             {"role": "user", "content": content}])
        try:
            resp = client().responses.create(reasoning={"effort": "medium"}, **kwargs)
        except TypeError:
            resp = client().responses.create(**kwargs)
        usage = {"input_tokens": getattr(resp.usage, "input_tokens", 0), "output_tokens": getattr(resp.usage, "output_tokens", 0)}
        if identified_by and identified_by.startswith("FaultLens"):
            usage = {k: usage[k] + id_usage.get(k, 0) for k in usage}
        result = None
        for item in resp.output:
            if getattr(item, "type", "") == "function_call" and item.name == "assess_part":
                result = json.loads(item.arguments or "{}")
        if result is None:
            raise RuntimeError("The model did not return an assessment: " + (getattr(resp, "output_text", "") or "")[:200])
        model = f"Muse Spark ({config.AGENT_MODEL})"
    n = result.get("part_no") if result.get("part_no") in parts.PARTS else (part_no or 2)
    titles = _fault_titles()
    out = {**result, "part_no": n, "part_name": parts.PARTS[n][0],
           "action_text": FIX.get(result.get("action"), result.get("action", "")),
           "role_text": ROLE_TEXT.get(result.get("role"), ""),
           "fault_titles": [titles.get(f, f) for f in result.get("faults", [])],
           "references_sent": sent, "model": model, "usage": usage, "known": known,
           "identified_by": "offline mode" if config.MOCK else identified_by}
    if known:
        out["known"]["fault_title"] = titles.get(known["fault"], "") if known.get("fault") else ""
        out["agrees_with_known"] = (known["part_no"] == n and known["condition"] == result.get("condition"))
    if outline and not config.MOCK:
        img, note = _outline(photo, n)
        out["outlined"], out["outline_note"] = img, note
    else:
        out["outlined"], out["outline_note"] = None, "" if not config.MOCK else "SAM 3.1 skipped in offline mode"
    out["seconds"] = round(time.time() - t0, 1)
    return out

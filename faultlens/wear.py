"""Preventive maintenance: how far along a part's wear is, and how long it typically has left.

For parts that wear gradually and visibly (filter dust, coil dirt, fan-hub rust, pipe corrosion)
the library holds reference photos of each stage (knowledge/<profile>_wear.yaml). Muse Spark places
the technician's photo on that scale; the remaining life comes from the table, as a typical range,
and is labelled as such. Capacitors and motor windings fail suddenly, so they have no scale.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from . import config, library
from .knowledge import KNOWLEDGE_DIR

NOTE = ("Remaining life is a typical range set by FaultLens for a split AC in normal use, not measured on "
        "this unit. Dusty sites, coastal air and heavy use shorten it.")


def load() -> dict:
    path = KNOWLEDGE_DIR / f"{config.EQUIPMENT_PROFILE}_wear.yaml"
    if not path.exists():
        return {"parts": {}, "actions": {}}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def scale(part_no: int) -> dict | None:
    return load()["parts"].get(part_no)


def has_scale(part_no: int) -> bool:
    return scale(part_no) is not None


def _paths(stage: dict) -> list[Path]:
    out = [(library.LIBRARY / stage["image"]).resolve()]
    if stage.get("same_as"):
        out.append((library.LIBRARY / stage["same_as"]).resolve())
    return out


def stages(part_no: int, exclude: Path | None = None) -> list[dict]:
    """The stages of this part's scale, with a flag on any stage that IS the photo being checked
    (that stage's picture is then left out of the references)."""
    sc = scale(part_no)
    if not sc:
        return []
    acts = load().get("actions", {})
    ex = exclude.resolve() if exclude else None
    out = []
    for s in sc["stages"]:
        out.append({**s, "path": library.LIBRARY / s["image"], "url": f"/library/{s['image']}",
                    "action_text": acts.get(s["action"], s["action"]),
                    "is_query": ex is not None and ex in _paths(s)})
    return out


def known_stage(path: Path | None) -> tuple[int, int] | None:
    """(part_no, stage) when a library photo is one of the stage references."""
    if path is None:
        return None
    p = path.resolve()
    for n, sc in load()["parts"].items():
        for s in sc["stages"]:
            if p in _paths(s):
                return n, s["stage"]
    return None


def result(part_no: int, stage: int | None, exclude: Path | None = None) -> dict | None:
    sts = stages(part_no, exclude)
    if not sts or not stage:
        return None
    stage = max(1, min(len(sts), int(stage)))
    s = sts[stage - 1]
    return {"part_no": part_no, "part_name": scale(part_no)["name"], "wear": scale(part_no)["wear"],
            "stage": stage, "of": len(sts), "label": s["label"], "life": s["life"], "action": s["action"],
            "action_text": s["action_text"], "note": NOTE,
            "strip": [{"stage": x["stage"], "label": x["label"], "url": x["url"], "is_query": x["is_query"]} for x in sts]}


def library_examples() -> list[dict]:
    """Stage photos to try in the app (the ones not already elsewhere in the library)."""
    out = []
    for n, sc in load()["parts"].items():
        for s in sc["stages"]:
            if not s.get("same_as"):
                out.append({"url": f"/library/{s['image']}", "part": n,
                            "caption": f"{sc['name']}, stage {s['stage']}: {s['label']}"})
    return out

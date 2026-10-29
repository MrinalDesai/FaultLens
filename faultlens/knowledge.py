"""Load the equipment profile (faults, checks, normal ranges, safety rules)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from . import config

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"


@lru_cache(maxsize=4)
def load(profile: str | None = None) -> dict:
    name = profile or config.EQUIPMENT_PROFILE
    with open(KNOWLEDGE_DIR / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fault_ids() -> list[str]:
    return list(load()["faults"].keys())


def fault_menu() -> list[dict]:
    return [{"fault_id": k, "title": v["title"], "summary": v["summary"], "sources": sources_for(k)}
            for k, v in load()["faults"].items()]


def sources(profile: str | None = None) -> dict:
    """Public documentation behind each fault (knowledge/<profile>_sources.yaml), if present."""
    name = profile or config.EQUIPMENT_PROFILE
    path = KNOWLEDGE_DIR / f"{name}_sources.yaml"
    if not path.exists():
        return {"sources": {}, "faults": {}, "not_covered": {}}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def sources_for(fault_id: str) -> list[dict]:
    src = sources()
    out = []
    for ref in src.get("faults", {}).get(fault_id, []):
        s = src["sources"].get(ref["source"])
        if s:
            out.append({"title": s["title"], "url": s["url"], "supports": ref.get("supports", "")})
    return out


def fault_guide(fault_id: str) -> dict | None:
    guide = load()["faults"].get(fault_id)
    if guide is None:
        return None
    nc = sources().get("not_covered", {}).get(fault_id)
    return {**guide, "public_sources": sources_for(fault_id), **({"not_covered_by_sources": nc} if nc else {})}


def find_check(check_id: str) -> tuple[str, dict] | None:
    for fid, fault in load()["faults"].items():
        for check in fault.get("checks", []):
            if check["id"] == check_id:
                return fid, check
    return None

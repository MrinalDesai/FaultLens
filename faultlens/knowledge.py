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
    return [{"fault_id": k, "title": v["title"], "summary": v["summary"]} for k, v in load()["faults"].items()]


def fault_guide(fault_id: str) -> dict | None:
    return load()["faults"].get(fault_id)


def find_check(check_id: str) -> tuple[str, dict] | None:
    for fid, fault in load()["faults"].items():
        for check in fault.get("checks", []):
            if check["id"] == check_id:
                return fid, check
    return None

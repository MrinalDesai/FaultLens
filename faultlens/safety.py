"""Fixed safety rules. These run after the model and the model cannot override them."""

from __future__ import annotations

from . import knowledge


def gate_check(check_id: str | None) -> dict:
    """Return the check plus any mandatory safety steps that must come before it."""
    if not check_id:
        return {"check": None, "warnings": []}
    found = knowledge.find_check(check_id)
    if not found:
        return {"check": None, "warnings": [f"Unknown check '{check_id}' - ignored."]}
    fault_id, check = found
    rules = knowledge.load()["safety_rules"]
    warnings = []
    if check.get("needs_power_isolation"):
        warnings.append("POWER ISOLATION REQUIRED. " + rules["power_isolation"])
    if check.get("certified_only"):
        warnings.append("CERTIFIED TECHNICIAN ONLY. " + rules["certified_only"])
    return {"check": {**check, "fault_id": fault_id}, "warnings": warnings}


def gate_assessment(assessment: dict) -> dict:
    """Attach safety warnings to the recommended next check. Also blocks the model from
    recommending a check that does not exist in the equipment profile."""
    gated = gate_check(assessment.get("next_check_id"))
    return {**assessment, "next_check": gated["check"], "safety_warnings": gated["warnings"]}

"""One full investigation round: parts -> agent -> safety rules -> checker."""

from __future__ import annotations

import time

from . import config, pattern, safety, sam
from .agent import Evidence, investigate
from .checker import check
from .knowledge import load


def locate_parts(ev: Evidence, phrases: list[str] | None = None) -> None:
    """Run SAM 3.1 on each keyframe and store the boxes on the evidence."""
    phrases = phrases or load()["parts_to_find"]
    ev.detections = []
    for i, frame in enumerate(ev.frames):
        for det in sam.find_parts(frame, phrases):
            det["frame_index"] = i
            ev.detections.append(det)


def match_patterns(ev: Evidence) -> str | None:
    """Run the fusion model on whatever signals are present. Returns a note if it could not run."""
    feats = ev.signal_features()
    if not feats:
        return None
    model = pattern.get_model()
    if model is None:
        return "Pattern model not trained yet. Run: python -m faultlens.pattern"
    ev.pattern_match = model.match(feats)
    return None


def run_round(ev: Evidence, use_checker: bool = True, timings: dict | None = None) -> dict:
    timings = timings if timings is not None else {}
    started = time.time()

    t = time.time()
    pattern_note = match_patterns(ev)
    if ev.pattern_match is not None:
        timings["pattern"] = round(time.time() - t, 2)

    t = time.time()
    result = investigate(ev)
    timings["agent"] = round(time.time() - t, 2)

    t = time.time()
    assessment = safety.gate_assessment(result["assessment"])
    timings["safety"] = round(time.time() - t, 2)

    t = time.time()
    verdict = check(ev, assessment) if use_checker else None
    if use_checker:
        timings["checker"] = round(time.time() - t, 2)
    if verdict and verdict["verdict"] == "reject":
        # The checker overrules confidence: ask for more evidence instead.
        assessment["status"] = "need_more_evidence"
        if verdict.get("missing_evidence"):
            assessment["observation_request"] = verdict["missing_evidence"]

    return {
        "assessment": assessment,
        "checker": verdict,
        "trace": result["trace"],
        "usage": result["usage"],
        "seconds": round(time.time() - started, 1),
        "timings": timings,
        "pattern_match": ev.pattern_match,
        "pattern_note": pattern_note,
        "mock": config.MOCK,
    }

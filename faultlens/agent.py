"""The investigation agent: Muse Spark with tools, ending in a structured assessment.

Flow for one round:
  1. The agent gets the evidence: sensor summary, technician notes, photos (with parts
     found by SAM 3.1), thermal images and a sound spectrogram it can look at, and the
     pattern-match result from the fusion model.
  2. It may call tools to read fault guides or re-read evidence.
  3. It must finish by calling submit_assessment: ranked causes, the evidence
     for and against each, and either a confident answer or a request for one
     more specific observation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from . import alarms, config, features, knowledge, render, sensors
from .meta_client import client, image_part, text_part

SYSTEM_PROMPT = """You are FaultLens, a diagnostic assistant for technicians repairing {equipment}.

Your job is to work out the most likely cause of the failure from the evidence, and say what to check next.

Rules:
- Only use fault_id values from list_faults. Read a fault's guide before ranking it highly.
- Every hypothesis needs evidence_for and evidence_against, quoting the actual sensor numbers,
  technician notes or what is visible in a numbered frame. Never invent evidence.
- If two causes still fit the evidence, or a key reading is missing, set status to
  "need_more_evidence" and ask for the single observation that would separate them.
  Asking for more evidence is better than guessing.
- Set status to "confident" only when the top cause is clearly ahead and nothing important contradicts it.
- next_check_id must be a check id from a fault guide.
- The pattern match comes from a model trained on SIMULATED reference signatures. Treat it as one
  piece of evidence, not the answer. Say which signals agree and which disagree. When signals
  disagree, explain the likely reason and ask for the observation that settles it.
- You can see the thermal images and the sound spectrogram yourself. Check that what you see is
  consistent with the pattern match, and mention it when it is not.
- The alarm history is the unit's own protection record: a code says what the unit protected itself
  from (too hot outside, coil freezing), not which part failed, and an empty log does not rule a fault out.
- If everything points to a healthy unit, rank "normal" first.
- Finish by calling submit_assessment exactly once.
"""


@dataclass
class Evidence:
    sensor_rows: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    frames: list[Image.Image] = field(default_factory=list)          # ordinary photos / video keyframes
    detections: list[dict] = field(default_factory=list)
    sound: np.ndarray | None = None                                  # 16 kHz mono, outdoor unit
    thermal_indoor: np.ndarray | None = None                         # temperatures in C
    thermal_outdoor: np.ndarray | None = None
    pattern_match: dict | None = None
    alarms: list[dict] = field(default_factory=list)                 # the unit's fault log
    alarms_given: bool = False                                       # True when a log was read, even if empty

    def signal_features(self) -> dict[str, dict]:
        out = {}
        if self.sound is not None:
            out["sound"] = features.sound_features(self.sound)
        if self.thermal_indoor is not None:
            out["thermal_indoor"] = features.thermal_indoor_features(self.thermal_indoor)
        if self.thermal_outdoor is not None:
            out["thermal_outdoor"] = features.thermal_outdoor_features(self.thermal_outdoor)
        if self.sensor_rows:
            out["sensors"] = features.sensor_features(self.sensor_rows)
        return out

    def pattern_summary(self) -> dict | None:
        """Compact version of the pattern match for the model and the checker."""
        pm = self.pattern_match
        if not pm:
            return None
        return {
            "note": "From a fusion model trained on simulated reference signatures. One piece of evidence.",
            "combined_ranking": pm["fused"][:4],
            "each_signal": {m: {"best_match": v["top"], "confidence": v["confidence"],
                                "agrees_with_combined": v["agrees_with_fused"],
                                "key_measurements": v["key_measurements"]}
                            for m, v in pm["signals"].items()},
            "signals_not_provided": pm["signals_missing"],
        }

    def sensor_summary(self) -> dict | None:
        return sensors.summarize(self.sensor_rows) if self.sensor_rows else None

    def parts_summary(self) -> list[dict]:
        out = []
        for d in self.detections:
            out.append({"part": d["label"], "frame": d.get("frame_index", d.get("frame", 0)),
                        "confidence": d.get("conf")})
        return out


def _tools() -> list[dict]:
    ids = knowledge.fault_ids()
    return [
        {"type": "function", "name": "list_faults",
         "description": "List the faults FaultLens knows for this equipment.",
         "parameters": {"type": "object", "properties": {}, "additionalProperties": False}},
        {"type": "function", "name": "lookup_fault_guide",
         "description": "Get the sensor signature, visual signs, checks and fix for one fault.",
         "parameters": {"type": "object",
                        "properties": {"fault_id": {"type": "string", "enum": ids}},
                        "required": ["fault_id"], "additionalProperties": False}},
        {"type": "function", "name": "get_sensor_summary",
         "description": "Sensor statistics, derived values and out-of-range flags for this unit.",
         "parameters": {"type": "object", "properties": {}, "additionalProperties": False}},
        {"type": "function", "name": "get_pattern_match",
         "description": "Result of the sound/thermal/sensor pattern-matching model: combined ranking, "
                        "each signal's best match, and the measurements behind it.",
         "parameters": {"type": "object", "properties": {}, "additionalProperties": False}},
        {"type": "function", "name": "get_detected_parts",
         "description": "Parts that SAM 3.1 located in the video frames, with frame numbers.",
         "parameters": {"type": "object", "properties": {}, "additionalProperties": False}},
        {"type": "function", "name": "submit_assessment",
         "description": "Submit the final assessment for this round. Call exactly once, last.",
         "parameters": {
             "type": "object",
             "properties": {
                 "status": {"type": "string", "enum": ["confident", "need_more_evidence"]},
                 "hypotheses": {
                     "type": "array",
                     "items": {
                         "type": "object",
                         "properties": {
                             "fault_id": {"type": "string", "enum": ids},
                             "probability": {"type": "number"},
                             "evidence_for": {"type": "array", "items": {"type": "string"}},
                             "evidence_against": {"type": "array", "items": {"type": "string"}},
                         },
                         "required": ["fault_id", "probability", "evidence_for", "evidence_against"],
                     },
                 },
                 "next_check_id": {"type": "string"},
                 "observation_request": {"type": "string",
                                         "description": "The single observation needed, if any."},
                 "summary": {"type": "string", "description": "Two sentences for the technician."},
             },
             "required": ["status", "hypotheses", "next_check_id", "summary"],
         }},
    ]


def _run_tool(name: str, args: dict, ev: Evidence):
    if name == "list_faults":
        return knowledge.fault_menu()
    if name == "lookup_fault_guide":
        return knowledge.fault_guide(args.get("fault_id", "")) or {"error": "unknown fault_id"}
    if name == "get_sensor_summary":
        return ev.sensor_summary() or {"note": "No sensor log was provided."}
    if name == "get_pattern_match":
        return ev.pattern_summary() or {"note": "No sound, thermal image or sensor log was provided."}
    if name == "get_detected_parts":
        return ev.parts_summary() or {"note": "No parts were detected or no video was provided."}
    return {"error": f"unknown tool {name}"}


def _opening_message(ev: Evidence) -> dict:
    parts = [text_part("New investigation. Evidence follows.")]
    summary = ev.sensor_summary()
    parts.append(text_part("Sensor summary:\n" + (json.dumps(summary, indent=1) if summary else "none provided")))
    if ev.alarms_given:
        parts.append(text_part("Alarm history (the unit's own fault log, last 7 days):\n" + json.dumps(alarms.summarize(ev.alarms), indent=1)))
    parts.append(text_part("Technician notes:\n" + ("\n".join(f"- {n}" for n in ev.notes) or "none")))
    pm = ev.pattern_summary()
    if pm:
        parts.append(text_part("Pattern match:\n" + json.dumps(pm, indent=1)))
    if ev.detections:
        parts.append(text_part("Parts located by SAM 3.1:\n" + json.dumps(ev.parts_summary())))
    for i, frame in enumerate(ev.frames):
        parts.append(text_part(f"Photo {i}:"))
        parts.append(image_part(frame))
    if ev.thermal_indoor is not None:
        parts.append(text_part("Thermal image, indoor coil (false colour, scale on the right, hottest and "
                               "coldest points marked):"))
        parts.append(image_part(render.thermal_view(ev.thermal_indoor)))
    if ev.thermal_outdoor is not None:
        parts.append(text_part("Thermal image, outdoor unit: coil on the left, compressor bay on the right "
                               "(capacitor small block at the top of the bay, compressor the round body):"))
        parts.append(image_part(render.thermal_view(ev.thermal_outdoor)))
    if ev.sound is not None:
        parts.append(text_part("Spectrogram of the outdoor-unit sound, 10 s left to right, pitch 0-4 kHz bottom "
                               "to top, brighter = louder:"))
        parts.append(image_part(render.spectrogram(ev.sound)))
    return {"role": "user", "content": parts}


def investigate(ev: Evidence) -> dict:
    """Run one investigation round. Returns {assessment, trace, usage}."""
    if config.MOCK:
        return _mock_investigate(ev)

    profile = knowledge.load()
    history: list = [
        {"role": "system", "content": SYSTEM_PROMPT.format(equipment=profile["equipment"])},
        _opening_message(ev),
    ]
    trace, usage = [], {"input_tokens": 0, "output_tokens": 0}
    tools = _tools()

    for _ in range(config.MAX_AGENT_TURNS):
        resp = client().responses.create(model=config.AGENT_MODEL, input=history, tools=tools)
        if resp.usage:
            usage["input_tokens"] += resp.usage.input_tokens
            usage["output_tokens"] += resp.usage.output_tokens

        calls = [item for item in resp.output if getattr(item, "type", "") == "function_call"]
        if not calls:
            text = getattr(resp, "output_text", "") or ""
            trace.append({"kind": "message", "text": text[:500]})
            history.append({"role": "assistant", "content": text or "(no text)"})
            history.append({"role": "user", "content": "Finish now by calling submit_assessment."})
            continue

        for call in calls:
            args = json.loads(call.arguments or "{}")
            history.append({"type": "function_call", "call_id": call.call_id,
                            "name": call.name, "arguments": call.arguments or "{}"})
            if call.name == "submit_assessment":
                trace.append({"kind": "tool", "name": call.name, "args": args})
                return {"assessment": normalise(args), "trace": trace, "usage": usage}
            result = _run_tool(call.name, args, ev)
            trace.append({"kind": "tool", "name": call.name, "args": args,
                          "result_preview": json.dumps(result)[:300]})
            history.append({"type": "function_call_output", "call_id": call.call_id,
                            "output": json.dumps(result)})

    return {
        "assessment": {"status": "need_more_evidence", "hypotheses": [], "next_check_id": "",
                       "observation_request": "The agent did not finish. Add a clearer photo or note and retry.",
                       "summary": "No assessment produced within the turn limit."},
        "trace": trace, "usage": usage,
    }


def normalise(a: dict) -> dict:
    """Clean up the model's assessment: valid fault ids only, probabilities summing to 1, sorted."""
    valid = set(knowledge.fault_ids())
    hyps = [h for h in a.get("hypotheses", []) if h.get("fault_id") in valid]
    total = sum(max(float(h.get("probability", 0)), 0) for h in hyps) or 1.0
    for h in hyps:
        h["probability"] = round(max(float(h.get("probability", 0)), 0) / total, 3)
    hyps.sort(key=lambda h: h["probability"], reverse=True)
    return {**a, "hypotheses": hyps}


def _mock_investigate(ev: Evidence) -> dict:
    summary = ev.sensor_summary()
    ids = knowledge.fault_ids()
    if ev.pattern_match:
        scores = {r["condition"]: r["probability"] for r in ev.pattern_match["fused"] if r["condition"] in ids}
    elif summary:
        scores = sensors.rule_scores(summary)
    else:
        scores = {f: 1 / len(ids) for f in ids}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
    top_id, top_p = ranked[0]
    guide = knowledge.fault_guide(top_id)
    hyps = [{"fault_id": f, "probability": p,
             "evidence_for": [f"(offline mode) pattern score {p}"], "evidence_against": []} for f, p in ranked]
    status = "confident" if top_p >= 0.6 else "need_more_evidence"
    return {
        "assessment": normalise({
            "status": status, "hypotheses": hyps, "next_check_id": guide["checks"][0]["id"],
            "observation_request": "" if status == "confident" else guide["checks"][0]["do"],
            "summary": f"(mock mode) Most likely: {guide['title']}.",
        }),
        "trace": [{"kind": "tool", "name": "mock", "args": {}}],
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }

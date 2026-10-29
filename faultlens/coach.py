"""The master diagnostician: Muse Spark coaching a technician through a live job, one step at a time.

Each turn it sees the complaint, every finding so far, the images revealed so far (thermal views,
spectrogram, photos), the pattern-model result and the list of possible actions. It replies through
one structured tool call: a short plain-language message, a ranked list of causes, the single next
action, and whether it is ready to conclude.

If the model is unavailable (offline mode or an API error), a rule-based coach takes over so the
game keeps working; the reply says which one answered.
"""

from __future__ import annotations

import json

from . import config, features, game, knowledge, pattern, render
from .meta_client import client, image_part, text_part

SYSTEM = """You are FaultLens, a master diagnostician coaching a technician on a live service job on a {equipment}.
You do not know the fault. You find it with the technician, one step at a time.

How to coach:
- Recommend exactly ONE next action from the action list, the one that best separates the remaining
  causes for the least effort. Never repeat an action already done.
- Explain why in one or two plain sentences a technician would use on site. No jargon.
- Safety first: actions marked power "off" need the power isolated, so recommend "power_off" first and
  say in the message which check you will do once it is off.
  Actions marked "on" need the unit running. Certified-only actions need a certified refrigerant technician.
- Use only the evidence so far. The pattern match comes from a model trained on simulated signatures:
  treat it as one piece of evidence and say when the signals disagree.
- Look at the images yourself (thermal images, spectrogram, photos) and use what you see.
- The previous visit's job card (read_job_card) is history, not proof: a past fix can point to a cause,
  but confirm it on the unit today.
- Set ready_to_conclude only when one cause is clearly ahead AND at least one direct check confirms it.
  Then give final_fault_id and the fix in plain words.
- If the technician asked a question, answer it first in the message, briefly.
Call the advise tool exactly once."""


def _signal_features(job: game.Job) -> dict:
    out = {}
    s = job.signals
    if "sound" in s:
        out["sound"] = features.sound_features(s["sound"])
    if "thermal_indoor" in s:
        out["thermal_indoor"] = features.thermal_indoor_features(s["thermal_indoor"])
    if "thermal_outdoor" in s:
        out["thermal_outdoor"] = features.thermal_outdoor_features(s["thermal_outdoor"])
    if "sensors" in s:
        out["sensors"] = features.sensor_features(s["sensors"])
    return out


def pattern_for(job: game.Job) -> dict | None:
    feats = _signal_features(job)
    model = pattern.get_model()
    if not feats or model is None:
        return None
    return model.match(feats)


def _pattern_summary(pm: dict | None) -> dict | None:
    if not pm:
        return None
    return {
        "note": "From a fusion model trained on simulated reference signatures. One piece of evidence.",
        "combined_ranking": pm["fused"][:4],
        "each_signal": {m: {"best_match": v["top"], "confidence": v["confidence"],
                            "agrees_with_combined": v["agrees_with_fused"],
                            "key_measurements": v["key_measurements"]} for m, v in pm["signals"].items()},
        "signals_not_collected": pm["signals_missing"],
    }


def _tool(job: game.Job) -> dict:
    ids = knowledge.fault_ids()
    actions = [a for a in game.ACTIONS if a not in job.done_ids()] + ["power_off", "power_on", "conclude"]
    return {"type": "function", "name": "advise",
            "description": "Your reply to the technician for this turn.",
            "parameters": {"type": "object", "properties": {
                "message": {"type": "string", "description": "What you say to the technician, 1-3 short sentences."},
                "hypotheses": {"type": "array", "items": {"type": "object", "properties": {
                    "fault_id": {"type": "string", "enum": ids},
                    "probability": {"type": "number"},
                    "why": {"type": "string"}}, "required": ["fault_id", "probability", "why"]}},
                "next_action_id": {"type": "string", "enum": actions},
                "ready_to_conclude": {"type": "boolean"},
                "final_fault_id": {"type": "string", "enum": ids},
                "fix": {"type": "string"}},
                "required": ["message", "hypotheses", "next_action_id", "ready_to_conclude"]}}


def _content(job: game.Job, pm: dict | None, question: str | None) -> list:
    menu = [{"id": a["id"], "title": a["title"], "power": a["power"], "certified_only": a["certified"],
             "done": a["done"]} for a in game.action_menu(job)]
    state = {
        "customer_complaint": job.complaint,
        "power": "on (unit running)" if job.power_on else "off (isolated)",
        "steps_so_far": [{"action": s["action"], "finding": s["finding"]} for s in job.steps],
        "pattern_match": _pattern_summary(pm),
        "actions": menu,
    }
    parts = [text_part("Job state:\n" + json.dumps(state, indent=1))]
    s = job.signals
    if "thermal_indoor" in s:
        parts += [text_part("Thermal image, indoor coil (scale on the right, hottest and coldest marked):"),
                  image_part(render.thermal_view(s["thermal_indoor"]))]
    if "thermal_outdoor" in s:
        parts += [text_part("Thermal image, outdoor unit (coil left; compressor round body and capacitor small "
                            "block on the right):"), image_part(render.thermal_view(s["thermal_outdoor"]))]
    if "sound" in s:
        parts += [text_part("Spectrogram of the outdoor-unit sound, 10 s left to right, pitch bottom to top:"),
                  image_part(render.spectrogram(s["sound"]))]
    if question:
        parts.append(text_part("The technician asks: " + question))
    else:
        parts.append(text_part("Advise the next step."))
    return parts


def _call_model(job: game.Job, pm: dict | None, question: str | None) -> tuple[dict, dict]:
    profile = knowledge.load()
    history = [{"role": "system", "content": SYSTEM.format(equipment=profile["equipment"])},
               {"role": "user", "content": _content(job, pm, question)}]
    kwargs = dict(model=config.AGENT_MODEL, input=history, tools=[_tool(job)])
    try:
        resp = client().responses.create(reasoning={"effort": config.COACH_REASONING}, **kwargs)
    except TypeError:
        resp = client().responses.create(**kwargs)
    usage = {"input_tokens": getattr(resp.usage, "input_tokens", 0), "output_tokens": getattr(resp.usage, "output_tokens", 0)}
    for item in resp.output:
        if getattr(item, "type", "") == "function_call" and item.name == "advise":
            return json.loads(item.arguments or "{}"), usage
    # No tool call: keep the text as the message, let the rule coach pick the step.
    fallback = _rule_coach(job, pm, question)
    fallback["message"] = (getattr(resp, "output_text", "") or fallback["message"]).strip()
    return fallback, usage


# ----------------------------------------------------------------------------- rule-based coach

CONFIRM = {
    "normal": "feel_supply_air",
    "clogged_air_filter": "inspect_filter",
    "condenser_fan_failure": "watch_outdoor_fan",
    "low_refrigerant": "inspect_joints",
    "dirty_condenser_coil": "inspect_condenser_fins",
    "failed_compressor_capacitor": "listen_compressor",
    "faulty_room_thermistor": "compare_thermometer",
}
SURVEY = ["read_job_card", "listen_outdoor", "thermal_outdoor", "read_sensors", "thermal_indoor"]


def _rule_coach(job: game.Job, pm: dict | None, question: str | None) -> dict:
    done = set(job.done_ids())
    ranking = [(r["condition"], r["probability"]) for r in pm["fused"]] if pm else []
    top, p_top = ranking[0] if ranking else (None, 0.0)
    nxt = None
    if top and p_top >= 0.6 and CONFIRM[top] not in done:
        nxt = CONFIRM[top]
    if nxt is None:
        nxt = next((a for a in SURVEY if a not in done), None)
    if nxt is None and top:
        nxt = CONFIRM[top] if CONFIRM[top] not in done else "conclude"
    nxt = nxt or "conclude"
    power = game.ACTIONS.get(nxt, (None, None, None, game.ANY))[3]
    if power == game.POWER_OFF and job.power_on:
        nxt = "power_off"
    elif power == game.POWER_ON and not job.power_on:
        nxt = "power_on"
    ready = bool(top) and p_top >= 0.8 and CONFIRM[top] in done
    titles = {f["fault_id"]: f["title"] for f in knowledge.fault_menu()}
    step_title = game.ACTIONS[nxt][1] if nxt in game.ACTIONS else {"power_off": "Isolate the power",
                                                                     "power_on": "Restore power", "conclude": "Conclude"}[nxt]
    msg = (f"Leading candidate: {titles.get(top, 'not clear yet')} ({p_top:.0%}). Next: {step_title.lower()}."
           if top else f"Start with a quick survey. Next: {step_title.lower()}.")
    if ready:
        msg = f"The evidence and the direct check agree: {titles[top]}."
    return {"message": msg, "next_action_id": "conclude" if ready else nxt, "ready_to_conclude": ready,
            "hypotheses": [{"fault_id": c, "probability": p, "why": "pattern match"} for c, p in ranking[:3]],
            "final_fault_id": top if ready else None,
            "fix": knowledge.fault_guide(top)["fix"] if ready and top else ""}


# ----------------------------------------------------------------------------- public

def advise(job: game.Job, question: str | None = None) -> dict:
    pm = pattern_for(job)
    usage = {"input_tokens": 0, "output_tokens": 0}
    source = "Muse Spark"
    error = None
    if config.MOCK:
        advice, source = _rule_coach(job, pm, question), "rule-based coach (offline mode)"
    else:
        try:
            advice, usage = _call_model(job, pm, question)
        except Exception as e:  # keep the job playable; show why
            advice, source, error = _rule_coach(job, pm, question), "rule-based coach (model unavailable)", str(e)

    # Clean up: valid ids only, probabilities normalised, never suggest a finished action.
    valid = set(knowledge.fault_ids())
    hyps = [h for h in advice.get("hypotheses", []) if h.get("fault_id") in valid]
    total = sum(max(float(h.get("probability", 0)), 0) for h in hyps) or 1
    for h in hyps:
        h["probability"] = round(max(float(h.get("probability", 0)), 0) / total, 3)
    hyps.sort(key=lambda h: -h["probability"])
    advice["hypotheses"] = hyps
    if advice.get("next_action_id") in job.done_ids():
        advice["next_action_id"] = _rule_coach(job, pm, None)["next_action_id"]
    nxt = advice.get("next_action_id")
    if nxt in game.ACTIONS:
        power = game.ACTIONS[nxt][3]
        if power == game.POWER_OFF and job.power_on:
            advice["safety_note"] = "That step needs the power isolated first."
        elif power == game.POWER_ON and not job.power_on:
            advice["safety_note"] = "That step needs the unit running. Restore power first."
    advice.update(source=source, usage=usage, pattern_match=pm, error=error)
    if not advice.get("final_fault_id") and hyps:
        advice["final_fault_id"] = hyps[0]["fault_id"]
    job.last_advice = advice
    job.chat.append({"from": "coach", "text": advice.get("message", ""), "question": question})
    return advice

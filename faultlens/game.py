"""The service-job simulator.

A job is a unit with a hidden condition. The technician performs actions one at a time
(listen, thermal camera, inspect, measure). Each action reveals what a real technician would
find on that unit. Signals (sound, thermal images, sensor log) are generated fresh for every
job from faultlens/simulate.py, so no two jobs are identical.

Safety is part of the game: running checks need the unit powered on; anything that opens
an electrical compartment or touches live parts needs the power isolated first. Blocked
attempts are counted and cost points.

The master diagnostician (faultlens/coach.py) suggests the next action; the technician
can follow it or choose their own.
"""

from __future__ import annotations

import json
import random
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from . import alarms, knowledge, library, render, sensors, simulate

POWER_ON, POWER_OFF, ANY = "on", "off", "any"

# ----------------------------------------------------------------------------- actions
# group, title, what the technician physically does, power needed, certified-only
ACTIONS = {
    "read_job_card":       ("History", "Read the last job card", "Photograph the previous visit's handwritten job card from the customer's file.", ANY, False),
    "read_alarm_log":      ("History", "Read the unit's fault history", "Press the diagnostic button on the remote and read the last 7 days of error codes from the display.", ANY, False),
    "listen_outdoor":      ("Listen", "Listen to the outdoor unit", "Stand 1 m from the outdoor unit and record 10 seconds of sound.", POWER_ON, False),
    "watch_outdoor_fan":   ("Listen", "Watch the outdoor fan", "Watch the outdoor fan blade through the grille for 30 seconds.", POWER_ON, False),
    "listen_compressor":   ("Listen", "Listen at compressor start", "Restart the unit from the remote and listen at the outdoor unit as the compressor starts.", POWER_ON, False),
    "feel_supply_air":     ("Listen", "Feel the air from the indoor unit", "Hold a hand and a thermometer at the indoor unit louvres on high fan.", POWER_ON, False),
    "thermal_indoor":      ("Thermal camera", "Thermal image of the indoor coil", "Lift the front panel and take a thermal image of the indoor coil.", POWER_ON, False),
    "thermal_outdoor":     ("Thermal camera", "Thermal image of the outdoor unit", "Take a thermal image of the outdoor unit with the side panel off.", POWER_ON, False),
    "read_sensors":        ("Instruments", "Log readings for 30 minutes", "Clip on the data logger: air temperatures, currents, pressures, one reading a minute.", POWER_ON, False),
    "compare_thermometer": ("Instruments", "Compare room reading with a thermometer", "Hold a reference thermometer at the air intake and compare with the room temperature the unit reports.", POWER_ON, False),
    "gauge_pressures":     ("Instruments", "Connect refrigerant gauges", "Connect gauges to the service port and read suction and discharge pressure.", POWER_ON, True),
    "inspect_filter":      ("Inspect", "Inspect the air filters", "Open the front panel, pull out the filters and hold them up to the light.", POWER_OFF, False),
    "inspect_coil":        ("Inspect", "Inspect the indoor coil", "Look at the indoor coil face with a torch.", ANY, False),
    "inspect_condenser_fins": ("Inspect", "Inspect the outdoor coil fins", "Shine a torch through the outdoor coil from the fan side.", POWER_OFF, False),
    "inspect_joints":      ("Inspect", "Inspect pipe joints and valves", "Check the flare nuts and service valves at both units for oily residue.", ANY, False),
    "inspect_capacitor":   ("Inspect", "Inspect the capacitors", "Remove the outdoor unit's electrical cover and look at the run and fan capacitors.", POWER_OFF, False),
    "inspect_fan_motor":   ("Inspect", "Open the outdoor fan motor", "Remove the fan motor's end cover and look at the windings with a torch.", POWER_OFF, False),
    "spin_fan_blade":      ("Inspect", "Spin the outdoor fan by hand", "With power isolated, spin the outdoor fan blade by hand.", POWER_OFF, False),
    "measure_capacitor":   ("Measure", "Measure the compressor capacitor", "Discharge the run capacitor and measure it with a capacitance meter.", POWER_OFF, False),
    "measure_fan_capacitor": ("Measure", "Measure the fan capacitor", "Discharge the fan capacitor and measure it.", POWER_OFF, False),
    "measure_thermistor":  ("Measure", "Measure the room sensor", "Unplug the indoor room-temperature sensor and measure its resistance.", POWER_OFF, False),
}

# The specific cause behind a fault, picked per job so findings and photos agree.
VARIANTS = {
    "condenser_fan_failure": {"bearing": "Seized, rusted fan bearing",
                              "motor": "Burnt fan motor winding",
                              "capacitor": "Failed fan capacitor"},
    "low_refrigerant": {"flare": "Leak at a flare nut", "pipe": "Pinhole in a corroded copper pipe",
                        "hidden": "Leak not visible (found with a leak detector)"},
    "failed_compressor_capacitor": {"domed": "Run capacitor scorched and leaking",
                                    "looks_normal": "Run capacitor failed inside, looks normal outside"},
    "faulty_room_thermistor": {"broken": "Room sensor cracked, wire broken",
                               "loose": "Room sensor fallen out of its clip onto the cold coil"},
}

# Signals revealed by an action, for the pattern model and the images.
SIGNAL_OF = {"listen_outdoor": "sound", "thermal_indoor": "thermal_indoor",
             "thermal_outdoor": "thermal_outdoor", "read_sensors": "sensors"}

COMPLAINTS = {
    "normal": ["It felt warm in the room yesterday evening. Seems okay now, but please check it.",
               "Please do the yearly check. It cools, but I'm not sure it's as cold as before."],
    "clogged_air_filter": ["Air from the AC is very weak, even on high fan. Water drips from the indoor unit sometimes.",
                           "The room takes ages to cool and the air coming out feels feeble."],
    "condenser_fan_failure": ["AC runs but the room doesn't cool. It switches itself off every few minutes.",
                              "Cooling stopped this afternoon. The outdoor box is very hot."],
    "low_refrigerant": ["Cooling has been getting worse every month. Now it barely cools at all.",
                        "It runs all day and the room never gets cold. It had a gas top-up last year."],
    "dirty_condenser_coil": ["Cools fine at night but struggles in the afternoon heat.",
                             "Bills have gone up and it cools poorly when it's hot outside."],
    "failed_compressor_capacitor": ["Indoor fan blows but the air isn't cold. Outside unit makes a loud hum and then clicks.",
                                    "No cooling since the power cut last week. The fan works."],
    "faulty_room_thermistor": ["The remote says the room is cold but it's actually hot. The outdoor unit doesn't start.",
                               "AC blows room-temperature air. Outside unit is silent."],
}


def _results(cond: str, s: float, rng: random.Random, sensor_summary: dict) -> dict[str, str]:
    """What each action finds on this unit. s = severity 0-1."""
    mild = s < 0.45
    lvl = library.severity_level(s)          # early / developing / advanced: findings match the photo shown

    def by(early, developing, advanced):
        return {"early": early, "developing": developing, "advanced": advanced}[lvl]
    st = sensor_summary["stats"]

    def med(ch):
        return st.get(ch, {}).get("median")

    # Defaults: what a healthy unit shows.
    r = {
        "watch_outdoor_fan": "The fan turns steadily at full speed. Warm air leaves the top of the unit.",
        "listen_compressor": "After the 3-minute delay the compressor starts smoothly with a short hum. The fan starts with it.",
        "feel_supply_air": f"Strong airflow. Air at the louvres measures {rng.uniform(12, 14.5):.1f} C.",
        "compare_thermometer": "The unit reads the room within 1 C of the reference thermometer.",
        "gauge_pressures": f"Suction {med('suction_psig'):.0f} psi, discharge {med('discharge_psig'):.0f} psi.",
        "inspect_filter": "Filters are clean. Light passes through the mesh easily.",
        "inspect_coil": "The indoor coil is clean and evenly cold. No frost.",
        "inspect_condenser_fins": "Light passes through the outdoor fins easily. Light dust only.",
        "inspect_joints": "Flare nuts and service valves are clean and dry.",
        "inspect_capacitor": "Both capacitors look normal: flat tops, no leaks, no scorch marks.",
        "spin_fan_blade": "The blade spins freely and coasts to a smooth stop.",
        "inspect_fan_motor": "The windings are clean copper, no discolouration or burnt smell.",
        "measure_capacitor": f"Run capacitor rated 35 uF reads {rng.uniform(33.8, 35.6):.1f} uF. Within tolerance.",
        "measure_fan_capacitor": f"Fan capacitor rated 2.5 uF reads {rng.uniform(2.4, 2.6):.2f} uF. Within tolerance.",
        "measure_thermistor": "Room sensor resistance matches its table for the current temperature.",
    }

    if cond == "clogged_air_filter":
        r["inspect_filter"] = by("About half the filter mesh is covered in dust. Some light still passes through.",
                                 "Most of the mesh is covered in a thick dust layer. Little light passes through.",
                                 "Filters are coated in grey dust. Light barely passes through.")
        r["inspect_coil"] = by("The coil is very cold and sweating: water droplets on the lower fins, no frost yet.",
                               "A thin frost line along the bottom rows of fins.",
                               "Frost and ice on the lower part of the indoor coil.")
        r["feel_supply_air"] = f"Weak airflow. The air is very cold: {rng.uniform(6, 9):.1f} C at the louvres."
    elif cond == "condenser_fan_failure":
        r["watch_outdoor_fan"] = ("The fan blade is not turning, although the compressor is running." if not mild
                                  else "The fan turns slowly, well below normal speed, and seems to struggle.")
        r["listen_compressor"] = "The compressor starts, labours louder and louder, and cuts out after a few minutes."
        r["feel_supply_air"] = f"Airflow is fine but the air is barely cool: {rng.uniform(17, 21):.1f} C."
        v = r["_variant"] = rng.choice(["bearing", "motor", "capacitor"])
        if v == "bearing":
            r["spin_fan_blade"] = by("The blade turns but feels rough. Surface rust on the hub: the bearing is starting to go.",
                                     "The blade is stiff to turn. The hub is heavily rusted: the bearing is wearing out.",
                                     "The blade is stiff and grinds when turned. The hub and shaft are rusted through: the bearing is seizing.")
            r["watch_outdoor_fan"] += by(" There is surface rust on the hub.", " The hub is heavily rusted.", " The hub is caked in rust.")
        else:
            r["spin_fan_blade"] = "The blade spins freely, so the motor or its capacitor is the problem."
        if v == "motor":
            r["inspect_fan_motor"] = by("The windings are slightly darkened in places, with a faint hot smell. The motor is overheating.",
                                        "The windings have brown scorched patches and discoloured varnish. The motor is failing.",
                                        "The windings are burnt black, with melted varnish and a strong burnt smell. The motor has failed.")
            r["measure_fan_capacitor"] += " The capacitor is fine."
        if v == "capacitor":
            u = rng.uniform(0, 1)
            val = by(2.05 + 0.25 * u, 1.3 + 0.5 * u, 0.2 + 0.7 * u)
            r["measure_fan_capacitor"] = f"Fan capacitor rated 2.5 uF reads {val:.2f} uF. Failed: it should be above 2.375 uF."
            r["inspect_capacitor"] = by("Both capacitors look normal from the outside.",
                                        "The run capacitor looks normal; the small fan capacitor's top is bulging, with a brown stain around a terminal.",
                                        "The run capacitor looks normal, but the small fan capacitor's top is bulged with brown oil running down its side.")
    elif cond == "low_refrigerant":
        r["inspect_coil"] = by("The inlet bends of the indoor coil are cold and sweating, no frost yet. The rest of the coil is not very cold.",
                               "Light frost on the inlet bends only. The rest of the coil is not cold.",
                               "Frost on the inlet end of the indoor coil only. The rest of the coil is not cold.")
        v = r["_variant"] = rng.choices(["flare", "pipe", "hidden"], weights=[5, 3, 2])[0]
        r["inspect_joints"] = {
            "flare": by("A wet, oily ring around the outdoor suction valve flare nut: an early leak.",
                        "An oily film with dust stuck to it on the outdoor suction valve flare nut.",
                        "Oily, dust-caked residue around the outdoor suction valve flare nut."),
            "pipe": by("The joints are dry, but under split insulation near the outdoor unit the copper has green corrosion patches; a leak detector finds the leak there.",
                       "The joints are dry, but under split insulation near the outdoor unit the copper is heavily pitted with corrosion; a leak detector finds the leak there.",
                       "The joints are dry, but under split insulation near the outdoor unit the copper is heavily pitted with corrosion and leaking."),
            "hidden": "No obvious residue at the joints or on the pipes.",
        }[v]
        r["feel_supply_air"] = f"Airflow is fine but the air is only cool: {rng.uniform(18, 21):.1f} C."
        r["gauge_pressures"] = (f"Suction {med('suction_psig') or rng.uniform(65, 85):.0f} psi (low), "
                                f"discharge {med('discharge_psig') or rng.uniform(230, 280):.0f} psi (low).")
    elif cond == "dirty_condenser_coil":
        r["inspect_condenser_fins"] = by("Patches of dirt, lint and a few leaves across the fins. Light passes in most places.",
                                         "Most of the fins are blocked with dust, lint and leaves. Light passes in small areas only.",
                                         "Little light passes: the outdoor fins are packed with dust, lint and leaves.") + \
            " A patch of fins is also bent flat, blocking more air."
        r["feel_supply_air"] = f"Airflow is fine. Air at the louvres is {rng.uniform(15, 17.5):.1f} C."
    elif cond == "failed_compressor_capacitor":
        r["listen_compressor"] = "A loud hum from the outdoor unit for 2-3 seconds, then a click and silence. The fan keeps running."
        r["feel_supply_air"] = f"Airflow is fine but the air is not cold: {rng.uniform(27, 30):.1f} C."
        # Visible damage (swelling, scorching, leaking) only shows on an advanced fault.
        v = r["_variant"] = rng.choices(["domed", "looks_normal"], weights=[7, 3])[0] if s >= 0.45 else "looks_normal"
        r["inspect_capacitor"] = (by("", "A scorch mark at one terminal of the run capacitor and a brown oil trace across its top.",
                                     "The run capacitor is scorched around the terminals, with brown oil running down the side.")
                                  if v == "domed" else "The run capacitor looks normal from the outside.")
        r["measure_capacitor"] = (f"Run capacitor rated 35 uF reads {library.run_cap_reading(s)} uF. "
                                  f"Failed: it should be within 5% of its rating (33.25-36.75 uF).")
        r["gauge_pressures"] = "Suction and discharge read the same: the compressor is not pumping."
    elif cond == "faulty_room_thermistor":
        r["watch_outdoor_fan"] = "The outdoor unit is not running at all."
        r["listen_compressor"] = "Nothing starts outside. The indoor fan keeps blowing."
        r["feel_supply_air"] = f"Airflow is fine but the air is at room temperature: {rng.uniform(28, 30):.1f} C."
        v = r["_variant"] = rng.choice(["broken", "loose"])
        r["compare_thermometer"] = by("The unit reports 22 C; the thermometer beside it reads 27 C.",
                                      "The unit reports 20 C; the thermometer beside it reads 27 C.",
                                      "The unit reports 18 C; the thermometer beside it reads 27 C.")
        r["measure_thermistor"] = (
            by("The wire insulation is cracked just behind the probe. The resistance jumps when the wire is moved.",
               "The wire insulation is split open behind the probe, copper strands showing. The reading flickers to open circuit when the wire is moved.",
               "The sensor tip is cracked and the wire insulation is split. It reads open circuit (no reading) at its connector.")
            if v == "broken" else
            "The sensor's clip has broken and it hangs against the cold coil, so it measures the coil, not the room. "
            "Its resistance is correct for the coil's temperature.")
        r["gauge_pressures"] = "Suction and discharge read the same: the compressor is off."
    return r


# ----------------------------------------------------------------------------- job state

@dataclass
class Job:
    id: str
    condition: str
    severity: float
    seed: int
    complaint: str
    practice: bool
    power_on: bool = True
    started: float = field(default_factory=time.time)
    steps: list[dict] = field(default_factory=list)          # actions done, in order
    blocked: list[dict] = field(default_factory=list)        # unsafe attempts
    chat: list[dict] = field(default_factory=list)           # coach and technician messages
    signals: dict = field(default_factory=dict)              # revealed signals (arrays / rows)
    results: dict = field(default_factory=dict)
    last_advice: dict | None = None
    concluded: dict | None = None
    sensor_rows: list = field(default_factory=list, repr=False)   # generated up front so gauges agree
    variant: str = ""                                             # the specific cause, see VARIANTS

    def done_ids(self) -> list[str]:
        return [s["action"] for s in self.steps]


JOBS: dict[str, Job] = {}


def new_job(condition: str | None = None, severity: float | None = None, seed: int | None = None) -> Job:
    rng = random.Random(seed)
    practice = condition is not None
    cond = condition or rng.choice(simulate.CONDITIONS)
    if cond not in simulate.CONDITIONS:
        raise ValueError(f"Unknown condition {cond}")
    sev = 1.0 if cond == "normal" else (severity if severity is not None else rng.uniform(0.35, 1.0))
    job = Job(id=uuid.uuid4().hex[:8], condition=cond, severity=round(sev, 2), seed=rng.randint(0, 10**8),
              complaint=rng.choice(COMPLAINTS[cond]), practice=practice)
    sensor_rows = sensors.generate(cond, seed=job.seed, severity=sev)
    job.results = _results(cond, sev, random.Random(job.seed), sensors.summarize(sensor_rows))
    job.variant = job.results.pop("_variant", "")
    job.sensor_rows = sensor_rows
    JOBS[job.id] = job
    return job


def get(job_id: str) -> Job:
    if job_id not in JOBS:
        raise KeyError(job_id)
    return JOBS[job_id]


def action_menu(job: Job) -> list[dict]:
    out = []
    for aid, (group, title, how, power, certified) in ACTIONS.items():
        out.append({"id": aid, "group": group, "title": title, "how": how, "power": power,
                    "certified": certified, "done": aid in job.done_ids()})
    return out


def _photos_for(cond: str, action_id: str, variant: str = "", severity: float | None = None) -> list[dict]:
    """Library photos for this action, as they look at this job's severity and specific cause."""
    return [p for p in library.photos_for(cond, severity, action_id, variant) if not p.get("hidden")]


def perform(job: Job, action_id: str) -> dict:
    """Do one action. Returns what was found, or a blocked notice if it is unsafe."""
    if job.concluded:
        raise ValueError("This job is finished. Start a new one.")
    if action_id not in ACTIONS:
        raise ValueError(f"Unknown action {action_id}")
    group, title, how, power, certified = ACTIONS[action_id]

    if power == POWER_OFF and job.power_on:
        job.blocked.append({"action": action_id, "reason": "power_on"})
        return {"blocked": True, "action": action_id, "title": title,
                "message": "Stopped: this step opens the unit or touches electrical parts. Isolate the power first "
                           "(isolator and breaker off, check zero volts, discharge capacitors)."}
    if power == POWER_ON and not job.power_on:
        return {"blocked": True, "action": action_id, "title": title, "safety_ok": True,
                "message": "The unit is switched off. Restore power and run it before this check."}

    found: dict = {"blocked": False, "action": action_id, "title": title, "how": how, "group": group}
    if certified:
        found["warning"] = "Certified refrigerant technicians only."
    sig = SIGNAL_OF.get(action_id)
    if sig == "sound":
        x = simulate.sound(job.condition, seed=job.seed, severity=job.severity)
        job.signals["sound"] = x
        found["finding"] = "Recorded 10 seconds at the outdoor unit."
    elif sig == "thermal_indoor":
        job.signals["thermal_indoor"] = simulate.thermal_indoor(job.condition, seed=job.seed, severity=job.severity)
        rd = render.thermal_readings(job.signals["thermal_indoor"])
        found["finding"] = (f"Indoor coil captured. Hottest {rd['hottest_point_c']} C ({rd['hottest_point_at']}), "
                            f"coldest {rd['coldest_point_c']} C ({rd['coldest_point_at']}).")
    elif sig == "thermal_outdoor":
        job.signals["thermal_outdoor"] = simulate.thermal_outdoor(job.condition, seed=job.seed, severity=job.severity)
        rd = render.thermal_readings(job.signals["thermal_outdoor"])
        found["finding"] = (f"Outdoor unit captured. Hottest {rd['hottest_point_c']} C ({rd['hottest_point_at']}), "
                            f"coldest {rd['coldest_point_c']} C ({rd['coldest_point_at']}).")
    elif sig == "sensors":
        job.signals["sensors"] = job.sensor_rows
        found["finding"] = "30 minutes of readings logged."
    elif action_id == "read_alarm_log":
        rows = alarms.generate(job.condition, seed=job.seed, severity=job.severity, variant=job.variant)
        job.signals["alarms"] = rows
        found["finding"] = alarms.describe(rows)
        found["alarms"] = {"rows": rows[-12:], "summary": alarms.summarize(rows), "total": len(rows)}
    elif action_id == "read_job_card":
        from . import cardreader, jobcards
        cards = [c for c in jobcards.listing() if c["condition"] == job.condition]
        if not cards:
            found["finding"] = "No previous job card in the customer's file."
        else:
            card = random.Random(job.seed).choice(cards)
            try:
                res = cardreader.read(jobcards.image_for(card["id"]), card["id"])
                found["finding"] = cardreader.history_summary(res["fields"])
                found["card"] = {"image": card["image"], "fields": res["fields"], "model": res["model"],
                                 "seconds": res["seconds"]}
            except Exception as e:
                found["finding"] = f"The job card could not be read ({e})."
                found["card"] = {"image": card["image"]}
    else:
        found["finding"] = job.results[action_id]
    photos = _photos_for(job.condition, action_id, job.variant, job.severity)
    if photos:
        found["photos"] = photos
        found["photo"] = photos[0]
    job.steps.append({"action": action_id, "title": title, "finding": found["finding"], "t": round(time.time() - job.started)})
    return found


def set_power(job: Job, on: bool) -> str:
    job.power_on = on
    msg = ("Power restored. The unit is running." if on else
           "Power isolated: isolator and breaker off, zero volts confirmed, capacitors discharged.")
    job.steps.append({"action": "power_on" if on else "power_off", "title": msg, "finding": msg,
                      "t": round(time.time() - job.started)})
    return msg


def score(job: Job, player_choice: str, coach_choice: str | None) -> dict:
    """Points for the reveal card."""
    correct = player_choice == job.condition
    checks = [s for s in job.steps if s["action"] in ACTIONS]
    pts = 100
    breakdown = []
    if not correct:
        pts -= 50
        breakdown.append(("Wrong diagnosis", -50))
    extra = max(0, len(checks) - 4)
    if extra:
        pts -= 5 * extra
        breakdown.append((f"{extra} check{'s' if extra > 1 else ''} beyond the first four", -5 * extra))
    unsafe = len(job.blocked)
    if unsafe:
        pts -= 15 * unsafe
        breakdown.append((f"{unsafe} unsafe attempt{'s' if unsafe > 1 else ''} stopped", -15 * unsafe))
    if correct and len(checks) <= 3 and not unsafe:
        pts += 10
        breakdown.append(("Fast, safe and right", +10))
    titles = {f["fault_id"]: f["title"] for f in knowledge.fault_menu()}
    return {
        "actual": job.condition, "actual_title": titles.get(job.condition, job.condition),
        "actual_cause": VARIANTS.get(job.condition, {}).get(job.variant, ""),
        "severity": job.severity, "player_choice": player_choice, "player_correct": correct,
        "coach_choice": coach_choice, "coach_correct": coach_choice == job.condition if coach_choice else None,
        "checks_used": len(checks), "unsafe_attempts": unsafe, "seconds": round(time.time() - job.started),
        "points": max(0, pts), "breakdown": breakdown,
    }

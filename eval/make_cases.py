"""Generate the synthetic test set: sensor logs + a technician's note + correct answer.

Notes are written in plain field language and are NOT copied from the fault guide,
so the agent cannot score by matching phrases.

Half the cases are "hard": a mild fault, and one key sensor channel is missing,
so the sensors alone cannot separate look-alike faults.

Run:  python -m eval.make_cases
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from faultlens import knowledge, sensors

OUT = Path(__file__).resolve().parent.parent / "data" / "cases"

# Channel to hide in hard cases, chosen so another condition looks similar without it.
HIDE = {
    "normal": None,
    "clogged_air_filter": "supply_air_c",
    "condenser_fan_failure": "outdoor_fan_amps",
    "dirty_condenser_coil": "outdoor_fan_amps",
    "low_refrigerant": "suction_psig",
    "failed_compressor_capacitor": "compressor_amps",
    "faulty_room_thermistor": "room_sensor_c",
}

# What a technician might actually say on site. Deliberately different wording from the guide.
FIELD_NOTES = {
    "normal": [
        "customer says it felt weak yesterday, seems fine right now",
        "cooling ok when I arrived, no noise",
    ],
    "clogged_air_filter": [
        "air from the indoor unit is very weak even on high fan",
        "customer has not had it serviced in over a year",
        "drops of water dripping from the indoor unit",
    ],
    "condenser_fan_failure": [
        "outdoor unit is very hot to touch and there is no breeze from it",
        "unit keeps cutting out after a few minutes",
        "outside unit is humming but quiet otherwise",
    ],
    "low_refrigerant": [
        "takes forever to cool the room, worse every month",
        "thin copper pipe outside has a white crust on it",
        "had a gas top-up last year",
    ],
    "dirty_condenser_coil": [
        "outdoor unit sits in a corner full of leaves and dust",
        "cooling weak in the afternoon heat, better at night",
        "outside unit sounds like it is working hard",
    ],
    "failed_compressor_capacitor": [
        "outside unit hums loudly for a few seconds then clicks",
        "outdoor fan spins but the air inside stays warm",
        "power cut and voltage fluctuation last week",
    ],
    "faulty_room_thermistor": [
        "remote shows the room much colder than it feels",
        "indoor fan blows but the outdoor unit never starts",
        "someone cleaned the indoor unit last week",
    ],
}


def main(per_fault: int = 4, seed: int = 42) -> None:
    rng = random.Random(seed)
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.json"):
        old.unlink()
    count = 0
    for fault in knowledge.fault_ids():
        for k in range(per_fault):
            hard = k % 2 == 1
            severity = rng.uniform(0.35, 0.6) if hard else rng.uniform(0.8, 1.0)
            rows = sensors.generate(fault, seed=rng.randint(0, 10**6), severity=severity)
            hidden = HIDE.get(fault) if hard else None
            if hidden:
                for r in rows:
                    r.pop(hidden, None)
            case = {
                "id": f"{fault}_{k}",
                "label": fault,
                "hard": hard,
                "severity": round(severity, 2),
                "hidden_channel": hidden,
                "sensor_csv": sensors.to_csv(rows),
                "notes": [rng.choice(FIELD_NOTES[fault])],
            }
            (OUT / f"{case['id']}.json").write_text(json.dumps(case, indent=1))
            count += 1
    print(f"Wrote {count} cases to {OUT}/")


if __name__ == "__main__":
    main()

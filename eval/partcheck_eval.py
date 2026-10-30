"""Measure Check a part on a balanced sample of library photos.

Each photo is checked exactly as in the app, with that photo (and any copy of it) left out of the
references, and compared with its known answer. SAM is skipped (it only draws the outline), so each
check costs only the Muse Spark calls.

Run:  python -m eval.partcheck_eval --limit 3      (first try: about 3 checks)
      python -m eval.partcheck_eval                (the whole sample, 15 photos)
      python -m eval.partcheck_eval --part-given   (tell it which part: skips the identify step)

Writes models/partcheck_report.json, shown under How it was tested.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image

from faultlens import config, library, partcheck

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "models" / "partcheck_report.json"

# One photo per kind of damage and level; the first three cover one of each kind of test.
SAMPLE = [
    ("wear", "wear/hub/stage_4.jpg"),
    ("severity", "condenser_fan_failure/photos/severity/winding_developing.jpg"),
    ("advanced", "failed_compressor_capacitor/photos/photo_01.jpg"),
    ("wear", "wear/filter/stage_2.jpg"),
    ("wear", "wear/coil/stage_3.jpg"),
    ("wear", "wear/pipe/stage_3.jpg"),
    ("severity", "failed_compressor_capacitor/photos/severity/meter_developing.jpg"),
    ("severity", "low_refrigerant/photos/severity/flare_developing.jpg"),
    ("severity", "low_refrigerant/photos/severity/inlet_early.jpg"),
    ("severity", "faulty_room_thermistor/photos/severity/sensor_developing.jpg"),
    ("severity", "faulty_room_thermistor/photos/severity/display_early.jpg"),
    ("advanced", "condenser_fan_failure/photos/photo_03.jpg"),
    ("advanced", "clogged_air_filter/photos/photo_02.jpg"),
    ("healthy", "parts/p15_run_capacitor.jpg"),
    ("healthy", "parts/p02_air_filter.jpg"),
]


def run(limit: int = 0, part_given: bool = False) -> dict:
    rows, t_all = [], time.time()
    sample = SAMPLE[:limit] if limit else SAMPLE
    for kind, rel in sample:
        path = library.LIBRARY / rel
        known = partcheck.known_answer(path)
        try:
            r = partcheck.check(Image.open(path), known["part_no"] if part_given else None, "", path, outline=False)
        except Exception as e:
            rows.append({"photo": rel, "kind": kind, "error": str(e)[:200]})
            print(f"{rel:62s} ERROR {e}")
            continue
        w, sv = r.get("wear"), r.get("severity")
        row = {"photo": rel, "kind": kind,
               "part_expected": known["part_no"], "part_got": r["part_no"], "part_right": known["part_no"] == r["part_no"],
               "condition_expected": known["condition"], "condition_got": r.get("condition"),
               "condition_right": known["condition"] == r.get("condition"),
               "stage_expected": known.get("stage"), "stage_got": w["stage"] if w else None,
               "stage_within_one": (abs(w["stage"] - known["stage"]) <= 1) if (w and known.get("stage")) else None,
               "level_expected": known.get("level"), "level_got": sv["level"] if sv else None,
               "level_right": (sv["level"] == known["level"]) if (sv and known.get("level")) else
                              (False if known.get("level") else None),
               "agrees": r.get("agrees_with_known"), "action": r.get("action_text"), "seconds": r["seconds"],
               "input_tokens": r["usage"].get("input_tokens", 0), "output_tokens": r["usage"].get("output_tokens", 0)}
        rows.append(row)
        mark = "OK " if row["agrees"] else "XX "
        print(f"{mark}{rel:60s} part {row['part_got']} (want {row['part_expected']}) | "
              f"{row['condition_got']} | stage {row['stage_got']}/{row['stage_expected']} | level {row['level_got']}/{row['level_expected']} | "
              f"{row['seconds']}s, {row['input_tokens'] + row['output_tokens']} tokens")

    ok = [x for x in rows if "error" not in x]

    def share(key):
        vals = [x[key] for x in ok if x.get(key) is not None]
        return {"right": sum(vals), "of": len(vals)}
    report = {"model": config.AGENT_MODEL, "mock": config.MOCK, "part_given": part_given, "photos": len(sample),
              "errors": len(rows) - len(ok),
              "right_part": share("part_right"), "right_condition": share("condition_right"),
              "wear_stage_within_one": share("stage_within_one"), "severity_level_exact": share("level_right"),
              "agrees_with_known": share("agrees"),
              "avg_seconds": round(sum(x["seconds"] for x in ok) / max(1, len(ok)), 1),
              "avg_tokens": round(sum(x["input_tokens"] + x["output_tokens"] for x in ok) / max(1, len(ok))),
              "note": "Library photos; each photo and any copy of it left out of the references. SAM outline skipped.",
              "rows": rows, "seconds_total": round(time.time() - t_all, 1)}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1))
    print(f"\nRight part {report['right_part']['right']}/{report['right_part']['of']} | "
          f"right condition {report['right_condition']['right']}/{report['right_condition']['of']} | "
          f"wear stage within one {report['wear_stage_within_one']['right']}/{report['wear_stage_within_one']['of']} | "
          f"severity level exact {report['severity_level_exact']['right']}/{report['severity_level_exact']['of']} | "
          f"about {report['avg_tokens']} tokens and {report['avg_seconds']} s per check")
    print(f"Saved {REPORT.relative_to(ROOT)}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Measure Check a part on library photos.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--part-given", action="store_true")
    a = ap.parse_args()
    run(a.limit, a.part_given)

"""Measure FaultLens on the synthetic test set.

Three ways of answering each case:
  rules          hand-written sensor rules only (baseline, free)
  agent_sensors  the agent with sensor data only
  agent_full     the agent with sensors + technician observation (+ checker)

Reports top-1 accuracy, how often it asked for more evidence, and accuracy when
it claimed to be confident. Results go to data/runs/.

Run:  python -m eval.run_eval              (all cases)
      python -m eval.run_eval --limit 6    (quick check, fewer API calls)
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from faultlens import config, sensors
from faultlens.agent import Evidence
from faultlens.pipeline import run_round

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "data" / "cases"
RUNS = ROOT / "data" / "runs"


def score(rows: list[dict]) -> dict:
    n = len(rows) or 1
    confident = [r for r in rows if r["status"] == "confident"]
    return {
        "cases": len(rows),
        "top1_accuracy": round(sum(r["correct"] for r in rows) / n, 3),
        "asked_for_more_evidence": round(sum(r["status"] != "confident" for r in rows) / n, 3),
        "accuracy_when_confident": round(sum(r["correct"] for r in confident) / len(confident), 3)
        if confident else None,
    }


def pick_cases(limit: int = 0) -> list[Path]:
    """All cases, or a balanced subset (alternating easy/hard across faults) when limited."""
    files = sorted(CASES.glob("*.json"))
    if not files:
        from eval.make_cases import main as make
        make()
        files = sorted(CASES.glob("*.json"))
    if limit and limit < len(files):
        # One easy and one hard per fault first, so a small run still covers every fault.
        ordered = sorted(files, key=lambda p: (int(p.stem.rsplit("_", 1)[1]) // 2, p.stem))
        files = ordered[:limit]
    return files


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--modes", default="rules,agent_sensors,agent_full")
    args = p.parse_args()
    _, report = evaluate(pick_cases(args.limit), args.modes.split(","), verbose=True)
    print(json.dumps(report, indent=1))


def evaluate(files: list[Path], modes: list[str], verbose: bool = False) -> tuple[list[dict], dict]:
    results = []
    for path in files:
        case = json.loads(path.read_text())
        rows = sensors.from_csv(case["sensor_csv"])
        for mode in modes:
            if mode == "rules":
                ranked = sorted(sensors.rule_scores(sensors.summarize(rows)).items(), key=lambda kv: -kv[1])
                top, prob = ranked[0]
                second = ranked[1][1]
                status = "confident" if prob - second >= 0.2 else "need_more_evidence"
                tokens = (0, 0)
            else:
                notes = case["notes"] if mode == "agent_full" else []
                out = run_round(Evidence(sensor_rows=rows, notes=notes), use_checker=(mode == "agent_full"))
                hyps = out["assessment"]["hypotheses"]
                top = hyps[0]["fault_id"] if hyps else None
                status = out["assessment"]["status"]
                tokens = (out["usage"]["input_tokens"], out["usage"]["output_tokens"])
            results.append({
                "case": case["id"], "label": case["label"], "hard": case["hard"], "mode": mode,
                "predicted": top, "correct": top == case["label"], "status": status,
                "input_tokens": tokens[0], "output_tokens": tokens[1],
            })
            if verbose:
                print(f"{case['id']:<34} {mode:<14} -> {top} ({status})")

    report = {"mock": config.MOCK, "model": config.AGENT_MODEL, "by_mode": {}}
    for mode in modes:
        mine = [r for r in results if r["mode"] == mode]
        report["by_mode"][mode] = {
            "all": score(mine),
            "easy": score([r for r in mine if not r["hard"]]),
            "hard": score([r for r in mine if r["hard"]]),
        }

    RUNS.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    with open(RUNS / f"results-{stamp}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    (RUNS / f"report-{stamp}.json").write_text(json.dumps(report, indent=1))
    return results, report


if __name__ == "__main__":
    main()

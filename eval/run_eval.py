"""Measure FaultLens on the synthetic test set.

Three ways of answering each case:
  rules          hand-written sensor rules only (baseline, free)
  agent_sensors  the agent with sensor data only
  agent_full     the agent with sensors + technician observation + alarm history (+ checker)

For the agent modes it also scores how well each conclusion cites its evidence (eval/citations.py).

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

from eval import citations
from faultlens import alarms, config, sensors
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
        **({"evidence": citations.summarize([r["citations"] for r in rows if r.get("citations")])}
           if any(r.get("citations") for r in rows) else {}),
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
    files = pick_cases(args.limit)
    results, report = evaluate(files, args.modes.split(","), verbose=True)
    print(json.dumps(report, indent=1))
    if not config.MOCK:
        # The same file the app shows under "How it was tested" (and the public demo shows instead of re-running).
        saved = ROOT / "models" / "eval_report.json"
        saved.write_text(json.dumps({"report": report, "results": results, "cases": len(files),
                                     "run_at": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}, indent=1))
        print(f"Saved {saved.relative_to(ROOT)}")


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
                cites = None
            else:
                full = mode == "agent_full"
                notes = case["notes"] if full else []
                alarm_rows = alarms.from_csv(case["alarm_csv"]) if full and case.get("alarm_csv") else []
                ev = Evidence(sensor_rows=rows, notes=notes, alarms=alarm_rows, alarms_given=bool(full and case.get("alarm_csv")))
                out = run_round(ev, use_checker=full)
                hyps = out["assessment"]["hypotheses"]
                top = hyps[0]["fault_id"] if hyps else None
                status = out["assessment"]["status"]
                tokens = (out["usage"]["input_tokens"], out["usage"]["output_tokens"])
                cites = citations.check_assessment(out["assessment"], rows, notes, alarm_rows, ev.alarms_given,
                                                   [case.get("hidden_channel")], {}, out.get("pattern_match"))
            results.append({
                "case": case["id"], "label": case["label"], "hard": case["hard"], "mode": mode,
                "predicted": top, "correct": top == case["label"], "status": status,
                "input_tokens": tokens[0], "output_tokens": tokens[1], "citations": cites,
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
        w = csv.DictWriter(f, fieldnames=[k for k in results[0].keys() if k != "citations"] + ["evidence_items", "evidence_grounded", "citation_problems"])
        w.writeheader()
        for r in results:
            c = r.get("citations") or {}
            w.writerow({**{k: v for k, v in r.items() if k != "citations"}, "evidence_items": c.get("items", ""),
                        "evidence_grounded": c.get("grounded", ""),
                        "citation_problems": "; ".join(f"{p['evidence']} -> {', '.join(p['problems'])}" for p in c.get("problems", []))})
    (RUNS / f"report-{stamp}.json").write_text(json.dumps(report, indent=1))
    return results, report


if __name__ == "__main__":
    main()

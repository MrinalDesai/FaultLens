"""Pattern recognition: one model per signal, combined into one fused answer.

Each signal (sound, two thermal images, sensor log) gets its own classifier.
Their probabilities are combined by adding log-probabilities (independent evidence),
so the fused answer works with any subset of signals that the technician uploads.

Trained only on SIMULATED reference signatures (faultlens/simulate.py). The report
says so, and the agent is told to treat the match as one piece of evidence.

Train:   python -m faultlens.pattern --per-class 200
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix

from . import features, sensors, simulate

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "pattern_model.joblib"
REPORT_PATH = ROOT / "models" / "pattern_report.json"
CLASSES = simulate.CONDITIONS
EPS = 0.02  # keeps one overconfident signal from vetoing the others


# ------------------------------------------------------------------ data

def make_sample(condition: str, seed: int, severity: float) -> dict[str, dict]:
    """All four signals for one simulated unit, as feature dicts."""
    return {
        "sound": features.sound_features(simulate.sound(condition, seed=seed, severity=severity)),
        "thermal_indoor": features.thermal_indoor_features(simulate.thermal_indoor(condition, seed=seed, severity=severity)),
        "thermal_outdoor": features.thermal_outdoor_features(simulate.thermal_outdoor(condition, seed=seed, severity=severity)),
        "sensors": features.sensor_features(sensors.generate(condition, seed=seed, severity=severity)),
    }


def _job(args):
    condition, seed, severity = args
    return condition, severity, make_sample(condition, seed, severity)


def build_dataset(per_class: int, seed_base: int, sev_range: tuple[float, float], workers: int = 0) -> list:
    rng = np.random.default_rng(seed_base)
    jobs = []
    for ci, c in enumerate(CLASSES):
        for i in range(per_class):
            sev = 1.0 if c == "normal" else float(rng.uniform(*sev_range))
            jobs.append((c, seed_base + ci * 100_000 + i, sev))
    if workers == 1:
        return [_job(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers or None) as ex:
        return list(ex.map(_job, jobs, chunksize=8))


def _matrix(rows: list, modality: str, names: list[str] | None = None):
    names = names or sorted(rows[0][2][modality].keys())
    X = np.array([[r[2][modality].get(n, np.nan) for n in names] for r in rows], dtype=np.float64)
    return X, names


# ------------------------------------------------------------------ model

class PatternModel:
    def __init__(self):
        self.models: dict = {}
        self.feature_names: dict[str, list[str]] = {}
        self.class_stats: dict[str, dict] = {}   # per modality: class -> (mean, std) per feature
        self.trained_on = ""

    def fit(self, rows: list) -> None:
        y = np.array([CLASSES.index(r[0]) for r in rows])
        for m in features.MODALITIES:
            X, names = _matrix(rows, m)
            clf = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, max_leaf_nodes=15,
                                                 l2_regularization=1.0, random_state=0)
            clf.fit(X, y)
            self.models[m] = clf
            self.feature_names[m] = names
            stats = {}
            for ci, c in enumerate(CLASSES):
                Xc = X[y == ci]
                stats[c] = (np.nanmean(Xc, axis=0), np.nanstd(Xc, axis=0) + 1e-6)
            self.class_stats[m] = stats

    def proba(self, modality: str, feats: dict) -> np.ndarray:
        names = self.feature_names[modality]
        x = np.array([[feats.get(n, np.nan) for n in names]], dtype=np.float64)
        p = np.zeros(len(CLASSES))
        model = self.models[modality]
        p[model.classes_] = model.predict_proba(x)[0]
        return p

    def fuse(self, per_modality: dict[str, np.ndarray]) -> np.ndarray:
        if not per_modality:
            return np.full(len(CLASSES), 1 / len(CLASSES))
        logp = sum(np.log(p + EPS) for p in per_modality.values())
        p = np.exp(logp - logp.max())
        return p / p.sum()

    def explain(self, modality: str, feats: dict, cls: str, top: int = 3) -> list[dict]:
        """The measurements that most separate this sample from a healthy unit, in the direction of `cls`."""
        names = self.feature_names[modality]
        x = np.array([feats.get(n, np.nan) for n in names], dtype=np.float64)
        mu_n, sd_n = self.class_stats[modality]["normal"]
        mu_c, _ = self.class_stats[modality][cls]
        z = (x - mu_n) / sd_n
        agrees = np.sign(x - mu_n) == np.sign(mu_c - mu_n)
        score = np.where(agrees & ~np.isnan(z), np.abs(z), 0)
        out = []
        for i in np.argsort(-score)[:top]:
            if score[i] < 2:
                break
            out.append({"measure": features.label(names[i]), "value": round(float(x[i]), 2),
                        "healthy_typical": round(float(mu_n[i]), 2)})
        return out

    def match(self, feats_by_modality: dict[str, dict]) -> dict:
        """Full result for the app and the agent."""
        per = {m: self.proba(m, f) for m, f in feats_by_modality.items() if f is not None}
        fused = self.fuse(per)
        order = np.argsort(-fused)
        top = CLASSES[order[0]]
        signals = {}
        for m, p in per.items():
            best = int(np.argmax(p))
            signals[m] = {
                "top": CLASSES[best],
                "confidence": round(float(p[best]), 3),
                "probabilities": {c: round(float(v), 3) for c, v in zip(CLASSES, p)},
                "agrees_with_fused": CLASSES[best] == top,
                "key_measurements": self.explain(m, feats_by_modality[m], top),
            }
        return {
            "fused": [{"condition": CLASSES[i], "probability": round(float(fused[i]), 3)} for i in order],
            "signals": signals,
            "signals_missing": [m for m in features.MODALITIES if m not in per],
            "trained_on": self.trained_on,
        }

    def save(self, path: Path = MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path = MODEL_PATH) -> "PatternModel | None":
        return joblib.load(path) if path.exists() else None


_cached: PatternModel | None = None


def get_model() -> PatternModel | None:
    global _cached
    if _cached is None:
        _cached = PatternModel.load()
    return _cached


# ------------------------------------------------------------------ evaluation

def evaluate(model: PatternModel, rows: list) -> dict:
    y = [r[0] for r in rows]

    def acc(pred):
        return round(float(np.mean([p == t for p, t in zip(pred, y)])), 3)

    per_mod = {m: [CLASSES[int(np.argmax(model.proba(m, r[2][m])))] for r in rows] for m in features.MODALITIES}
    fused_pred = [CLASSES[int(np.argmax(model.fuse({m: model.proba(m, r[2][m]) for m in features.MODALITIES})))]
                  for r in rows]
    leave_one_out = {}
    for drop in features.MODALITIES:
        keep = [m for m in features.MODALITIES if m != drop]
        pred = [CLASSES[int(np.argmax(model.fuse({m: model.proba(m, r[2][m]) for m in keep})))] for r in rows]
        leave_one_out[f"without_{drop}"] = acc(pred)
    cm = confusion_matrix(y, fused_pred, labels=CLASSES).tolist()
    return {
        "cases": len(rows),
        "single_signal_accuracy": {m: acc(p) for m, p in per_mod.items()},
        "fused_accuracy": acc(fused_pred),
        "fused_accuracy_missing_one_signal": leave_one_out,
        "confusion_matrix": {"labels": CLASSES, "rows_true_cols_predicted": cm},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the FaultLens pattern model on simulated signatures.")
    ap.add_argument("--per-class", type=int, default=200)
    ap.add_argument("--test-per-class", type=int, default=60)
    ap.add_argument("--workers", type=int, default=0, help="0 = all CPU cores")
    args = ap.parse_args()

    t0 = time.time()
    print(f"Generating {args.per_class * len(CLASSES)} training units...")
    train = build_dataset(args.per_class, seed_base=1_000_000, sev_range=(0.15, 1.0), workers=args.workers)
    print("Generating separate test units (new seeds, never seen in training)...")
    test_all = build_dataset(args.test_per_class, seed_base=9_000_000, sev_range=(0.15, 1.0), workers=args.workers)
    test_mild = build_dataset(args.test_per_class, seed_base=7_000_000, sev_range=(0.15, 0.35), workers=args.workers)

    # Build the model from the imported module, not __main__, so the saved file
    # loads correctly from the server and other scripts.
    from faultlens import pattern as _module
    model = _module.PatternModel()
    model.trained_on = (f"{len(train)} simulated units ({args.per_class} per condition), "
                        f"fault severity 15-100%")
    print("Training...")
    model.fit(train)
    model.save()

    report = {
        "data": "SIMULATED reference signatures only (faultlens/simulate.py). Not field recordings.",
        "trained_on": model.trained_on,
        "test_all_severities": evaluate(model, test_all),
        "test_mild_faults_only": evaluate(model, test_mild),
        "seconds": round(time.time() - t0, 1),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=1))
    t = report["test_all_severities"]
    mild = report["test_mild_faults_only"]
    print(f"\nSaved {MODEL_PATH.relative_to(ROOT)} and {REPORT_PATH.relative_to(ROOT)} in {report['seconds']} s")
    print("Single signal accuracy:", t["single_signal_accuracy"])
    print("Fused accuracy:", t["fused_accuracy"], "| mild faults only:", mild["fused_accuracy"])
    print("Fused, one signal missing:", t["fused_accuracy_missing_one_signal"])


if __name__ == "__main__":
    main()

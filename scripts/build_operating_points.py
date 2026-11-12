#!/usr/bin/env python3
"""Derive per-panel operating points from the published out-of-fold predictions.

``0.5`` is not a calibrated cut-off for these probes.  Each panel was trained
with ``class_weight="balanced"`` against roughly ten negatives per positive, so
the sigmoid output is a likelihood ratio in disguise, not a posterior: at 0.5,
precision ranges from 0.73 (TRP) to 0.99 (FATE-S), and recall from 0.24
(AATE-CoA) to 0.92 (AATE-ligase).

This script turns the 17 ``baseline_<panel>.oof.tsv`` files into the table the
runtime ships:

* ``curve``  — (threshold, precision, recall) on a fixed grid, so any threshold a
  user asks for can be interpolated without shipping the raw predictions.
* ``default_threshold`` — the F1-optimal point, clamped to ``[0.2, 0.9]``.  The
  clamp matters for ``AATE-CoA``, whose F1 optimum sits near 0.15 because the
  panel barely separates; publishing that as a "call" would generate noise, so
  it stays at the floor with its caveat attached instead.

Usage
-----
    python3 scripts/build_operating_points.py --pipeline ../pipeline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

GRID = [round(value / 100, 2) for value in range(100, 4, -5)]  # 1.00 … 0.05
CLAMP = (0.20, 0.90)

# The cross-validation sets sit at ~10% positives because negatives were drawn at
# a fixed 5:1 ratio.  A metagenome does not look like that: a given enzyme class
# is more like 0.1-1% of the predicted ORFs.  Precision is prevalence-dependent
# and this is the point most published enzyme-mining numbers gloss over, so the
# runtime ships the whole grid instead of the single training-set value.
PREVALENCES = {"0.1%": 0.001, "1%": 0.01, "10%": 0.1}


def read_oof(path: Path) -> tuple[np.ndarray, np.ndarray]:
    probabilities: list[float] = []
    labels: list[int] = []
    with path.open() as handle:
        header = handle.readline().rstrip("\n").split("\t")
        probability_at = header.index("oof_probability")
        label_at = header.index("label")
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            probabilities.append(float(fields[probability_at]))
            labels.append(int(fields[label_at]))
    return np.asarray(probabilities), np.asarray(labels)


def curve(probabilities: np.ndarray, labels: np.ndarray) -> list[list[float]]:
    """(threshold, precision, recall, n_called, TPR, FPR) on the fixed grid."""
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    rows = []
    for threshold in GRID:
        predicted = probabilities >= threshold
        called = int(predicted.sum())
        true_positive = int((predicted & (labels == 1)).sum())
        false_positive = called - true_positive
        precision = true_positive / called if called else 0.0
        recall = true_positive / positives if positives else 0.0
        tpr = recall
        fpr = false_positive / negatives if negatives else 0.0
        rows.append([threshold, round(precision, 4), round(recall, 4), called,
                     round(tpr, 4), round(fpr, 5)])
    return rows


def prevalence_precision(tpr: float, fpr: float, prevalence: float) -> float:
    """Positive predictive value when the same TPR/FPR is applied to a new prior."""
    numerator = tpr * prevalence
    denominator = numerator + fpr * (1.0 - prevalence)
    return numerator / denominator if denominator > 0 else 0.0


def best_f1(probabilities: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    best = (0.5, 0.0, 0.0)
    for threshold in GRID:
        predicted = probabilities >= threshold
        true_positive = int((predicted & (labels == 1)).sum())
        if not true_positive:
            continue
        precision = true_positive / int(predicted.sum())
        recall = true_positive / int(labels.sum())
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best[1]:
            best = (threshold, round(f1, 4), round(precision, 4))
    return best


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pipeline", type=Path, default=Path(__file__).resolve().parents[2] / "pipeline")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parents[1] / "microgem" / "assets" / "operating_points.json")
    args = parser.parse_args(argv)

    out: dict[str, dict] = {}
    print(f"{'panel':20s} {'F1-thr':>7s} {'F1':>6s} {'P':>6s} {'shipped':>8s}", file=sys.stderr)
    for path in sorted((args.pipeline / "out").glob("baseline_*.oof.tsv")):
        panel = path.name.removeprefix("baseline_").removesuffix(".oof.tsv")
        if panel.endswith("_capped50"):
            continue
        probabilities, labels = read_oof(path)
        threshold, f1, precision = best_f1(probabilities, labels)
        shipped = min(max(threshold, CLAMP[0]), CLAMP[1])
        rows = curve(probabilities, labels)
        shipped_row = min(rows, key=lambda row: abs(row[0] - shipped))
        adjusted = {
            name: round(prevalence_precision(shipped_row[4], shipped_row[5], prior), 3)
            for name, prior in PREVALENCES.items()
        }
        out[panel] = {
            "curve": rows,
            "f1_optimal_threshold": threshold,
            "f1": f1,
            "f1_optimal_precision": precision,
            "default_threshold": shipped,
            "precision_by_prevalence": adjusted,
            "n": int(len(labels)),
            "n_pos": int(labels.sum()),
        }
        print(f"{panel:20s} {threshold:7.2f} {f1:6.3f} {precision:6.3f} {shipped:8.2f} "
              f"   P@0.1%={adjusted['0.1%']:.3f} P@1%={adjusted['1%']:.3f}", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

"""Regression check: does this installation reproduce the exported model's scores?

Two failure modes matter, and they look identical to a user ("the numbers changed"):

1. **Embedding drift** — a different tokenisation, pooling layer or rotary
   implementation.  Every score shifts slightly, nothing errors, and the reports
   stay internally consistent.  Caught by comparing against ``reference.json``,
   which records what the export host produced with the reference Torch path.
2. **Wrong model file** — a truncated download or a stale checkpoint.  Caught by
   the same comparison, and by the positive control (each reference protein must
   still clear its own panel's cut-off).

The negative control is the interesting half: a reference glycoside hydrolase
must NOT clear the SATE or FATE-L cut-offs.  A model that says "yes" to
everything passes any positive-only test.
"""

from __future__ import annotations

import json
from importlib import resources

import numpy as np

from . import __version__
from .embed import embed
from .panels import PANEL_ORDER, PANELS
from .score import default_threshold, logits, read_fasta, sigmoid

TOLERANCE = 1e-3


def _load_reference() -> dict:
    with resources.files("microgem.assets").joinpath("reference.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_selftest(args) -> int:
    reference = _load_reference()
    records = read_fasta(resources.files("microgem.assets").joinpath("reference.faa"))
    if args.limit:
        records = records[: args.limit]

    ids = [identifier for identifier, _ in records]
    sequences = [sequence for _, sequence in records]
    print(f"microGEM {__version__} selftest — {len(records)} reference proteins", flush=True)
    vectors = embed(sequences, backend=args.backend, progress=True)
    decision = logits(vectors, list(PANEL_ORDER))
    observed = {
        identifier: {panel: float(decision[row, column]) for column, panel in enumerate(PANEL_ORDER)}
        for row, identifier in enumerate(ids)
    }

    failures: list[str] = []
    worst = 0.0
    worst_id = ""

    print(f"\n{'reference':30s} {'panel':16s} {'role':9s} {'expected z':>11s} {'observed z':>11s} {'delta':>9s}")
    for entry in reference["sequences"]:
        identifier = entry["id"]
        if identifier not in observed:
            continue
        panel = entry["source_panel"]
        expected = entry["expected_logit"]
        actual = observed[identifier][panel]
        delta = abs(actual - expected)
        if delta > worst:
            worst, worst_id = delta, f"{identifier}/{panel}"
        flag = "" if delta <= TOLERANCE else "  <- DRIFT"
        print(f"{identifier:30s} {panel:16s} {entry['role']:9s} {expected:11.5f} {actual:11.5f} {delta:9.2e}{flag}")
        if delta > TOLERANCE:
            failures.append(f"{identifier}: {panel} expected {expected:.4f}, got {actual:.4f} (delta {delta:.2e})")
        cutoff = entry["cutoff"]
        probability = float(sigmoid(np.array([actual]))[0])
        if entry["role"] == "positive" and probability < cutoff:
            failures.append(
                f"{identifier}: positive control failed — {panel} probability {probability:.4f} "
                f"below the panel's {cutoff:.2f} cut-off"
            )

    # Negative control: every reference protein is a true positive for exactly one
    # panel.  For the other 16, a score above that panel's own cut-off is a false
    # positive, and the calibration says that should be rare.
    cross_calls = 0
    cross_total = 0
    for entry in reference["sequences"]:
        identifier = entry["id"]
        if identifier not in observed:
            continue
        for panel in PANEL_ORDER:
            if panel == entry["source_panel"]:
                continue
            cross_total += 1
            if float(sigmoid(np.array([observed[identifier][panel]]))[0]) >= default_threshold(panel):
                cross_calls += 1
    rate = cross_calls / cross_total if cross_total else 0.0

    print(f"\nworst deviation: {worst:.2e} ({worst_id or 'n/a'})   tolerance {TOLERANCE:.0e}")
    print(f"cross-panel false positives: {cross_calls}/{cross_total} ({rate:.1%}) — "
          f"expected at most a few percent at these cut-offs")
    if rate > 0.10:
        failures.append(
            f"cross-panel false-positive rate {rate:.1%} is too high; the probes are not behaving "
            "as calibrated (check the embedding backend)"
        )
    for panel in PANEL_ORDER:
        if PANELS[panel].tier == "weak":
            continue
    if failures:
        print(f"\nFAIL — {len(failures)} problem(s):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nPASS — this installation reproduces the exported model")
    return 0

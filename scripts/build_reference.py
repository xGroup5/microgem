#!/usr/bin/env python3
"""Pick reference proteins and expected scores for ``microgem selftest``.

The reference set has to answer a specific question: *is this install the same
model that produced the published numbers?*  A hand-picked "looks like a lipase"
example would not — it would only show that the pipeline runs.

Two constraints shape the picks, and both come from how sigmoid saturation hides
errors:

* **Drift probes must not be saturated.**  A true positive that scores 0.99996
  stays 0.99996 through almost any embedding bug; the last three decimal places
  are where the evidence lives.  So each panel contributes *negatives* whose
  scores sit in the middle of the range (0.05-0.95), where the decision function
  is steep and a change in the embedding shows up immediately.
* **There must be a positive control per panel.**  Sensitivity without a floor
  proves nothing: a broken model that outputs 0.5 for everything would pass a
  mid-range-only test.  Each panel also contributes one of its own positives,
  which must clear that panel's cut-off.

Usage
-----
    python3 scripts/build_reference.py --pipeline ../pipeline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from microgem.embed import embed  # noqa: E402
from microgem.panels import PANEL_ORDER  # noqa: E402
from microgem.score import _weights, default_threshold  # noqa: E402

MAX_LENGTH = 1024
DRIFT_BAND = (0.05, 0.95)  # where the sigmoid still has slope


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records, identifier, chunks = [], None, []
    with path.open() as handle:
        for line in handle:
            if line.startswith(">"):
                if identifier is not None:
                    records.append((identifier, "".join(chunks)))
                header = line[1:].strip()
                identifier = header.split("|")[1] if header.count("|") >= 2 else header.split()[0]
                chunks = []
            else:
                chunks.append(line.strip())
    if identifier is not None:
        records.append((identifier, "".join(chunks)))
    return records


def spread_pick(records: list[tuple[str, str]], count: int, rng: np.random.Generator) -> list[tuple[str, str]]:
    """Sample across the length distribution rather than taking the first N."""
    if len(records) <= count:
        return list(records)
    order = np.argsort([len(sequence) for _, sequence in records])
    buckets = np.array_split(order, count)
    return [records[int(rng.choice(bucket))] for bucket in buckets if len(bucket)]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pipeline", type=Path, default=Path(__file__).resolve().parents[2] / "pipeline")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parents[1] / "microgem" / "assets")
    parser.add_argument("--drift-per-panel", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261107)
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)

    # Candidate pool per panel: all non-ambiguous sequences once, then reuse.
    pool: dict[str, tuple[list[tuple[str, str]], list[tuple[str, str]]]] = {}
    for panel in PANEL_ORDER:
        positives = read_fasta(args.pipeline / "out" / f"{panel}.nr.faa")
        negatives = read_fasta(args.pipeline / "out" / f"negatives_{panel}.faa")
        pool[panel] = (positives, negatives)

    # 1) Drift probes: negatives sampled from the whole negative set, then scored
    #    with the shipped probes to find the ones landing in the sensitive band.
    candidates: dict[str, list[tuple[str, str]]] = {}
    flat: list[tuple[str, str, str]] = []
    for panel in PANEL_ORDER:
        _, negatives = pool[panel]
        picks = spread_pick(negatives, 400, rng)
        candidates[panel] = picks
        flat.extend((panel, accession, sequence) for accession, sequence in picks)

    print(f"scoring {len(flat)} candidate negatives to find drift probes …", file=sys.stderr)
    vectors = embed([sequence for _, _, sequence in flat], backend="torch", progress=True)
    offset = 0
    drift_picks: list[tuple[str, str, str]] = []
    for panel in PANEL_ORDER:
        mean, scale, coef, intercept = _weights(panel)
        count = len(candidates[panel])
        block = vectors[offset:offset + count]
        offset += count
        probabilities = 1.0 / (1.0 + np.exp(-(((block - mean) / scale) @ coef + intercept)))
        order = np.argsort(probabilities)
        # Take from the middle of the band: far from both 0 and 1, and spread out,
        # so the probe is exercised over several distinct decision values.
        inside = [index for index in order if DRIFT_BAND[0] <= probabilities[index] <= DRIFT_BAND[1]]
        if not inside:
            inside = list(order)
        stride = max(1, len(inside) // args.drift_per_panel)
        for index in inside[::stride][: args.drift_per_panel]:
            accession, sequence = candidates[panel][index]
            drift_picks.append((panel, accession, sequence))
            print(f"  {panel:16s} {accession:10s} negative at {probabilities[index]:.4f}", file=sys.stderr)

    # 2) Positive controls: the panel's own positives must clear its cut-off.
    positives: list[tuple[str, str, str]] = []
    for panel in PANEL_ORDER:
        picks = spread_pick(pool[panel][0], 3, rng)
        flat_pos = embed([sequence for _, sequence in picks], backend="torch", progress=False)
        mean, scale, coef, intercept = _weights(panel)
        probabilities = 1.0 / (1.0 + np.exp(-(((flat_pos - mean) / scale) @ coef + intercept)))
        best = int(np.argmax(probabilities))
        accession, sequence = picks[best]
        positives.append((panel, accession, sequence))
        print(f"  {panel:16s} {accession:10s} positive at {probabilities[best]:.4f}", file=sys.stderr)

    entries = [(panel, accession, sequence, "drift") for panel, accession, sequence in drift_picks]
    entries += [(panel, accession, sequence, "positive") for panel, accession, sequence in positives]

    # Score everything once more, together, so the recorded values are what the
    # runtime will see for exactly these sequences.
    all_vectors = embed([sequence for _, _, sequence, _ in entries], backend="torch", progress=True)

    seen: set[str] = set()
    sequences_json = []
    fasta_lines = []
    for index, (panel, accession, sequence, role) in enumerate(entries):
        identifier = f"{panel}|{accession}"
        if identifier in seen:
            continue
        seen.add(identifier)
        mean, scale, coef, intercept = _weights(panel)
        z = float(((all_vectors[index] - mean) / scale) @ coef + intercept)
        sequences_json.append({
            "id": identifier,
            "source_panel": panel,
            "role": role,
            "length": len(sequence),
            # Logits, not probabilities. At 0.99996 a probability is insensitive:
            # the sigmoid runs out of significant digits before the embedding
            # error does. z keeps ~7 decimals of discriminating power.
            "expected_logit": round(z, 6),
            "expected_probability": round(float(1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))), 6),
            "cutoff": default_threshold(panel),
        })
        fasta_lines.append(f">{identifier}\n")
        for start in range(0, len(sequence), 60):
            fasta_lines.append(sequence[start:start + 60] + "\n")

    (args.out / "reference.faa").write_text("".join(fasta_lines))
    manifest = {
        "note": (
            "For each sequence, `expected` is the probability the shipped probe gives it on "
            "the export host's reference Torch embedding path. `role=drift` sequences sit in "
            "the un-saturated band and are the sensitive check; `role=positive` sequences are "
            "the per-panel positive control. microgem selftest requires |observed - expected| "
            "<= 1e-3 and no positive control below its cut-off."
        ),
        "tolerance": 1e-3,
        "sequences": sequences_json,
    }
    (args.out / "reference.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {len(sequences_json)} reference proteins → {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

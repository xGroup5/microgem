#!/usr/bin/env python3
"""Export microGEM's frozen-ESM-2 linear probes as portable NumPy weights.

The research pipeline (``pipeline/bin/07_baseline_linear_probe.py``) evaluates a
panel with a cross-validation loop: it never keeps the model that produced the
out-of-fold numbers, because the point of that script is the estimate, not the
artefact.  To *use* microGEM you need the other half: one probe trained on
every positive and negative available for the panel, so it can score sequences
that were never in the evaluation set.

This script closes that gap.  For each panel it replays the exact evaluation
pipeline — same EMBEDDINGS, same cluster-aware folds, same StandardScaler +
LogisticRegression(C=1.0, class_weight="balanced") — then refits the same
estimator on 100% of the panel and writes the weights as a ``.npz`` that needs
only NumPy at inference time.

Why NumPy and not pickle/ONNX
-----------------------------
A pickled scikit-learn estimator hard-pins the sklearn minor version and drags
a ~100 MB dependency into any machine that wants to score a protein.  The model
is ``standardise → dot(w) + b → sigmoid``; storing three float arrays makes the
runtime ``numpy`` only, keeps the whole weight file under 2 MB, and makes the
decision function auditable by eye.

Usage
-----
    python3 scripts/build_weights.py --pipeline ../pipeline
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Names must match the pipeline's own constants, or the exported vectors will
# not line up with the probes that were trained on them.
MAX_LENGTH = 1024


def read_fasta(path: Path) -> list[tuple[str, str]]:
    """``sp|ACC|NAME`` headers normalise to the bare accession, as in the pipeline."""
    records: list[tuple[str, str]] = []
    identifier = None
    chunks: list[str] = []
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


def panel_names(pipeline: Path) -> list[str]:
    out = pipeline / "out"
    names = sorted(path.name.removesuffix(".nr.faa") for path in out.glob("*.nr.faa"))
    # The stratified-sampling control set is an experiment, not a panel.
    return [name for name in names if not name.endswith("_capped50")]


def cluster_folds(pipeline: Path, panel: str, bitscore: float) -> dict[str, str]:
    """Reuse the evaluated fold assignment so the export is verifiable.

    ``bin/07`` writes ``baseline_<panel>.oof.tsv`` with ``accession, label,
    cluster, oof_probability``; the cluster column is exactly the component id
    the fold splitter used.  Reading it instead of re-running MMseqs2 keeps this
    script seconds long and makes it impossible for the exported probe to be
    trained on a split that differs from the published numbers.
    """
    path = pipeline / "out" / f"baseline_{panel}.oof.tsv"
    clusters: dict[str, str] = {}
    with path.open() as handle:
        header = handle.readline().rstrip("\n").split("\t")
        accession_at = header.index("accession")
        cluster_at = header.index("cluster")
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            clusters[fields[accession_at]] = fields[cluster_at]
    return clusters


def grouped_folds(clusters: dict[str, str], k: int, seed: int) -> dict[str, int]:
    """Identical to ``bin/07_baseline_linear_probe.py:grouped_folds``.

    The upstream function seeds a generator it never draws from; the assignment
    is therefore fully deterministic and the ``seed`` argument is kept only to
    make that explicit.
    """
    del seed
    sizes: dict[str, int] = {}
    for cluster in clusters.values():
        sizes[cluster] = sizes.get(cluster, 0) + 1
    order = sorted(sizes, key=lambda c: (-sizes[c], c))
    loads = [0] * k
    assignment: dict[str, int] = {}
    for cluster in order:
        target = int(np.argmin(loads))
        assignment[cluster] = target
        loads[target] += sizes[cluster]
    return assignment


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pipeline", type=Path, default=Path(__file__).resolve().parents[2] / "pipeline")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "microgem" / "assets")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20251105)
    parser.add_argument("--bitscore", type=float, default=200.0)
    parser.add_argument("--panel", default="all")
    args = parser.parse_args(argv)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    pipeline = args.pipeline.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    shared = pipeline / "out" / "embeddings" / "shared.npz"
    print(f"loading embedding cache {shared} …", file=sys.stderr)
    cache = np.load(shared, allow_pickle=False)
    vectors = set(cache.files)
    print(f"  {len(vectors)} cached vectors", file=sys.stderr)

    panels = panel_names(pipeline)
    if args.panel != "all":
        panels = [args.panel]

    manifest: dict[str, dict] = {}
    for panel in panels:
        started = time.time()
        positives = read_fasta(pipeline / "out" / f"{panel}.nr.faa")
        negatives = read_fasta(pipeline / "out" / f"negatives_{panel}.faa")
        records = positives + negatives
        labels = {name: 1 for name, _ in positives}
        for name, _ in negatives:
            labels[name] = 0

        missing = [name for name, _ in records if name not in vectors]
        if missing:
            raise SystemExit(f"{panel}: {len(missing)} sequences have no cached embedding; run bin/07 first")

        names = [name for name, _ in records]
        # float32 throughout: the probes are linear, so the 1e-7 difference from
        # float64 is far below the seed-to-seed spread we are trying to preserve.
        X = np.stack([cache[name] for name in names]).astype(np.float32)
        y = np.array([labels[name] for name in names], dtype=np.int64)

        # 1) Replay the published evaluation, so the manifest can state AUPRC and
        #    confirm that this export reproduces bin/07's numbers exactly.
        clusters = cluster_folds(pipeline, panel, args.bitscore)
        fold_assignment = grouped_folds(clusters, args.folds, args.seed)
        fold_of = np.array([fold_assignment[clusters[name]] for name in names])
        oof = np.zeros(len(y), dtype=float)
        for fold in range(args.folds):
            test = fold_of == fold
            train = ~test
            scaler = StandardScaler().fit(X[train])
            model = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", n_jobs=-1)
            model.fit(scaler.transform(X[train]), y[train])
            oof[test] = model.predict_proba(scaler.transform(X[test]))[:, 1]
        auc = roc_auc_score(y, oof)
        auprc = average_precision_score(y, oof)

        # 2) Refit on every row: this is the model that ships.
        scaler = StandardScaler().fit(X)
        model = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", n_jobs=-1)
        model.fit(scaler.transform(X), y)

        weight_path = args.out / f"{panel}.npz"
        np.savez_compressed(
            weight_path,
            mean=scaler.mean_.astype(np.float32),
            scale=scaler.scale_.astype(np.float32),
            coef=model.coef_.ravel().astype(np.float32),
            intercept=np.float32(model.intercept_[0]),
        )

        # 3) Sanity check: the serialised decision function must reproduce the
        #    sklearn one on the training matrix to float32 precision.
        loaded = np.load(weight_path)
        z = ((X - loaded["mean"]) / loaded["scale"]) @ loaded["coef"] + loaded["intercept"]
        reproduced = 1.0 / (1.0 + np.exp(-z))
        reference = model.predict_proba(scaler.transform(X))[:, 1]
        drift = float(np.abs(reproduced - reference).max())
        if drift > 1e-4:
            raise SystemExit(f"{panel}: exported weights drift {drift:.2e} from sklearn")

        published = json.loads((pipeline / "out" / f"baseline_{panel}.json").read_text())
        manifest[panel] = {
            "n": int(len(y)),
            "n_pos": int(y.sum()),
            "n_neg": int(len(y) - y.sum()),
            "clusters": len(set(clusters.values())),
            "auc": round(float(auc), 4),
            "auprc": round(float(auprc), 4),
            "published_auc": published["auc"],
            "published_auprc": published["auprc"],
            "baseline_auprc": round(float(y.mean()), 4),
            "reproduced": abs(auc - published["auc"]) < 5e-4 and abs(auprc - published["auprc"]) < 5e-4,
            "export_drift": drift,
            "weight_bytes": weight_path.stat().st_size,
        }
        print(
            f"{panel:20s} n={len(y):6d} pos={int(y.sum()):5d} "
            f"AUC={auc:.4f} (published {published['auc']:.4f}) "
            f"AUPRC={auprc:.4f} (published {published['auprc']:.4f}) "
            f"{'OK' if manifest[panel]['reproduced'] else 'MISMATCH'} "
            f"{time.time() - started:.1f}s",
            file=sys.stderr,
        )

    mismatched = [panel for panel, row in manifest.items() if not row["reproduced"]]
    if mismatched:
        raise SystemExit(f"panels whose evaluation did not reproduce: {', '.join(mismatched)}")

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    total = sum(row["weight_bytes"] for row in manifest.values())
    print(f"\n{len(manifest)} panels exported, {total / 1024:.0f} KiB total → {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

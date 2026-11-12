"""Scoring: 17 linear probes over one frozen embedding, plus the reporting layer.

The model itself is trivial — ``standardise → dot → sigmoid`` — and deliberately
so.  What carries the scientific weight is everything around it:

* **The threshold is not 0.5.**  A panel trained at 10% prevalence with balanced
  class weights produces a number that is *not* a posterior probability.  At 0.5
  the observed precision ranges from 0.73 (TRP) to 0.99 (FATE-S, AATE-CoA); the
  per-panel operating points in ``assets/operating_points.json`` are the honest
  cut-offs and the default when you ask for calls.
* **Panels are ranked, not summed.**  A sequence that scores 0.9 on SATE and 0.88
  on AATE-lyase is not "90% something"; the two probes were trained on different
  negatives and their scales are not comparable.  The score report says which
  panel is the best explanation, and by how much it beats the runner-up.
* **Weak panels say they are weak.**  ``AATE-CoA`` has a 30-seed AUPRC of
  0.62 ± 0.068; printing 0.503 next to AATE-ligase's 0.925 without comment
  invites exactly the wrong conclusion.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from pathlib import Path

import numpy as np

from . import embed as embed_mod
from .panels import PANELS, PANEL_ORDER, PanelInfo  # noqa: F401  (re-exported)

ASSET_PACKAGE = "microgem.assets"
DEFAULT_THRESHOLD = 0.5
TOP_PANELS = 3


@dataclass
class PanelScore:
    panel: str
    probability: float
    tier: str
    auprc: float

    @property
    def info(self) -> PanelInfo:
        return PANELS[self.panel]


@dataclass
class SequenceResult:
    """One protein: what microGEM thinks it is, and how confident that is."""

    identifier: str
    length: int
    sequence_sha1: str
    scores: dict[str, float]
    called: list[str] = field(default_factory=list)
    best: PanelScore | None = None
    runner_up: PanelScore | None = None

    @property
    def margin(self) -> float | None:
        """Probability gap between the best panel and the second best.

        A large margin means one panel explains the sequence; a small one means
        the probe cannot separate two chemistries — typically a protein at the
        boundary between FATE-L-modifying and FATE-L-releasing, or an
        amino-acid enzyme that sits between two AATE sub-panels.
        """
        if self.best is None or self.runner_up is None:
            return None
        return self.best.probability - self.runner_up.probability

    def ranking(self, top: int = TOP_PANELS) -> list[PanelScore]:
        ordered = sorted(self.scores.items(), key=lambda item: -item[1])
        return [
            PanelScore(panel, round(probability, 4), PANELS[panel].tier, PANELS[panel].auprc)
            for panel, probability in ordered[:top]
        ]

    def to_dict(self) -> dict:
        return {
            "id": self.identifier,
            "length": self.length,
            "sha1": self.sequence_sha1,
            "called": self.called,
            "best": self.best.panel if self.best else None,
            "best_probability": round(self.best.probability, 4) if self.best else None,
            "margin": round(self.margin, 4) if self.margin is not None else None,
            "scores": {panel: round(value, 4) for panel, value in self.scores.items()},
        }


@dataclass
class ScoreResult:
    """A batch result plus everything needed to reproduce it."""

    sequences: list[SequenceResult]
    panels: list[str]
    cutoffs: dict[str, float]
    backend: str
    model: str
    microgem_version: str

    @property
    def threshold(self) -> float | None:
        """The single cut-off the caller forced, or ``None`` for per-panel ones.

        Reporting ``0.5`` here when the panels actually applied their own
        calibrated cut-offs would make ``--show-operating-points`` describe a
        model nobody is running.
        """
        values = set(self.cutoffs.values())
        return values.pop() if len(values) == 1 else None

    def __len__(self) -> int:
        return len(self.sequences)

    def __iter__(self):
        return iter(self.sequences)

    @property
    def hits(self) -> list[SequenceResult]:
        return [item for item in self.sequences if item.called]

    def to_dict(self) -> dict:
        return {
            "microgem_version": self.microgem_version,
            "model": self.model,
            "backend": self.backend,
            "panels": self.panels,
            "cutoffs": {panel: round(value, 4) for panel, value in self.cutoffs.items()},
            "n_sequences": len(self.sequences),
            "n_called": len(self.hits),
            "results": [item.to_dict() for item in self.sequences],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# ── assets ─────────────────────────────────────────────────────────────────

@lru_cache(maxsize=None)
def _weights(panel: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    with resources.files(ASSET_PACKAGE).joinpath(f"{panel}.npz").open("rb") as handle:
        stored = np.load(handle)
        mean = stored["mean"].astype(np.float64)
        scale = stored["scale"].astype(np.float64)
        coef = stored["coef"].astype(np.float64)
        intercept = float(stored["intercept"])
    return mean, scale, coef, intercept


@lru_cache(maxsize=1)
def _operating_points() -> dict[str, dict]:
    """Threshold → (precision, recall) per panel, measured out-of-fold.

    Computed from the published cross-validated predictions, so these are the
    precision/recall you would actually have seen on held-out families — not on
    the training set the shipped probe was fitted on.
    """
    with resources.files(ASSET_PACKAGE).joinpath("operating_points.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def operating_point(panel: str, threshold: float = DEFAULT_THRESHOLD) -> tuple[float, float] | None:
    """Interpolated (precision, recall) for a panel at an arbitrary threshold."""
    points = _operating_points().get(panel)
    if not points:
        return None
    rows = sorted(points["curve"], key=lambda row: row[0], reverse=True)
    if threshold >= rows[0][0]:
        return rows[0][1], rows[0][2]
    if threshold <= rows[-1][0]:
        return rows[-1][1], rows[-1][2]
    # Rows are [threshold, precision, recall, n_called, TPR, FPR]; only the
    # first three are interpolated, the rest are diagnostics.
    for high, low in zip(rows, rows[1:]):
        high_t, high_p, high_r = high[0], high[1], high[2]
        low_t, low_p, low_r = low[0], low[1], low[2]
        if low_t <= threshold <= high_t:
            fraction = (threshold - low_t) / (high_t - low_t) if high_t > low_t else 0.0
            return (low_p + fraction * (high_p - low_p), low_r + fraction * (high_r - low_r))
    return None


def default_threshold(panel: str) -> float:
    """The panel's calibrated calling threshold.

    Chosen as the highest threshold whose out-of-fold recall is still >= 0.5, so
    that a panel reports at least half of its own positives; when even that is
    generous (``AATE-CoA`` manages 0.32 at 0.5), the published 0.5 is kept and
    the caveat is what carries the warning.
    """
    points = _operating_points().get(panel)
    if not points:
        return DEFAULT_THRESHOLD
    return float(points.get("default_threshold", DEFAULT_THRESHOLD))


# ── core ───────────────────────────────────────────────────────────────────

def logits(vectors: np.ndarray, panels: list[str]) -> np.ndarray:
    """The panels' raw decision values: ``standardise → dot(w) + b``.

    Exposed because the probability saturates: 0.99996 and 0.99992 are the same
    number to three decimals but differ by z = 0.19, and that difference is the
    only thing an embedding regression test can measure. ``microgem selftest``
    compares logits for exactly this reason.
    """
    out = np.zeros((len(vectors), len(panels)), dtype=np.float64)
    for column, panel in enumerate(panels):
        mean, scale, coef, intercept = _weights(panel)
        out[:, column] = ((vectors - mean) / scale) @ coef + intercept
    return out


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function.

    The clip at +/-40 is not lossy in any decision-relevant sense: ``sigmoid(-40)``
    is 4e-18 and ``sigmoid(40)`` is 1 - 4e-18, both already indistinguishable
    from 0 and 1 in float64. Clipping keeps ``exp`` from overflowing (which warns)
    and pins the saturated ends to exactly 0.0 and 1.0, so callers can compare
    against 0 and 1 without an epsilon.
    """
    clipped = np.clip(z, -40.0, 40.0)
    out = 1.0 / (1.0 + np.exp(-clipped))
    return np.where(clipped <= -40.0, 0.0, np.where(clipped >= 40.0, 1.0, out))


def _probabilities(vectors: np.ndarray, panels: list[str]) -> np.ndarray:
    return sigmoid(logits(vectors, panels)).astype(np.float32)


def describe() -> dict:
    """What this installation can do, without touching a model file."""
    from . import __version__

    return {
        "version": __version__,
        "backend": embed_mod.backend_available(),
        "cache_dir": str(embed_mod.cache_dir()),
        "numpy_model": embed_mod.numpy_available(),
        "panels": {
            panel: {
                "full_name": info.full_name,
                "tier": info.tier,
                "auprc": info.auprc,
                "auc": info.auc,
                "n_pos": info.n_pos,
                "n_neg": info.n_neg,
                "default_threshold": default_threshold(panel),
                "caveat": info.caveat,
            }
            for panel, info in PANELS.items()
        },
    }


def score(
    sequences: str | list[str] | dict[str, str],
    panel: str | None = None,
    threshold: float | None = None,
    model: str | Path | None = None,
    device: str | None = None,
    backend: str = "auto",
    batch_size: int = 16,
    progress: bool = False,
) -> ScoreResult:
    """Score one or more protein sequences against microGEM's panels.

    Parameters
    ----------
    sequences
        A single sequence string, a list of sequences, or ``{id: sequence}``.
        Use :func:`score_file` for FASTA input.
    panel
        One panel name, ``"all"``, or ``None`` (all 17). Panel names are
        case-insensitive and the shipped families autocomplete: ``"FATE-L"``
        expands to its two sub-panels.
    threshold
        Overrides each panel's calibrated cut-off. Passing 0.5 restores the
        plain "probability > half" rule, which is *not* what the panels were
        calibrated for.
    """
    records = normalise_input(sequences)
    selected = resolve_panels(panel)
    cutoffs = {
        name: (threshold if threshold is not None else default_threshold(name))
        for name in selected
    }

    cleaned = [embed_mod.clean_sequence(sequence) for _, sequence in records]
    vectors = embed_mod.embed(
        cleaned,
        model_file=Path(model) if model is not None else None,
        device=device,
        batch_size=batch_size,
        backend=backend,
        progress=progress,
    )
    if vectors.shape[0] != len(cleaned):
        raise RuntimeError(f"embedded {vectors.shape[0]} sequences for {len(cleaned)} inputs")
    probabilities = _probabilities(vectors, selected)

    results: list[SequenceResult] = []
    for index, (identifier, _) in enumerate(records):
        sequence = cleaned[index]
        scores = {name: float(probabilities[index, column]) for column, name in enumerate(selected)}
        ordered = sorted(scores.items(), key=lambda item: -item[1])
        best = PanelScore(ordered[0][0], ordered[0][1], PANELS[ordered[0][0]].tier, PANELS[ordered[0][0]].auprc)
        runner_up = (
            PanelScore(ordered[1][0], ordered[1][1], PANELS[ordered[1][0]].tier, PANELS[ordered[1][0]].auprc)
            if len(ordered) > 1
            else None
        )
        results.append(
            SequenceResult(
                identifier=identifier,
                length=len(sequence),
                sequence_sha1=hashlib.sha1(sequence.encode("ascii")).hexdigest(),
                scores=scores,
                called=[name for name in selected if scores[name] >= cutoffs[name]],
                best=best,
                runner_up=runner_up,
            )
        )

    return ScoreResult(
        sequences=results,
        panels=selected,
        cutoffs=cutoffs,
        backend=backend,
        model=embed_mod.MODEL_NAME,
        microgem_version=_version(),
    )


def _version() -> str:
    from . import __version__

    return __version__


# ── input handling ─────────────────────────────────────────────────────────

def normalise_input(sequences) -> list[tuple[str, str]]:
    """Coerce the accepted input shapes into ``[(identifier, sequence)]``."""
    if isinstance(sequences, str):
        value = sequences.strip()
        if value.startswith(">") or Path(value).is_file():
            records = read_fasta(value if Path(value).is_file() else value)
            return records
        if not value:
            raise ValueError("empty sequence")
        return [("sequence_1", value)]
    if isinstance(sequences, dict):
        records = [(str(key), str(value)) for key, value in sequences.items()]
    else:
        records = [(f"sequence_{index + 1}", str(value)) for index, value in enumerate(sequences)]
    if not records:
        raise ValueError("no sequences given")
    for identifier, sequence in records:
        if not embed_mod.clean_sequence(sequence):
            raise ValueError(f"{identifier} has an empty sequence")
    return records


def read_fasta(source: str | Path) -> list[tuple[str, str]]:
    """Parse FASTA text or a FASTA file into ``[(identifier, sequence)]``.

    The identifier is the first whitespace-delimited token of the header, except
    for UniProt-style ``sp|ACC|NAME`` headers where the accession is used, so
    that results can be joined back onto UniProt metadata without string surgery.
    Lines starting with ``;`` and blank lines are ignored, as in the pipeline.
    """
    path = Path(source) if not str(source).startswith(">") else None
    if path is not None:
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="replace")
        elif "\n" in str(source) or str(source).startswith(";"):
            # Multi-line text that is not a file: parse it, and let the "no
            # records" check below produce the actionable error.
            text = str(source)
        else:
            raise FileNotFoundError(f"no such FASTA file: {path}")
    else:
        text = str(source)

    records: list[tuple[str, str]] = []
    identifier: str | None = None
    chunks: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith(">"):
            if identifier is not None:
                records.append((identifier, "".join(chunks)))
            header = line[1:].strip()
            if header.count("|") >= 2:
                identifier = header.split("|")[1]
            else:
                identifier = header.split()[0] if header else "unnamed"
            chunks = []
        else:
            chunks.append(line)
    if identifier is not None:
        records.append((identifier, "".join(chunks)))
    if not records:
        raise ValueError(f"no FASTA records found in {path or 'the given text'}")
    return records


def score_file(
    path: str | Path,
    panel: str | None = None,
    threshold: float | None = None,
    **kwargs,
) -> ScoreResult:
    """Score every record in a FASTA file."""
    records = read_fasta(path)
    return score(dict(records), panel=panel, threshold=threshold, **kwargs)


# ── panel selection ────────────────────────────────────────────────────────

_FAMILY_ALIASES = {
    "FATE-L": ["FATE-L-modifying", "FATE-L-releasing"],
    "FATE": ["FATE-L-modifying", "FATE-L-releasing", "FATE-S"],
    "AATE": ["AATE-hydrolysis", "AATE-redox", "AATE-lyase", "AATE-ligase", "AATE-CoA"],
    "SAM": ["SATE", "SALE"],
}


def resolve_panels(panel: str | None) -> list[str]:
    """Expand a user's panel request into the ordered list of concrete panels."""
    if panel is None or str(panel).lower() in {"all", "*", ""}:
        return list(PANEL_ORDER)

    requested: list[str] = []
    for token in str(panel).replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        if token in PANELS:
            requested.append(token)
            continue
        alias = _FAMILY_ALIASES.get(token.upper())
        if alias:
            requested.extend(alias)
            continue
        lowered = {name.lower(): name for name in PANELS}
        if token.lower() in lowered:
            requested.append(lowered[token.lower()])
            continue
        raise ValueError(
            f"unknown panel {token!r}. Choose from: {', '.join(PANEL_ORDER)}; "
            f"or a family: {', '.join(_FAMILY_ALIASES)}"
        )
    # Preserve PANEL_ORDER so the matrix column order is stable across calls.
    seen = set(requested)
    return [name for name in PANEL_ORDER if name in seen]

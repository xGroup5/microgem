"""microGEM — 17-panel microbial enzyme scoring from protein sequence.

The package turns the research pipeline in ``../pipeline`` into something that
can be installed and called: a frozen ESM-2 embedding, seventeen exported linear
probes, and the reporting layer that turns 17 independent probabilities into a
claim a wet-lab scientist can act on.

Public surface::

    from microgem import score, PANELS, DESCRIPTIONS

    result = score(["MKVL...", "MSQEA..."])       # sequences as strings
    result = score("proteins.faa", panel="GLY")   # or a FASTA path, any panel
"""

from __future__ import annotations

from .score import (  # noqa: F401
    DEFAULT_THRESHOLD,
    PANELS,
    PANEL_ORDER,
    PanelInfo,
    ScoreResult,
    describe,
    score,
    score_file,
)

__version__ = "1.0.0"

__all__ = [
    "DEFAULT_THRESHOLD",
    "PANELS",
    "PANEL_ORDER",
    "PanelInfo",
    "ScoreResult",
    "__version__",
    "describe",
    "score",
    "score_file",
]

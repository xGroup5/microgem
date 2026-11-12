"""Tests that do not need the 2.6 GB checkpoint.

Everything here runs on a laptop with NumPy only: asset integrity, input parsing,
panel resolution, the decision function, the CLI surface. The embedding path is
covered separately by ``microgem selftest`` (needs the model) and by
``tests/test_embedding.py`` (skipped when torch is absent).
"""

from __future__ import annotations

import json
import subprocess
import sys
from importlib import resources
import numpy as np
import pytest

import microgem
from microgem import PANELS, score
from microgem.embed import clean_sequence, tokenize, TOKENS
from microgem.panels import PANEL_ORDER
from microgem.score import (
    _weights,
    default_threshold,
    logits,
    operating_point,
    read_fasta,
    resolve_panels,
    sigmoid,
)

ASSETS = resources.files("microgem.assets")


# ── assets ─────────────────────────────────────────────────────────────────

def test_every_panel_has_weights():
    for panel in PANEL_ORDER:
        assert ASSETS.joinpath(f"{panel}.npz").is_file(), f"missing weights for {panel}"


def test_weight_shapes_match_the_encoder():
    for panel in PANEL_ORDER:
        mean, scale, coef, intercept = _weights(panel)
        assert mean.shape == (1280,), panel
        assert scale.shape == (1280,), panel
        assert coef.shape == (1280,), panel
        assert np.isfinite(mean).all() and np.isfinite(coef).all(), panel
        # A zero scale would divide by zero; a negative one would silently flip
        # the sign of that feature.
        assert (scale > 0).all(), panel


def test_manifest_matches_the_panel_registry():
    manifest = json.loads(ASSETS.joinpath("manifest.json").read_text())
    assert set(manifest) == set(PANELS)
    for panel, row in manifest.items():
        assert row["reproduced"], f"{panel} did not reproduce its published metrics"
        assert row["n"] == row["n_pos"] + row["n_neg"]
        assert 0.0 < row["auprc"] <= 1.0


def test_panels_are_pairwise_distinct_names():
    names = [PANELS[panel].name for panel in PANEL_ORDER]
    assert len(names) == len(set(names)) == 17


def test_tiers_follow_the_documented_cutoffs():
    for panel in PANEL_ORDER:
        info = PANELS[panel]
        if info.auprc >= 0.85:
            assert info.tier == "reliable", panel
        elif info.auprc >= 0.65:
            assert info.tier == "moderate", panel
        else:
            assert info.tier == "weak", panel


def test_operating_points_cover_every_panel():
    """Every panel must have a usable (precision, recall) at any sensible cut-off.

    The curve grid runs 1.00 → 0.05, so interpolation must succeed throughout;
    a gap would mean a panel's out-of-fold predictions were never folded in.
    """
    table = json.loads(ASSETS.joinpath("operating_points.json").read_text())
    assert set(table) == set(PANEL_ORDER)
    for panel in PANEL_ORDER:
        for threshold in (0.05, 0.2, 0.5, 0.75, 0.95):
            point = operating_point(panel, threshold)
            assert point is not None, (panel, threshold)
            precision, recall = point
            assert 0.0 <= precision <= 1.0 and 0.0 <= recall <= 1.0, (panel, threshold)
        # The stored curve must be ordered by descending threshold, which is what
        # makes the interpolation above well-defined.
        thresholds = [row[0] for row in table[panel]["curve"]]
        assert thresholds == sorted(thresholds, reverse=True), panel


def test_reference_set_has_one_positive_control_per_panel():
    reference = json.loads(ASSETS.joinpath("reference.json").read_text())
    panels = {entry["source_panel"] for entry in reference["sequences"] if entry["role"] == "positive"}
    assert panels == set(PANEL_ORDER)


# ── decision function ──────────────────────────────────────────────────────

def test_sigmoid_is_stable_at_the_extremes():
    z = np.array([-1000.0, -50.0, 0.0, 50.0, 1000.0])
    with np.errstate(over="raise"):
        out = sigmoid(z)
    assert out[0] == 0.0 and out[-1] == 1.0
    assert out[2] == 0.5
    # Monotone non-decreasing everywhere (saturation makes the ends flat), and
    # strictly increasing where the sigmoid still has slope.
    assert np.all(np.diff(out) >= 0)
    assert np.all(np.diff(out[1:-1]) > 0)


def test_logits_are_affine_in_the_embedding():
    """Doubling the deviation from the mean must not be a no-op.

    This is a guard against the classic silent bug where the standardisation is
    skipped or applied twice: both leave the code running and the scores finite.
    """
    mean, scale, coef, intercept = _weights("GLY")
    base = mean.astype(np.float32)
    vector = np.stack([base, base + scale.astype(np.float32)])
    values = logits(vector, ["GLY"]).ravel()
    assert abs(values[0] - intercept) < 1e-3
    assert abs(values[1] - (intercept + float(coef.sum()))) < 1e-2


# ── input handling ─────────────────────────────────────────────────────────

def test_clean_sequence_maps_unknown_residues_to_x():
    assert clean_sequence("acd ef") == "ACDEF"
    assert clean_sequence("AC*D-") == "ACXDX"
    # B, U, Z and O are real ESM-2 tokens and must survive.
    assert clean_sequence("BUZO") == "BUZO"


def test_tokenize_matches_the_esm2_vocabulary():
    tokens = tokenize("MKV")
    assert tokens.tolist() == [TOKENS.index("<cls>"), TOKENS.index("M"),
                               TOKENS.index("K"), TOKENS.index("V"), TOKENS.index("<eos>")]


def test_read_fasta_extracts_uniprot_accessions():
    records = read_fasta(">sp|P04054|PA2GA_HUMAN Phospholipase\nMKV\n>plain_name\nAAA\n")
    assert records == [("P04054", "MKV"), ("plain_name", "AAA")]


def test_read_fasta_ignores_comments_and_blank_lines():
    records = read_fasta("; a comment\n\n>a\nMK\n\nLL\n")
    assert records == [("a", "MKLL")]


def test_read_fasta_rejects_empty_input():
    with pytest.raises(ValueError):
        read_fasta("; nothing here\n")


def test_resolve_panels_expands_families_and_keeps_order():
    assert resolve_panels(None) == list(PANEL_ORDER)
    assert resolve_panels("FATE-L") == ["FATE-L-modifying", "FATE-L-releasing"]
    assert resolve_panels("AATE") == ["AATE-hydrolysis", "AATE-redox", "AATE-lyase",
                                      "AATE-ligase", "AATE-CoA"]
    assert resolve_panels("gly") == ["GLY"]
    assert resolve_panels("GLY,SATE") == ["SATE", "GLY"]  # PANEL_ORDER wins


def test_resolve_panels_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown panel"):
        resolve_panels("NOT_A_PANEL")


def test_default_thresholds_are_inside_the_documented_clamp():
    for panel in PANEL_ORDER:
        assert 0.2 <= default_threshold(panel) <= 0.9, panel


# ── scoring without a model ────────────────────────────────────────────────

def test_score_uses_the_embedding_function(monkeypatch):
    """End-to-end without torch: stub the embedding, keep everything else real."""
    rng = np.random.default_rng(0)
    stub = rng.normal(size=(2, 1280)).astype(np.float32)

    def fake_embed(sequences, **kwargs):
        assert len(sequences) == 2
        return stub

    monkeypatch.setattr("microgem.embed.embed", fake_embed)
    result = score(["MKV", "AAA"])
    assert len(result) == 2
    assert set(result.sequences[0].scores) == set(PANEL_ORDER)
    for item in result.sequences:
        assert all(0.0 <= value <= 1.0 for value in item.scores.values())
        assert item.best is not None
        # Every call must be a panel that is genuinely above its own cut-off.
        for panel in item.called:
            assert item.scores[panel] >= default_threshold(panel)


def test_score_result_serialises_to_stable_json(monkeypatch):
    monkeypatch.setattr("microgem.embed.embed",
                        lambda sequences, **kwargs: np.zeros((len(sequences), 1280), dtype=np.float32))
    payload = json.loads(score(["MKV"]).to_json())
    assert payload["n_sequences"] == 1
    assert payload["panels"] == list(PANEL_ORDER)
    assert payload["results"][0]["id"] == "sequence_1"
    assert len(payload["results"][0]["scores"]) == 17


# ── CLI ────────────────────────────────────────────────────────────────────

def _cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "microgem.cli", *args],
                          capture_output=True, text=True, timeout=120)


def test_cli_version_and_help():
    assert microgem.__version__ in _cli("--version").stdout
    assert "score" in _cli("--help").stdout


def test_cli_panels_lists_all_seventeen():
    completed = _cli("panels")
    assert completed.returncode == 0, completed.stderr
    for panel in PANEL_ORDER:
        assert panel in completed.stdout


def test_cli_info_runs_without_a_model():
    completed = _cli("info", "--format", "json")
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["version"] == microgem.__version__
    assert len(payload["panels"]) == 17


def test_cli_rejects_a_missing_file(tmp_path):
    completed = _cli("score", str(tmp_path / "nope.faa"))
    assert completed.returncode != 0
    assert "no such FASTA" in completed.stderr or "error" in completed.stderr.lower()

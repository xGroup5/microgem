"""Embedding-path tests. Skipped automatically when the model is unavailable.

The point of these is not "does the code run" — ``microgem selftest`` covers that
end to end. It is that the two backends agree: the NumPy forward pass is a
hand-written reimplementation, and the only thing keeping it honest is a
comparison against the reference implementation on the same sequences.
"""

from __future__ import annotations

import numpy as np
import pytest

from microgem.embed import (
    backend_available,
    embed,
    numpy_available,
    tokenize,
)
from microgem.score import logits, read_fasta

HAS_TORCH = backend_available() == "torch"
HAS_NUMPY = numpy_available()

SEQUENCES = [
    "MKVLAAGIVGLNLGGWLSAALGDSQ",                       # short
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR",
    "MSTAGKVIKCKAAVLWEEKKPFSIEEVEVAPPKAHEVRIKMVAVGICGTDDHVVSGTMVTPLPVILGHEAAGIVESIGEGVTTVKPGDKVIPL",  # long
]


def _torch_and_numpy(sequences: list[str]) -> tuple[np.ndarray, np.ndarray]:
    return embed(sequences, backend="torch"), embed(sequences, backend="numpy")


@pytest.mark.skipif(not HAS_TORCH, reason="torch backend unavailable")
def test_embedding_is_deterministic():
    first = embed(SEQUENCES[:1], backend="torch")
    second = embed(SEQUENCES[:1], backend="torch")
    assert np.array_equal(first, second)


@pytest.mark.skipif(not HAS_TORCH, reason="torch backend unavailable")
def test_embedding_ignores_sequence_case_and_whitespace():
    upper = embed(["MKVLAAGIVGLNLGG"], backend="torch")
    messy = embed(["  mkv laagivglnlgg \n"], backend="torch")
    assert np.allclose(upper, messy, atol=1e-6)


@pytest.mark.skipif(not (HAS_TORCH and HAS_NUMPY), reason="both backends required")
def test_the_two_backends_produce_the_same_vector():
    """The NumPy path must match torch to float32 noise.

    A looser assertion would pass while a real bug (wrong rotary frequencies,
    wrong layer norm epsilon, wrong GELU) stayed in: those all shift the pooled
    vector by ~1e-1 to 1.0, three orders of magnitude above this bound.
    """
    reference, replica = _torch_and_numpy(SEQUENCES)
    assert reference.shape == replica.shape == (len(SEQUENCES), 1280)
    worst = float(np.abs(reference - replica).max())
    assert worst < 1e-4, f"NumPy backend deviates from torch by {worst:.2e}"


@pytest.mark.skipif(not (HAS_TORCH and HAS_NUMPY), reason="both backends required")
def test_the_two_backends_produce_the_same_scores():
    """The end-to-end consequence: identical enzyme calls, not just vectors."""
    reference, replica = _torch_and_numpy(SEQUENCES)
    panels = ["GLY", "SATE", "FATE-L-modifying", "RDX"]
    difference = float(np.abs(logits(reference, panels) - logits(replica, panels)).max())
    assert difference < 1e-3, f"decision values differ by {difference:.2e}"


@pytest.mark.skipif(not HAS_TORCH, reason="torch backend unavailable")
def test_long_sequences_are_truncated_not_rejected():
    """1024 tokens is the model limit; longer proteins must be cut, not error."""
    long_sequence = "MKVLAAGIVGLNLGGWLSA" * 100  # 1900 aa
    vector = embed([long_sequence], backend="torch")
    assert vector.shape == (1, 1280)
    assert np.isfinite(vector).all()


@pytest.mark.skipif(not HAS_TORCH, reason="torch backend unavailable")
def test_padding_does_not_change_a_pooled_vector():
    """Batching must not leak <pad> into the mean.

    Two sequences of different length embedded together must give the same
    vectors as embedding them one at a time; fair-esm masks padding, and the
    pooling slices to ``len(sequence)``. If either broke, this test fails while
    single-sequence scoring keeps working — exactly the silent failure mode.
    """
    together = embed(SEQUENCES[:2], backend="torch", batch_size=8)
    alone_first = embed(SEQUENCES[:1], backend="torch")
    alone_second = embed(SEQUENCES[1:2], backend="torch")
    assert np.allclose(together[0], alone_first[0], atol=1e-5)
    assert np.allclose(together[1], alone_second[0], atol=1e-5)


def test_tokenize_pads_only_when_asked():
    short = tokenize("MKV")
    padded = tokenize("MKV", length=6)
    assert len(short) == 5
    assert len(padded) == 8
    assert padded[:5].tolist() == short.tolist()


@pytest.mark.skipif(not HAS_TORCH, reason="torch backend unavailable")
def test_empty_input_returns_an_empty_matrix():
    assert embed([], backend="torch").shape == (0, 1280)


@pytest.mark.skipif(not HAS_TORCH, reason="torch backend unavailable")
def test_score_of_a_known_positive_is_high():
    """One end-to-end assertion with a real protein.

    Q8H116 is a reference glycoside hydrolase; if the embedding path or the GLY
    probe breaks, this drops below the cut-off and the test says so directly
    rather than through a 47-row selftest diff.
    """
    from microgem import score

    records = dict(read_fasta(__file__.replace("test_embedding.py", "../microgem/assets/reference.faa")))
    entry = next(key for key in records if key == "GLY|Q8H116")
    result = score({entry: records[entry]}, panel="GLY", backend="torch")
    assert result.sequences[0].scores["GLY"] > 0.9

"""A torch-free ESM-2 forward pass, for machines that have NumPy and nothing else.

This is a straight reimplementation of ``fair-esm``'s ESM-2 encoder, verified
against the PyTorch implementation on the 202,555 reference embeddings in the
research pipeline's cache (``microgem selftest --reference``). It is *not* a
reimplementation of the language-model head — only the encoder representation
that microGEM pools over.

Speed and space
---------------
The reference implementation runs at ~0.2 s/sequence on a modern GPU. This one is
an unoptimised CPU forward pass, so expect roughly 5-15x slower depending on how
many cores the BLAS backend uses. Weights are stored as one ``.npy`` per tensor
and memory-mapped, so the resident set stays a few hundred MB instead of the
2.6 GB the checkpoint occupies.

Faithfulness notes
------------------
Three details make the difference between "close" and "wrong", and all three are
easy to get subtly wrong when reimplementing ESM-2:

1. **Token-dropout rescaling happens at inference too.** ``ESM2.forward`` applies
   ``x * 0.88`` whenever ``token_dropout`` is on, regardless of ``eval()`` mode,
   because with no ``<mask>`` tokens the observed ratio is 0. Skipping it scales
   every representation by 1/0.88 and silently degrades every score.
2. **The layer-33 representation is post-``emb_layer_norm_after``.** The
   in-loop assignment is overwritten after the loop.
3. **Rotary embeddings are applied to q and k after the 1/sqrt(d) scaling**, and
   the scaling is applied to q only.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

EMBED_DIM = 1280
N_LAYERS = 33
N_HEADS = 20
HEAD_DIM = EMBED_DIM // N_HEADS
FFN_DIM = 5120
LAYER_NORM_EPS = 1e-5  # torch.nn.LayerNorm default; fair-esm uses it for ESM-2
TOKEN_DROPOUT_SCALE = 0.15 * 0.8  # fair-esm's mask_ratio_train


def _require_torch():
    try:
        import torch
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise SystemExit(
            "converting the checkpoint needs torch (for reading the .pt file only).\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            "or skip the conversion and use the torch backend directly."
        ) from error
    return torch


# Numerical Recipes' erfc Chebyshev fit: |absolute error| < 1.2e-7 over the whole
# real line, which is far below float32 spacing (1.2e-7 near 1.0) and ~10^5x
# tighter than the tanh approximation of GELU would be.
_ERFC_COEFFICIENTS = (
    1.00002368, 0.37409196, 0.09678418, -0.18628806, 0.27886807,
    -1.13520398, 1.48851587, -0.82215223, 0.17087277,
)


def _erfc(x: np.ndarray) -> np.ndarray:
    """Complementary error function, Numerical Recipes 6.2.2.

    ``t`` is the Chebyshev variable ``1 / (1 + 0.5 |x|)`` and the polynomial is
    ``c0 + t*(c1 + t*(c2 + …))``, so Horner's rule starts at the *last*
    coefficient and works backwards. Accumulating forwards gives a plausible
    number that is wrong in the second digit, which then shows up 33 layers later
    as a 0.1 error in an activation — see the selftest.
    """
    z = np.abs(x)
    t = 1.0 / (1.0 + 0.5 * z)
    poly = _ERFC_COEFFICIENTS[-1]
    for coefficient in reversed(_ERFC_COEFFICIENTS[:-1]):
        poly = coefficient + t * poly
    result = t * np.exp(-z * z - 1.26551223 + t * poly)
    return np.where(x >= 0, result, 2.0 - result)


def _erf(x: np.ndarray) -> np.ndarray:
    return 1.0 - _erfc(x)


def convert_checkpoint(checkpoint: Path, out_dir: Path) -> Path:
    """Split the fair-esm ``.pt`` into per-tensor ``.npy`` files.

    One file per tensor rather than a single array blob: it lets the runtime
    memory-map exactly the tensors it needs, so a 2.6 GB model does not become
    2.6 GB of resident set. Uncompressed on purpose — decompressing a 2.6 GB
    blob on every run costs more than the disk it saves.
    """
    torch = _require_torch()
    import argparse

    torch.serialization.add_safe_globals([argparse.Namespace])
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)["model"]
    out_dir.mkdir(parents=True, exist_ok=True)

    def dump(key: str, name: str) -> None:
        np.save(out_dir / f"{name}.npy", state[key].detach().cpu().numpy().astype(np.float32))

    dump("encoder.sentence_encoder.embed_tokens.weight", "embed_tokens")
    dump("encoder.sentence_encoder.emb_layer_norm_after.weight", "final_norm_weight")
    dump("encoder.sentence_encoder.emb_layer_norm_after.bias", "final_norm_bias")

    for layer in range(N_LAYERS):
        prefix = f"encoder.sentence_encoder.layers.{layer}"
        stem = f"l{layer:02d}"
        for part in ("q", "k", "v", "out"):
            dump(f"{prefix}.self_attn.{part}_proj.weight", f"{stem}_{part}_w")
            dump(f"{prefix}.self_attn.{part}_proj.bias", f"{stem}_{part}_b")
        dump(f"{prefix}.self_attn_layer_norm.weight", f"{stem}_attn_norm_w")
        dump(f"{prefix}.self_attn_layer_norm.bias", f"{stem}_attn_norm_b")
        dump(f"{prefix}.fc1.weight", f"{stem}_fc1_w")
        dump(f"{prefix}.fc1.bias", f"{stem}_fc1_b")
        dump(f"{prefix}.fc2.weight", f"{stem}_fc2_w")
        dump(f"{prefix}.fc2.bias", f"{stem}_fc2_b")
        dump(f"{prefix}.final_layer_norm.weight", f"{stem}_ffn_norm_w")
        dump(f"{prefix}.final_layer_norm.bias", f"{stem}_ffn_norm_b")

    # Take the buffer from the checkpoint itself rather than recomputing it.
    # fair-esm derives it in torch.float32, and recomputing in float32/float64
    # gives frequencies that differ in the 4th decimal — enough to move the
    # attention scores by ~2e-3 and the pooled vector by ~1.4, which is far more
    # than the float32 noise we tolerate. Copying the trained buffer removes the
    # question.
    inv_freq = state["encoder.sentence_encoder.layers.0.self_attn.rot_emb.inv_freq"]
    np.save(out_dir / "rotary_inv_freq.npy", inv_freq.detach().cpu().numpy().astype(np.float32))
    (out_dir / "config.json").write_text(json.dumps({
        "model": "esm2_t33_650M_UR50D",
        "layers": N_LAYERS,
        "heads": N_HEADS,
        "embed_dim": EMBED_DIM,
        "layer_norm_eps": LAYER_NORM_EPS,
        "token_dropout_scale": 1.0 - TOKEN_DROPOUT_SCALE,
        "source": str(checkpoint),
    }, indent=2) + "\n")
    return out_dir


class NumpyESM2:
    """Encoder-only ESM-2 with memory-mapped float32 weights."""

    def __init__(self, weights_dir: Path):
        self.dir = Path(weights_dir)
        if not (self.dir / "config.json").is_file():
            raise FileNotFoundError(f"{self.dir} is not a converted ESM-2 directory")
        self.embed_tokens = self._load("embed_tokens")
        self.final_norm_w = self._load("final_norm_weight")
        self.final_norm_b = self._load("final_norm_bias")
        inv_freq = self._load("rotary_inv_freq")
        self.scaling = HEAD_DIM ** -0.5
        self._inv_freq = inv_freq.astype(np.float32)

    def _load(self, name: str) -> np.ndarray:
        return np.load(self.dir / f"{name}.npy", mmap_mode="r")

    # ── primitives ─────────────────────────────────────────────────────────

    @staticmethod
    def _layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + LAYER_NORM_EPS) * weight + bias

    @staticmethod
    def _gelu(x: np.ndarray) -> np.ndarray:
        # fair-esm's gelu is the exact erf form, not the tanh approximation:
        # the two differ by up to 0.08 per activation, which compounds over 33
        # layers into a visible shift in the pooled vector.
        return x * 0.5 * (1.0 + _erf(x / math.sqrt(2.0)))

    @staticmethod
    def _rotate_half(x: np.ndarray) -> np.ndarray:
        first, second = np.split(x, 2, axis=-1)
        return np.concatenate((-second, first), axis=-1)

    def _rotary(self, length: int) -> tuple[np.ndarray, np.ndarray]:
        positions = np.arange(length, dtype=np.float32)
        freqs = np.outer(positions, self._inv_freq)
        emb = np.concatenate([freqs, freqs], axis=-1)
        return np.cos(emb), np.sin(emb)

    def _attention(self, x: np.ndarray, layer: str) -> np.ndarray:
        length = x.shape[0]
        q = x @ self._load(f"{layer}_q_w").T + self._load(f"{layer}_q_b")
        k = x @ self._load(f"{layer}_k_w").T + self._load(f"{layer}_k_b")
        v = x @ self._load(f"{layer}_v_w").T + self._load(f"{layer}_v_b")

        q = (q * self.scaling).reshape(length, N_HEADS, HEAD_DIM).transpose(1, 0, 2)
        k = k.reshape(length, N_HEADS, HEAD_DIM).transpose(1, 0, 2)
        v = v.reshape(length, N_HEADS, HEAD_DIM).transpose(1, 0, 2)

        cos, sin = self._rotary(length)
        # cos/sin are (length, HEAD_DIM); broadcast over the head axis.
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin

        scores = q @ k.transpose(0, 2, 1) * (1.0 / 1.0)
        scores = scores - scores.max(axis=-1, keepdims=True)
        weights = np.exp(scores)
        weights /= weights.sum(axis=-1, keepdims=True)
        context = weights @ v

        context = context.transpose(1, 0, 2).reshape(length, EMBED_DIM)
        return context @ self._load(f"{layer}_out_w").T + self._load(f"{layer}_out_b")

    # ── forward ────────────────────────────────────────────────────────────

    def embed(self, token_ids: np.ndarray) -> np.ndarray:
        """Mean-pooled layer-33 representation for one unpadded token sequence.

        The caller supplies ``token_ids`` (``microgem.embed.tokenize``) rather
        than a raw string, so the class does not need to know the vocabulary and
        the two backends cannot drift apart on tokenisation.
        """

        token_ids = np.asarray(token_ids)
        # fair-esm applies the token-dropout rescaling at inference as well.
        x = np.asarray(self.embed_tokens[token_ids], dtype=np.float32) * (1.0 - TOKEN_DROPOUT_SCALE)

        for layer in range(N_LAYERS):
            stem = f"l{layer:02d}"
            residual = x
            x = self._layer_norm(x, self._load(f"{stem}_attn_norm_w"), self._load(f"{stem}_attn_norm_b"))
            x = residual + self._attention(x, stem)

            residual = x
            x = self._layer_norm(x, self._load(f"{stem}_ffn_norm_w"), self._load(f"{stem}_ffn_norm_b"))
            x = self._gelu(x @ self._load(f"{stem}_fc1_w").T + self._load(f"{stem}_fc1_b"))
            x = x @ self._load(f"{stem}_fc2_w").T + self._load(f"{stem}_fc2_b")
            x = residual + x

        x = self._layer_norm(x, self.final_norm_w, self.final_norm_b)
        # Drop <cls> and <eos>; pool the residues only.
        return x[1:-1].mean(axis=0).astype(np.float32)


def load_checkpoint_weights(checkpoint: Path) -> dict[str, np.ndarray]:
    """Kept for API symmetry; the real conversion writes one .npy per tensor."""
    raise NotImplementedError("use microgem.numpy_model.convert_checkpoint")

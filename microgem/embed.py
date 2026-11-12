"""Frozen ESM-2 embeddings, with every heavy dependency confined to this module.

Two backends compute the same vector:

``torch``  the reference implementation — fair-esm + the 2.6 GB ``.pt``
           checkpoint, ~0.2 s/sequence on a modern GPU, ~1.4 s/sequence on CPU.
``numpy``  a pure-NumPy forward pass over the identical weights, converted once
           by ``microgem download --convert``. No torch, no CUDA, no GPU. The
           forward pass is unoptimised, so budget ~1 s/sequence; it exists so
           that ``pip install microgem`` on a laptop is enough to score a
           protein, without a 3 GB PyTorch install.

Both paths must produce the same vector to within float32 noise; ``microgem
selftest`` asserts that on reference proteins.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

MAX_LENGTH = 1024  # matches pipeline/bin/07_baseline_linear_probe.py
MODEL_NAME = "esm2_t33_650M_UR50D"
REPO_LAYER = 33

# The fair-esm release, which is what the research pipeline embedded with.
DEFAULT_MODEL_URL = f"https://dl.fbaipublicfiles.com/fair-esm/models/{MODEL_NAME}.pt"

# ESM-2 vocabulary, in checkpoint order.
TOKENS = [
    "<cls>", "<pad>", "<eos>", "<unk>", "L", "A", "G", "V", "S", "E", "R", "T",
    "I", "D", "P", "K", "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U",
    "Z", "O", ".", "-", "<null_1>", "<mask>",
]
TOKEN_INDEX = {token: index for index, token in enumerate(TOKENS)}
CLS, PAD, EOS, UNK = 0, 1, 2, 3
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
NONSTANDARD_AA = set("XBZUO")


def cache_dir() -> Path:
    """Where downloaded and converted models live.

    ``MICROGEM_HOME`` wins, then ``XDG_CACHE_HOME``, then ``~/.cache/microgem``.
    ``microgem info`` prints the resolved path, because a model cache the user
    cannot find is a support ticket waiting to happen.
    """
    override = os.environ.get("MICROGEM_HOME")
    if override:
        root = Path(override).expanduser()
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        root = (Path(xdg).expanduser() if xdg else Path.home() / ".cache") / "microgem"
    root.mkdir(parents=True, exist_ok=True)
    return root


def model_path(explicit: str | Path | None = None) -> Path:
    """Locate the ESM-2 checkpoint, or raise with actionable instructions.

    Search order: an explicit path, ``$MICROGEM_ESM2``, the microGEM cache, then
    the locations the research host already uses (``~/models/esm2`` and the
    torch hub checkpoints directory) so that machine needs no second copy.
    """
    if explicit is not None:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"--model points at {path}, which does not exist")
        return path

    candidates: list[Path] = []
    env = os.environ.get("MICROGEM_ESM2")
    if env:
        candidates.append(Path(env).expanduser())
    candidates.append(cache_dir() / f"{MODEL_NAME}.pt")
    candidates.append(Path.home() / "models" / "esm2" / f"{MODEL_NAME}.pt")
    candidates.append(Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / f"{MODEL_NAME}.pt")

    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"ESM-2 weights ({MODEL_NAME}.pt, 2.6 GB) were not found. Any of these fixes it:\n"
        f"  1. microgem download                 # fetches into {cache_dir()}\n"
        f"  2. export MICROGEM_ESM2=/path/to/{MODEL_NAME}.pt\n"
        f"  3. microgem score ... --model /path/to/{MODEL_NAME}.pt\n"
        "Searched: " + ", ".join(str(path) for path in candidates)
    )


def numpy_weights_dir() -> Path:
    """Directory holding the torch-free copy of the model (one ``.npy`` per tensor)."""
    return cache_dir() / f"{MODEL_NAME}_numpy"


def numpy_available() -> bool:
    return (numpy_weights_dir() / "config.json").is_file()


# ── sequence handling ──────────────────────────────────────────────────────

def clean_sequence(sequence: str) -> str:
    """Uppercase, strip whitespace, and map unsupported residues to ``X``.

    ``X``/``B``/``U``/``Z``/``O`` are valid ESM-2 tokens and are kept. Anything
    else (``*``, digits, ``-``) becomes ``X`` rather than being deleted: deleting
    shifts every downstream position and silently changes the pooled vector.
    """
    sequence = "".join(sequence.split()).upper()
    return "".join(char if char in STANDARD_AA or char in NONSTANDARD_AA else "X" for char in sequence)


def tokenize(sequence: str, length: int | None = None) -> np.ndarray:
    """``<cls> residues <eos>`` padded with ``<pad>``, exactly as fair-esm does.

    ``length`` defaults to ``len(sequence)``; pass a smaller value to pad, which
    only happens when batching.  The NumPy path scores one sequence at a time and
    therefore never pads — padding would change the attention mask and the mean.
    """
    length = len(sequence) if length is None else length
    ids = [CLS] + [TOKEN_INDEX.get(char, UNK) for char in sequence] + [EOS]
    if len(ids) < length + 2:
        ids.extend([PAD] * (length + 2 - len(ids)))
    return np.asarray(ids, dtype=np.int64)


# ── torch backend ──────────────────────────────────────────────────────────

def _load_torch_model(model_file: Path, device: str):
    import argparse

    import esm  # type: ignore[import-not-found]
    import torch

    # fair-esm pickles an argparse.Namespace into the checkpoint; PyTorch >= 2.6
    # refuses unknown globals by default, so the class has to be allowed
    # explicitly. This is a plain namespace object, not arbitrary code.
    torch.serialization.add_safe_globals([argparse.Namespace])
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(model_file))
    model.eval().to(device)
    return model, alphabet, torch


def embed_torch(
    sequences: list[str],
    model_file: Path | None = None,
    device: str | None = None,
    batch_size: int = 16,
    progress: bool = False,
) -> np.ndarray:
    if not sequences:
        return np.zeros((0, 1280), dtype=np.float32)
    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    model_file = model_path(model_file)
    model, alphabet, torch = _load_torch_model(model_file, device)
    converter = alphabet.get_batch_converter()

    vectors = np.zeros((len(sequences), 1280), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            chunk = sequences[start:start + batch_size]
            batch = [(str(index), sequence[: MAX_LENGTH - 2]) for index, sequence in enumerate(chunk)]
            _, _, tokens = converter(batch)
            tokens = tokens.to(device)
            hidden = model(tokens, repr_layers=[REPO_LAYER], return_contacts=False)["representations"][REPO_LAYER]
            for index, (_, sequence) in enumerate(batch):
                # Mean over residues only: position 0 is <cls> and len+1 is <eos>,
                # and including either would make every vector depend on length.
                pooled = hidden[index, 1:len(sequence) + 1].mean(dim=0)
                vectors[start + index] = pooled.float().cpu().numpy()
            if progress:
                done = min(start + batch_size, len(sequences))
                print(f"  embedded {done}/{len(sequences)}", file=sys.stderr, flush=True)
    return vectors


# ── NumPy backend ──────────────────────────────────────────────────────────

def convert_to_numpy(model_file: Path | None = None, out_dir: Path | None = None) -> Path:
    """Write the torch-free copy of the encoder weights (see ``numpy_model``)."""
    from .numpy_model import convert_checkpoint  # local import: keeps torch optional

    target = out_dir or numpy_weights_dir()
    convert_checkpoint(model_path(model_file), target)
    return target


def embed_numpy(
    sequences: list[str],
    model_file: Path | None = None,
    progress: bool = False,
) -> np.ndarray:
    from .numpy_model import NumpyESM2

    if not sequences:
        return np.zeros((0, 1280), dtype=np.float32)
    if not numpy_available():
        raise FileNotFoundError(
            f"the torch-free model is missing ({numpy_weights_dir()}).\n"
            "Create it once with:  microgem download --convert\n"
            "(that step needs torch+fair-esm; scoring afterwards does not)"
        )
    model = NumpyESM2(numpy_weights_dir())
    vectors = np.zeros((len(sequences), 1280), dtype=np.float32)
    for index, sequence in enumerate(sequences):
        trimmed = sequence[: MAX_LENGTH - 2]
        vectors[index] = model.embed(tokenize(trimmed))
        if progress and (index + 1) % 10 == 0:
            print(f"  embedded {index + 1}/{len(sequences)}", file=sys.stderr, flush=True)
    return vectors


# ── dispatch ───────────────────────────────────────────────────────────────

def embed(
    sequences: list[str],
    model_file: Path | None = None,
    device: str | None = None,
    batch_size: int = 16,
    backend: str = "auto",
    progress: bool = False,
) -> np.ndarray:
    """Embed sequences with whichever backend is available.

    ``backend`` is ``auto`` (torch when importable and the checkpoint exists,
    else NumPy), ``torch`` or ``numpy``.  ``auto`` prefers torch because it is
    far faster on GPU and also faster on CPU; the NumPy path is the fallback for
    machines without a deep-learning stack, not the recommended default.
    """
    sequences = [clean_sequence(sequence) for sequence in sequences]
    if backend == "auto":
        if _torch_ready():
            backend = "torch"
        elif numpy_available():
            backend = "numpy"
        else:
            raise FileNotFoundError(no_backend_help())
    if backend == "torch":
        return embed_torch(sequences, model_file, device, batch_size, progress)
    if backend == "numpy":
        return embed_numpy(sequences, model_file, progress)
    raise ValueError(f"unknown backend {backend!r}; expected auto, torch or numpy")


def _torch_ready() -> bool:
    """True when the PyTorch backend could actually run right now.

    Checks both halves: the packages must import (``importlib.util.find_spec``
    avoids importing 2 GB of torch just to answer the question) and the
    checkpoint must be on disk.
    """
    for module in ("torch", "esm"):
        if importlib.util.find_spec(module) is None:
            return False
    try:
        model_path()
    except FileNotFoundError:
        return False
    return True


def no_backend_help() -> str:
    """Explain the missing piece precisely: no torch, no model, or neither.

    "Nothing can embed" plus two suggestions is useless when the user's actual
    problem is a 2.6 GB download they have not run.  Distinguish the states.
    """
    have_torch = _torch_ready()
    try:
        model_path()
        have_model = True
    except FileNotFoundError:
        have_model = False

    if not have_torch and not have_model:
        return (
            "no embedding backend is usable: PyTorch is not installed and the ESM-2 "
            "weights are missing.\n"
            "  pip install torch fair-esm && microgem download"
        )
    if not have_torch and not numpy_available():
        return (
            "the ESM-2 weights are present but PyTorch is not, and the torch-free copy "
            "has not been made yet. Run the conversion once on any machine with PyTorch:\n"
            "  microgem download --convert\n"
            "or install torch here:  pip install torch fair-esm"
        )
    if have_torch and not have_model:
        return "PyTorch is installed but the ESM-2 weights are missing. Run: microgem download"
    return "no embedding backend is usable; run 'microgem info' to see what is missing"


def backend_available() -> str:
    """Which backend ``auto`` would pick right now, and whether it can run."""
    if _torch_ready():
        return "torch"
    if numpy_available():
        return "numpy"
    return "none"

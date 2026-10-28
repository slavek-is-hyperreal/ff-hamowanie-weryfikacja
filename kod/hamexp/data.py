"""Data loading, hash verification, splits and input encoding (PROTOKOL_v6 §3.2, §6).

Test sets (MNIST/F-MNIST test, QMNIST-50k) are reachable only after ``enable_test_access()``,
which is called exclusively by ``evaluate_test`` (§5.8, test T4).
"""
import gzip
import hashlib
import os
import struct
from dataclasses import dataclass

import numpy as np
import torch

from . import streams

DATA_DIR = os.environ.get(
    "HAMEXP_DATA", os.path.join(os.path.dirname(__file__), "..", "..", "dane", "raw")
)
SUMS_FILE = os.path.join(os.path.dirname(DATA_DIR.rstrip("/")), "SHA256SUMS")

FILES = {
    "mnist": {
        "train": ("mnist/train-images-idx3-ubyte.gz", "mnist/train-labels-idx1-ubyte.gz"),
        "test": ("mnist/t10k-images-idx3-ubyte.gz", "mnist/t10k-labels-idx1-ubyte.gz"),
    },
    "fmnist": {
        "train": ("fmnist/train-images-idx3-ubyte.gz", "fmnist/train-labels-idx1-ubyte.gz"),
        "test": ("fmnist/t10k-images-idx3-ubyte.gz", "fmnist/t10k-labels-idx1-ubyte.gz"),
    },
    "qmnist": {
        "test50k": ("qmnist/qmnist-test-images-idx3-ubyte.gz", "qmnist/qmnist-test-labels-idx2-int.gz"),
    },
}
TEST_PARTS = {("mnist", "test"), ("fmnist", "test"), ("qmnist", "test50k")}
N_TRAIN, N_VAL = 20000, 10000
LABEL_VALUE = 0.1  # lambda, §3.2
IN_DIM = 794

_TEST_ACCESS = False


class TestAccessError(RuntimeError):
    pass


def enable_test_access() -> None:
    global _TEST_ACCESS
    _TEST_ACCESS = True


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_sums() -> dict:
    sums = {}
    with open(SUMS_FILE) as f:
        for line in f:
            if line.strip():
                digest, rel = line.split()
                sums[rel] = digest
    return sums


def verify_file(rel: str) -> str:
    path = os.path.join(DATA_DIR, rel)
    expected = load_sums().get(rel)
    if expected is None:
        raise RuntimeError(f"no hash for {rel} in {SUMS_FILE}")
    if sha256_file(path) != expected:
        raise RuntimeError(f"hash mismatch for {rel}")
    return path


def _read_idx(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        raw = f.read()
    zero, dtype_code, ndim = struct.unpack(">HBB", raw[:4])
    assert zero == 0
    dims = struct.unpack(">" + "I" * ndim, raw[4 : 4 + 4 * ndim])
    dtype = {0x08: np.dtype(">u1"), 0x0C: np.dtype(">i4")}[dtype_code]
    return np.frombuffer(raw, dtype=dtype, offset=4 + 4 * ndim).reshape(dims)


def load(dataset: str, part: str):
    """Return (X float32 [n, 784] = x/255, y int64 [n]) on CPU."""
    if (dataset, part) in TEST_PARTS and not _TEST_ACCESS:
        raise TestAccessError(f"test data {dataset}/{part} is not accessible in training mode")
    img_rel, lab_rel = FILES[dataset][part]
    images = _read_idx(verify_file(img_rel))
    labels = _read_idx(verify_file(lab_rel))
    if dataset == "qmnist":
        images, labels = images[10000:], labels[10000:, 0]
    x = torch.from_numpy(images.reshape(len(images), -1).astype(np.float32) / 255.0)
    return x, torch.from_numpy(labels.astype(np.int64))


@dataclass
class Split:
    train_idx: torch.Tensor  # 20 000
    val_idx: torch.Tensor  # 10 000


def make_split(seed: int, n: int = 60000) -> Split:
    """§6: 60 000 -> 10 000 validation (fixed per seed) + 50 000; 20 000 training drawn from the 50 000."""
    g = streams.split_gen(seed)
    perm = torch.randperm(n, generator=g)
    rest = perm[N_VAL:]
    pick = torch.randperm(len(rest), generator=g)[:N_TRAIN]
    return Split(train_idx=rest[pick], val_idx=perm[:N_VAL])


def normalize_pixels(x: torch.Tensor) -> torch.Tensor:
    """§3.2: L2 to norm 1, x / max(||x||, 1e-12)."""
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


def encode(xn: torch.Tensor, y) -> torch.Tensor:
    """Normalized pixels [n, 784] + appended one-hot label of value LABEL_VALUE -> [n, 794]."""
    lab = torch.zeros(len(xn), 10, dtype=xn.dtype, device=xn.device)
    if isinstance(y, int):
        lab[:, y] = LABEL_VALUE
    else:
        lab[torch.arange(len(xn), device=xn.device), y] = LABEL_VALUE
    return torch.cat([xn, lab], dim=1)


def wrong_labels(y: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """Shift by 1..9 mod 10 (§3.1), drawn from stream ``neg``."""
    shift = torch.randint(1, 10, (len(y),), generator=g).to(y.device)
    return (y + shift) % 10

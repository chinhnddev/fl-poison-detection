from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

NDArrays = List[np.ndarray]


def flatten_ndarrays(arrs: Sequence[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(a).ravel() for a in arrs]).astype(np.float64, copy=False)


def unflatten_like(vec: np.ndarray, like: Sequence[np.ndarray]) -> NDArrays:
    out: NDArrays = []
    i = 0
    v = np.asarray(vec)
    for a in like:
        a = np.asarray(a)
        n = int(a.size)
        chunk = v[i : i + n].reshape(a.shape)
        out.append(np.asarray(chunk, dtype=a.dtype))
        i += n
    if i != int(v.size):
        raise ValueError("Vector size does not match shapes")
    return out


def l2_norm(vec: np.ndarray) -> float:
    return float(np.linalg.norm(vec))


def orthogonal_noise(direction: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Generate unit-norm noise approximately orthogonal to direction."""
    d = np.asarray(direction, dtype=np.float64).reshape(-1)
    n = rng.normal(0.0, 1.0, size=d.shape).astype(np.float64)
    # remove projection on direction
    dot = float(np.dot(n, d))
    n = n - dot * d
    nn = float(np.linalg.norm(n)) + 1e-12
    return (n / nn).reshape(direction.shape)


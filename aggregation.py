from typing import List, Sequence, Tuple
import numpy as np

NDArrays = List[np.ndarray]
ClientWeights = Tuple[str, NDArrays, int]  # (cid, weights, num_examples)


def weighted_fedavg(updates: Sequence[ClientWeights]) -> NDArrays:
    if not updates:
        raise ValueError("No updates to aggregate")
    total = sum(n for _, _, n in updates)
    if total <= 0:
        raise ValueError("Total num_examples must be > 0")

    n_layers = len(updates[0][1])
    out: NDArrays = []
    for li in range(n_layers):
        w0 = np.asarray(updates[0][1][li])
        acc = np.zeros_like(w0, dtype=np.float64)
        for _, w, n in updates:
            wl = np.asarray(w[li])
            acc += wl.astype(np.float64, copy=False) * n
        out.append(np.asarray((acc / total).astype(w0.dtype, copy=False)))
    return out


def _flatten(weights: NDArrays) -> np.ndarray:
    return np.concatenate([np.asarray(x).ravel() for x in weights]).astype(np.float64)


def krum(updates: Sequence[ClientWeights], byzantine_count: int = 1) -> NDArrays:
    if len(updates) == 1:
        return updates[0][1]
    vectors = [_flatten(w) for _, w, _ in updates]
    n = len(vectors)
    m = n - byzantine_count - 2
    if m < 1:
        # fallback
        return weighted_fedavg(updates)

    d = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            dij = np.linalg.norm(vectors[i] - vectors[j]) ** 2
            d[i, j] = dij
            d[j, i] = dij

    scores = []
    for i in range(n):
        row = np.sort(np.delete(d[i], i))
        scores.append(np.sum(row[:m]))
    idx = int(np.argmin(scores))
    return updates[idx][1]

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

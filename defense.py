from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import numpy as np

NDArrays = List[np.ndarray]
ClientWeights = Tuple[str, NDArrays, int]  # (cid, weights, num_examples)


@dataclass
class DefenseConfig:
    enabled: bool = True
    cosine_z: float = 1.0
    norm_z: float = 2.5
    min_votes: int = 2


def _flatten(w: NDArrays) -> np.ndarray:
    return np.concatenate([x.ravel() for x in w]).astype(np.float64)


def _z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-12)


def compute_distance_matrix(updates: Sequence[ClientWeights]) -> np.ndarray:
    vecs = [_flatten(w) for _, w, _ in updates]
    n = len(vecs)
    dm = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            v = np.linalg.norm(vecs[i] - vecs[j])
            dm[i, j] = v
            dm[j, i] = v
    return dm


def filter_suspicious(
    updates: Sequence[ClientWeights], cfg: DefenseConfig
) -> Tuple[List[ClientWeights], Dict]:
    if not cfg.enabled or len(updates) < 4:
        return list(updates), {"removed": [], "reason": "defense_disabled_or_small_n"}

    vecs = [_flatten(w) for _, w, _ in updates]
    n = len(vecs)
    votes = np.zeros(n, dtype=int)

    # 1) cosine similarity outlier
    norms = np.array([np.linalg.norm(v) + 1e-12 for v in vecs])
    cos = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            c = float(np.dot(vecs[i], vecs[j]) / (norms[i] * norms[j]))
            cos[i, j] = c
            cos[j, i] = c
    avg_cos = (cos.sum(axis=1) - 1.0) / (n - 1)
    z_cos = _z(avg_cos)
    votes[np.where(z_cos < -abs(cfg.cosine_z))[0]] += 1

    # 2) norm anomaly
    z_norm = _z(norms)
    votes[np.where(np.abs(z_norm) > abs(cfg.norm_z))[0]] += 1

    # 3) clustering (simple: minority by distance-to-centroid split)
    centroid = np.mean(np.stack(vecs, axis=0), axis=0)
    dist = np.array([np.linalg.norm(v - centroid) for v in vecs])
    cut = np.median(dist)
    minority = np.where(dist > cut)[0]
    if len(minority) < n / 2:
        votes[minority] += 1

    bad_idx = np.where(votes >= cfg.min_votes)[0].tolist()
    keep = [u for i, u in enumerate(updates) if i not in set(bad_idx)]
    removed_ids = [updates[i][0] for i in bad_idx]

    # avoid empty set
    if not keep:
        keep = list(updates)
        removed_ids = []
        bad_idx = []

    return keep, {
        "removed": removed_ids,
        "removed_idx": bad_idx,
        "votes": votes.tolist(),
    }
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

NDArrays = List[np.ndarray]
ClientUpdate = Tuple[str, NDArrays, int]  # (cid, delta_weights, num_examples)


@dataclass
class DefenseConfig:
    """Robust outlier filter on update vectors (delta).

    Uses robust statistics (median + MAD) to flag:
    - cosine similarity to centroid (low)
    - norm anomaly (high |z|)
    - distance to centroid (high)
    """

    enabled: bool = True
    cosine_z_threshold: float = 1.8
    norm_z_threshold: float = 2.5
    dist_z_threshold: float = 2.0
    min_score: float = 2.5
    weight_cosine: float = 1.5
    weight_norm: float = 1.0
    weight_dist: float = 1.0
    use_mad: bool = True
    min_clients: int = 4


def _flatten(w: NDArrays) -> np.ndarray:
    return np.concatenate([np.asarray(x).ravel() for x in w]).astype(np.float64, copy=False)


def _safe_norm(x: np.ndarray) -> float:
    n = float(np.linalg.norm(x))
    return n if n > 1e-12 else 1e-12


def _robust_z(x: np.ndarray, use_mad: bool) -> np.ndarray:
    if x.size < 3:
        return np.zeros_like(x, dtype=np.float64)
    if use_mad:
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + 1e-12
        return (x - med) / (1.4826 * mad)
    mean = np.mean(x)
    std = np.std(x) + 1e-12
    return (x - mean) / std


def _weighted_centroid(vecs: List[np.ndarray], weights: np.ndarray) -> np.ndarray:
    w = weights / (weights.sum() + 1e-12)
    return np.sum([wi * v for wi, v in zip(w, vecs)], axis=0)


def robust_filter(updates: Sequence[ClientUpdate], cfg: DefenseConfig) -> Tuple[List[ClientUpdate], Dict]:
    if not cfg.enabled or len(updates) < int(cfg.min_clients):
        return list(updates), {"removed_cids": [], "reason": "disabled_or_too_few_clients"}

    cids = [cid for cid, _, _ in updates]
    vecs = [_flatten(delta) for _, delta, _ in updates]
    num_examples = np.array([n for _, _, n in updates], dtype=np.float64)

    centroid = _weighted_centroid(vecs, num_examples)
    cn = _safe_norm(centroid)

    norms = np.array([_safe_norm(v) for v in vecs], dtype=np.float64)
    cos_sim = np.array([float(np.dot(v, centroid) / (_safe_norm(v) * cn)) for v in vecs], dtype=np.float64)
    distances = np.array([float(np.linalg.norm(v - centroid)) for v in vecs], dtype=np.float64)

    z_cos = _robust_z(cos_sim, cfg.use_mad)  # low cosine => negative z
    z_norm = _robust_z(norms, cfg.use_mad)
    z_dist = _robust_z(distances, cfg.use_mad)

    scores = np.zeros(len(vecs), dtype=np.float64)
    scores += float(cfg.weight_cosine) * (z_cos < -abs(float(cfg.cosine_z_threshold)))
    scores += float(cfg.weight_norm) * (np.abs(z_norm) > abs(float(cfg.norm_z_threshold)))
    scores += float(cfg.weight_dist) * (z_dist > abs(float(cfg.dist_z_threshold)))

    bad_idx = np.where(scores >= float(cfg.min_score))[0].tolist()
    bad = set(bad_idx)
    kept = [u for i, u in enumerate(updates) if i not in bad]
    removed_cids = [cids[i] for i in bad_idx]

    if not kept:
        kept = list(updates)
        removed_cids = []
        bad_idx = []

    info = {
        "removed_cids": removed_cids,
        "removed_idx": bad_idx,
        "scores": scores.tolist(),
        "z_cos": z_cos.tolist(),
        "z_norm": z_norm.tolist(),
        "z_dist": z_dist.tolist(),
        "cos_sim": cos_sim.tolist(),
        "norms": norms.tolist(),
        "distances": distances.tolist(),
    }
    return kept, info


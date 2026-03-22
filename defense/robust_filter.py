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

    Detection-aware options:
    - layer_aware: enables separate anomaly scoring for detection head layers.
      Object-detection models (e.g. YOLOv8) have a backbone/neck followed by a
      detection head. Perception poisoning attacks (backdoor, label-flip) primarily
      corrupt the head; scoring head-layer updates independently and amplifying their
      weight allows the filter to catch subtler attacks that look benign in the full
      parameter space.
    - head_layer_fraction: fraction of trailing parameter tensors treated as the
      detection head (default 0.2 ≈ last 20% of tensors).
    - head_weight_multiplier: extra anomaly-score weight applied when the head-layer
      group scores above threshold (making the filter more sensitive to head updates).

    Per-layer norm clipping:
    - clip_norm: when True, each client's per-layer delta is clipped so its L2 norm
      does not exceed clip_norm_multiplier × median_norm before aggregation. This
      bounds the maximum influence any single client can exert on individual layers,
      reducing the impact of layer-targeted perception poisoning.
    - clip_norm_multiplier: clipping multiplier relative to per-layer median norm.
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
    # Detection-aware defense
    layer_aware: bool = True
    head_layer_fraction: float = 0.2
    head_weight_multiplier: float = 2.0
    # Per-layer norm clipping
    clip_norm: bool = False
    clip_norm_multiplier: float = 5.0


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


def clip_layer_norms(updates: Sequence[ClientUpdate], multiplier: float) -> List[ClientUpdate]:
    """Clip each client's per-layer delta so its L2 norm does not exceed
    ``multiplier`` times the per-layer median norm across all clients.

    This bounds the maximum per-layer influence of any individual client,
    limiting layer-targeted perception poisoning (e.g. attacks that inject
    large weight shifts only into the detection head while keeping the
    backbone update small and normal-looking).

    Args:
        updates: Sequence of ``(cid, delta_weights, num_examples)`` tuples.
        multiplier: Each layer's delta is clipped to
            ``multiplier * median_norm_that_layer``.

    Returns:
        A new list of updates with per-layer norms clipped.
    """
    if not updates:
        return list(updates)
    n_layers = len(updates[0][1])
    if n_layers == 0:
        return list(updates)

    all_norms = np.zeros((len(updates), n_layers), dtype=np.float64)
    for ci in range(len(updates)):
        for li in range(n_layers):
            all_norms[ci, li] = _safe_norm(np.asarray(updates[ci][1][li]).ravel())
    med_norms = np.median(all_norms, axis=0) + 1e-12  # shape (n_layers,)
    clip_vals = multiplier * med_norms

    clipped: List[ClientUpdate] = []
    for cid, delta, n in updates:
        new_delta: NDArrays = []
        for li, layer in enumerate(delta):
            orig_dtype = np.asarray(layer).dtype
            arr = np.asarray(layer, dtype=np.float64)
            ln = _safe_norm(arr.ravel())
            if ln > clip_vals[li]:
                arr = arr * (clip_vals[li] / ln)
            new_delta.append(arr.astype(orig_dtype, copy=False))
        clipped.append((cid, new_delta, n))
    return clipped


def _group_anomaly_scores(
    layer_deltas: List[NDArrays],
    layer_start: int,
    layer_end: int,
    weights: np.ndarray,
    cfg: DefenseConfig,
) -> np.ndarray:
    """Compute anomaly scores for a contiguous group of layers.

    This is used by the detection-aware path to score the detection head
    layers independently from the backbone/neck layers, allowing the filter
    to be more sensitive to subtle head-only poisoning.

    Args:
        layer_deltas: List of per-layer delta arrays for each client.
        layer_start: First layer index of the group (inclusive).
        layer_end: Last layer index of the group (exclusive).
        weights: Client weights (num_examples) for centroid computation.
        cfg: Defense configuration.

    Returns:
        Array of anomaly scores, one per client.
    """
    n_clients = len(layer_deltas)
    if n_clients < 3 or layer_end <= layer_start:
        return np.zeros(n_clients, dtype=np.float64)

    group_vecs = [
        np.concatenate([np.asarray(layers[li]).ravel().astype(np.float64) for li in range(layer_start, layer_end)])
        for layers in layer_deltas
    ]

    centroid = _weighted_centroid(group_vecs, weights)
    cn = _safe_norm(centroid)

    norms = np.array([_safe_norm(v) for v in group_vecs], dtype=np.float64)
    cos_sim = np.array([float(np.dot(v, centroid) / (norms[i] * cn)) for i, v in enumerate(group_vecs)], dtype=np.float64)
    distances = np.array([float(np.linalg.norm(v - centroid)) for v in group_vecs], dtype=np.float64)

    z_cos = _robust_z(cos_sim, cfg.use_mad)
    z_norm = _robust_z(norms, cfg.use_mad)
    z_dist = _robust_z(distances, cfg.use_mad)

    scores = np.zeros(n_clients, dtype=np.float64)
    scores += float(cfg.weight_cosine) * (z_cos < -abs(float(cfg.cosine_z_threshold)))
    scores += float(cfg.weight_norm) * (np.abs(z_norm) > abs(float(cfg.norm_z_threshold)))
    scores += float(cfg.weight_dist) * (z_dist > abs(float(cfg.dist_z_threshold)))
    return scores


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

    # Detection-aware scoring: compute separate anomaly scores for detection head
    # layers and add them (weighted) to the global scores.  The detection head is
    # the sub-network most targeted by perception poisoning (backdoor, label-flip),
    # so analysing it independently gives the filter higher sensitivity to subtle
    # head-only attacks that look benign when the full parameter vector is inspected.
    head_scores = np.zeros(len(vecs), dtype=np.float64)
    if cfg.layer_aware and len(updates) >= 3:
        n_layers = len(updates[0][1])
        head_start = max(0, n_layers - max(1, int(round(n_layers * float(cfg.head_layer_fraction)))))
        layer_deltas = [delta for _, delta, _ in updates]
        head_scores = _group_anomaly_scores(layer_deltas, head_start, n_layers, num_examples, cfg)
        scores += float(cfg.head_weight_multiplier) * head_scores

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
        "head_scores": head_scores.tolist(),
    }
    return kept, info


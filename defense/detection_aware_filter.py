"""Detection-aware defense for federated object detection.

Combines gradient-based outlier detection (cosine similarity, norm, distance)
with detection-output statistics reported by clients:

- Class frequency deviation   — catches label-flip and backdoor attacks
- Bounding box distribution   — catches bbox distortion attacks
- Detection rate deviation    — catches object removal attacks
- IoU consistency scoring     — catches poisoning that disrupts prediction geometry

Clients encode a compact ``detection_stats`` JSON string in their Flower
``FitRes.metrics`` dict.  The server parses these, computes per-client
outlier scores, and removes clients whose combined score exceeds ``min_score``.

Detection statistics schema (JSON, all optional):
    {
      "class_freq":  {"0": 42, "1": 7, ...},  # raw per-class detection counts
      "bbox_w_mean": 0.23,
      "bbox_w_std":  0.09,
      "bbox_h_mean": 0.31,
      "bbox_h_std":  0.11,
      "bbox_xc_mean": 0.51,
      "bbox_yc_mean": 0.48,
      "total_detections": 123,
      "num_images":  50,
      "mean_iou_vs_global": 0.68   # optional: mean IoU of client preds vs global preds
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .robust_filter import _flatten, _robust_z, _safe_norm, _weighted_centroid

NDArrays = List[np.ndarray]
ClientUpdate = Tuple[str, NDArrays, int]  # (cid, delta_weights, num_examples)


@dataclass
class DetectionAwareDefenseConfig:
    """Configuration for the detection-aware defense.

    Gradient-based and detection-based scoring are combined additively.
    A client is removed if its total score reaches ``min_score``.
    """

    enabled: bool = True

    # ── Gradient-based thresholds (same semantics as DefenseConfig) ──────────
    cosine_z_threshold: float = 1.8
    norm_z_threshold: float = 2.5
    dist_z_threshold: float = 2.0
    use_mad: bool = True

    # ── Detection-based thresholds ────────────────────────────────────────────
    # z-score threshold for class frequency L1 distance from median
    class_freq_z_threshold: float = 2.0
    # z-score threshold for bounding-box distribution deviation from median
    bbox_z_threshold: float = 2.0
    # |z|-score threshold for detection-rate (dets/image) deviation
    detection_rate_z_threshold: float = 2.0
    # z-score threshold for IoU-vs-global deviation (higher = further from global)
    iou_z_threshold: float = 2.0

    # ── Scoring weights ────────────────────────────────────────────────────────
    weight_gradient_cosine: float = 1.5
    weight_gradient_norm: float = 1.0
    weight_gradient_dist: float = 1.0
    # High weight: class distribution is the clearest signal for label-flip/backdoor
    weight_class_freq: float = 2.0
    weight_bbox: float = 1.0
    weight_detection_rate: float = 1.0
    weight_iou: float = 1.5

    # ── General ───────────────────────────────────────────────────────────────
    min_score: float = 2.5
    min_clients: int = 4
    # Number of YOLO classes used to size the class-frequency vector
    nc: int = 80


def parse_detection_stats(metrics: Dict) -> Optional[Dict]:
    """Extract detection statistics from a Flower FitRes metrics dict.

    Returns a parsed dict or ``None`` if the stats are absent / malformed.
    """
    raw = metrics.get("detection_stats", "")
    if not raw or not isinstance(raw, str):
        return None
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _class_freq_vector(stats: Dict, nc: int) -> np.ndarray:
    """Build a *normalized* class-frequency vector of length ``nc``.

    Values sum to 1.0 (or stay all-zero if no detections were made).
    """
    freq = stats.get("class_freq") or {}
    v = np.zeros(nc, dtype=np.float64)
    for k, cnt in freq.items():
        try:
            idx = int(k)
        except (ValueError, TypeError):
            continue
        if 0 <= idx < nc:
            v[idx] += max(0.0, float(cnt))
    total = v.sum()
    if total > 0.0:
        v /= total
    return v


def _bbox_feature_vector(stats: Dict) -> np.ndarray:
    """Build a 4-element feature vector: [w_mean, h_mean, w_std, h_std]."""
    return np.array(
        [
            float(stats.get("bbox_w_mean", 0.0)),
            float(stats.get("bbox_h_mean", 0.0)),
            float(stats.get("bbox_w_std", 0.0)),
            float(stats.get("bbox_h_std", 0.0)),
        ],
        dtype=np.float64,
    )


def _detection_rate(stats: Dict) -> float:
    """Return detections-per-image (0 if unavailable)."""
    n_imgs = max(1, int(stats.get("num_images", 1) or 1))
    n_dets = max(0, int(stats.get("total_detections", 0) or 0))
    return float(n_dets) / float(n_imgs)


def detection_aware_filter(
    updates: Sequence[ClientUpdate],
    client_metrics: Sequence[Dict],
    cfg: DetectionAwareDefenseConfig,
) -> Tuple[List[ClientUpdate], Dict]:
    """Filter client updates using gradient + detection statistics.

    Args:
        updates: Sequence of ``(cid, delta_weights, num_examples)`` from all clients.
        client_metrics: Per-client Flower metrics dicts, same order as *updates*.
        cfg: Detection-aware defense configuration.

    Returns:
        A tuple ``(kept_updates, info)`` where *info* is a diagnostic dict.
    """
    if not cfg.enabled or len(updates) < int(cfg.min_clients):
        return list(updates), {"removed_cids": [], "reason": "disabled_or_too_few_clients"}

    n = len(updates)
    cids = [cid for cid, _, _ in updates]
    vecs = [_flatten(delta) for _, delta, _ in updates]
    num_examples = np.array([ne for _, _, ne in updates], dtype=np.float64)

    # ── Gradient-based anomaly scores ────────────────────────────────────────
    centroid = _weighted_centroid(vecs, num_examples)
    cn = _safe_norm(centroid)

    norms = np.array([_safe_norm(v) for v in vecs], dtype=np.float64)
    cos_sim = np.array(
        [float(np.dot(v, centroid) / (_safe_norm(v) * cn)) for v in vecs],
        dtype=np.float64,
    )
    distances = np.array(
        [float(np.linalg.norm(v - centroid)) for v in vecs],
        dtype=np.float64,
    )

    z_cos = _robust_z(cos_sim, cfg.use_mad)   # low cosine ⇒ negative z
    z_norm = _robust_z(norms, cfg.use_mad)
    z_dist = _robust_z(distances, cfg.use_mad)

    scores = np.zeros(n, dtype=np.float64)
    thr_cos = float(cfg.cosine_z_threshold)
    thr_norm = float(cfg.norm_z_threshold)
    thr_dist = float(cfg.dist_z_threshold)
    scores += float(cfg.weight_gradient_cosine) * (z_cos < -thr_cos)
    scores += float(cfg.weight_gradient_norm) * (np.abs(z_norm) > thr_norm)
    scores += float(cfg.weight_gradient_dist) * (z_dist > thr_dist)

    # ── Detection-based anomaly scores ───────────────────────────────────────
    det_stats = [parse_detection_stats(m) for m in client_metrics]
    has_det_stats = sum(1 for d in det_stats if d is not None)

    class_freq_z = np.zeros(n, dtype=np.float64)
    bbox_z = np.zeros(n, dtype=np.float64)
    det_rate_z = np.zeros(n, dtype=np.float64)
    iou_z = np.zeros(n, dtype=np.float64)

    if has_det_stats >= max(3, int(cfg.min_clients)):
        # ── Class frequency deviation (catches label-flip / backdoor) ─────────
        freq_pairs = [(i, _class_freq_vector(ds, int(cfg.nc))) for i, ds in enumerate(det_stats) if ds is not None]
        if len(freq_pairs) >= 3:
            idxs_f = [i for i, _ in freq_pairs]
            fmat = np.stack([fv for _, fv in freq_pairs], axis=0)  # (m, nc)
            median_freq = np.median(fmat, axis=0)
            l1_dists = np.array(
                [float(np.sum(np.abs(fv - median_freq))) for fv in fmat],
                dtype=np.float64,
            )
            zf = _robust_z(l1_dists, cfg.use_mad)
            for j, i in enumerate(idxs_f):
                class_freq_z[i] = float(zf[j])

        # ── Bounding-box distribution deviation (catches bbox distortion) ──────
        bbox_pairs = [(i, _bbox_feature_vector(ds)) for i, ds in enumerate(det_stats) if ds is not None]
        if len(bbox_pairs) >= 3:
            idxs_b = [i for i, _ in bbox_pairs]
            bmat = np.stack([bv for _, bv in bbox_pairs], axis=0)  # (m, 4)
            median_b = np.median(bmat, axis=0)
            l1_b = np.array(
                [float(np.sum(np.abs(bv - median_b))) for bv in bmat],
                dtype=np.float64,
            )
            zb = _robust_z(l1_b, cfg.use_mad)
            for j, i in enumerate(idxs_b):
                bbox_z[i] = float(zb[j])

        # ── Detection rate deviation (catches object removal) ─────────────────
        dr_pairs = [(i, _detection_rate(ds)) for i, ds in enumerate(det_stats) if ds is not None]
        if len(dr_pairs) >= 3:
            idxs_r = [i for i, _ in dr_pairs]
            rv = np.array([v for _, v in dr_pairs], dtype=np.float64)
            zr = _robust_z(rv, cfg.use_mad)
            for j, i in enumerate(idxs_r):
                det_rate_z[i] = float(zr[j])

        # ── IoU-vs-global deviation (catches poisoning that distorts geometry) ─
        iou_pairs = [
            (i, float(ds["mean_iou_vs_global"]))
            for i, ds in enumerate(det_stats)
            if ds is not None and "mean_iou_vs_global" in ds
        ]
        if len(iou_pairs) >= 3:
            idxs_i = [i for i, _ in iou_pairs]
            iv = np.array([v for _, v in iou_pairs], dtype=np.float64)
            # Low IoU vs global → negative z after _robust_z; flag as anomalous.
            zi = _robust_z(iv, cfg.use_mad)
            for rank, orig_idx in enumerate(idxs_i):
                iou_z[orig_idx] = float(zi[rank])

        thr_cf = float(cfg.class_freq_z_threshold)
        thr_bb = float(cfg.bbox_z_threshold)
        thr_dr = float(cfg.detection_rate_z_threshold)
        thr_iou = float(cfg.iou_z_threshold)
        scores += float(cfg.weight_class_freq) * (class_freq_z > thr_cf)
        scores += float(cfg.weight_bbox) * (bbox_z > thr_bb)
        scores += float(cfg.weight_detection_rate) * (np.abs(det_rate_z) > thr_dr)
        # Low IoU → suspicious (negative z for low values of iou_z)
        scores += float(cfg.weight_iou) * (iou_z < -thr_iou)

    bad_idx = np.where(scores >= float(cfg.min_score))[0].tolist()
    bad = set(bad_idx)
    kept = [u for i, u in enumerate(updates) if i not in bad]
    removed_cids = [cids[i] for i in bad_idx]

    # Safety: never remove everyone
    if not kept:
        kept = list(updates)
        removed_cids = []
        bad_idx = []

    info: Dict = {
        "removed_cids": removed_cids,
        "removed_idx": bad_idx,
        "scores": scores.tolist(),
        # Gradient signals
        "z_cos": z_cos.tolist(),
        "z_norm": z_norm.tolist(),
        "z_dist": z_dist.tolist(),
        "cos_sim": cos_sim.tolist(),
        "norms": norms.tolist(),
        "distances": distances.tolist(),
        # Detection signals
        "class_freq_z": class_freq_z.tolist(),
        "bbox_z": bbox_z.tolist(),
        "det_rate_z": det_rate_z.tolist(),
        "iou_z": iou_z.tolist(),
        "has_detection_stats": int(has_det_stats),
    }
    return kept, info

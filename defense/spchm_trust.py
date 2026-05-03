from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from train_yolo import (
    ReusableYOLOPredictor,
    build_root_delta,
    load_dataset_images,
    prepare_inference_image_paths,
)

NDArrays = List[np.ndarray]
ClientUpdate = Tuple[str, NDArrays, int]


@dataclass
class SPCHMTrustConfig:
    enabled: bool = True
    proxy_data_yaml: str = ""
    root_data_yaml: str = ""
    proxy_max_images: int = 32
    proxy_conf: float = 0.25
    proxy_imgsz: int = 320
    match_iou_threshold: float = 0.5
    proxy_trigger: bool = False
    proxy_trigger_size: int = 40
    proxy_trigger_value: int = 255
    proxy_trigger_position: str = "bottom_right"
    proxy_trigger_mode: str = "max"
    tau: float = 2.0
    eps: float = 1e-8
    lambda_box: float = 1.0
    lambda_cls: float = 1.0
    lambda_miss: float = 1.0
    lambda_ghost: float = 1.0
    hungarian_class_penalty: float = 0.5
    root_epochs: int = 1
    root_batch: int = 4
    root_num_workers: int = 0
    root_imgsz: int = 320
    root_device: str = "cpu"
    trust_floor: float = 0.0
    tmp_dir: str = "./artifacts/tmp_spchm"
    train_runs_dir: str = "./runs_fl/spchm_trust"
    seed: int = 1234
    train_overrides: Optional[Dict[str, Any]] = None


def _flatten_arrays(arrays: NDArrays) -> np.ndarray:
    if not arrays:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate([np.asarray(a).ravel() for a in arrays]).astype(np.float64, copy=False)


def _safe_norm(x: np.ndarray, eps: float) -> float:
    return max(float(np.linalg.norm(x)), float(eps))


def _cosine_similarity(a: NDArrays, b: NDArrays, eps: float) -> float:
    va = _flatten_arrays(a)
    vb = _flatten_arrays(b)
    if va.size == 0 or vb.size == 0:
        return 0.0
    return float(np.dot(va, vb) / (_safe_norm(va, eps) * _safe_norm(vb, eps)))


def _cosine_root_similarity(delta: NDArrays, delta_root: Optional[NDArrays], eps: float) -> float:
    if delta_root is None:
        return 1.0
    return max(0.0, _cosine_similarity(delta, delta_root, eps))


def _xyxy_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0.0 else 0.0


def score_prediction_consistency(
    reference_detections: Sequence[Dict[str, Any]],
    client_detections: Sequence[Dict[str, Any]],
    class_penalty: float,
    match_iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    ref_count = len(reference_detections)
    client_count = len(client_detections)

    if ref_count == 0 and client_count == 0:
        return {
            "matched_pairs": 0,
            "d_box": 0.0,
            "d_cls": 0.0,
            "r_miss": 0.0,
            "r_ghost": 0.0,
            "matched_iou": [],
        }

    matched_rows = np.array([], dtype=np.int64)
    matched_cols = np.array([], dtype=np.int64)
    matched_iou: List[float] = []
    matched_cls_mismatch: List[float] = []

    if ref_count > 0 and client_count > 0:
        cost = np.zeros((ref_count, client_count), dtype=np.float64)
        iou_matrix = np.zeros((ref_count, client_count), dtype=np.float64)
        for row_idx, ref_det in enumerate(reference_detections):
            for col_idx, client_det in enumerate(client_detections):
                iou_val = _xyxy_iou(ref_det["xyxy"], client_det["xyxy"])
                cls_mismatch = float(int(ref_det["cls"]) != int(client_det["cls"]))
                iou_matrix[row_idx, col_idx] = iou_val
                cost[row_idx, col_idx] = (1.0 - iou_val) + float(class_penalty) * cls_mismatch
        matched_rows, matched_cols = linear_sum_assignment(cost)
        for row_idx, col_idx in zip(matched_rows.tolist(), matched_cols.tolist()):
            ref_det = reference_detections[row_idx]
            client_det = client_detections[col_idx]
            iou_val = float(iou_matrix[row_idx, col_idx])
            if iou_val < float(match_iou_threshold):
                continue
            matched_iou.append(iou_val)
            matched_cls_mismatch.append(float(int(ref_det["cls"]) != int(client_det["cls"])))

    matched_pairs = len(matched_iou)
    d_box = float(np.mean([1.0 - iou_val for iou_val in matched_iou])) if matched_iou else 0.0
    d_cls = float(np.mean(matched_cls_mismatch)) if matched_cls_mismatch else 0.0

    unmatched_ref = max(0, ref_count - matched_pairs)
    unmatched_client = max(0, client_count - matched_pairs)
    r_miss = float(unmatched_ref) / float(max(1, ref_count))
    r_ghost = float(unmatched_client) / float(max(1, client_count))

    return {
        "matched_pairs": matched_pairs,
        "d_box": d_box,
        "d_cls": d_cls,
        "r_miss": r_miss,
        "r_ghost": r_ghost,
        "matched_iou": matched_iou,
    }


def aggregate_client_consistency(
    reference_predictions: Sequence[Dict[str, Any]],
    client_predictions: Sequence[Dict[str, Any]],
    cfg: SPCHMTrustConfig,
) -> Dict[str, float]:
    by_image_ref = {str(item["image_id"]): item.get("detections", []) for item in reference_predictions}
    by_image_client = {str(item["image_id"]): item.get("detections", []) for item in client_predictions}
    image_ids = sorted(set(by_image_ref) | set(by_image_client))

    if not image_ids:
        return {"d_box": 0.0, "d_cls": 0.0, "r_miss": 0.0, "r_ghost": 0.0, "s_i": 0.0}

    per_image = []
    for image_id in image_ids:
        per_image.append(
            score_prediction_consistency(
                reference_detections=by_image_ref.get(image_id, []),
                client_detections=by_image_client.get(image_id, []),
                class_penalty=float(cfg.hungarian_class_penalty),
                    match_iou_threshold=float(cfg.match_iou_threshold),
            )
        )

    d_box = float(np.mean([row["d_box"] for row in per_image])) if per_image else 0.0
    d_cls = float(np.mean([row["d_cls"] for row in per_image])) if per_image else 0.0
    r_miss = float(np.mean([row["r_miss"] for row in per_image])) if per_image else 0.0
    r_ghost = float(np.mean([row["r_ghost"] for row in per_image])) if per_image else 0.0
    s_i = compute_composite_score(d_box=d_box, d_cls=d_cls, r_miss=r_miss, r_ghost=r_ghost, cfg=cfg)

    return {"d_box": d_box, "d_cls": d_cls, "r_miss": r_miss, "r_ghost": r_ghost, "s_i": s_i}


def combine_consistency_metrics(
    base_metrics: Dict[str, float],
    extra_metrics: Optional[Dict[str, float]],
    cfg: SPCHMTrustConfig,
) -> Dict[str, float]:
    if not extra_metrics:
        return dict(base_metrics)

    mode = str(cfg.proxy_trigger_mode).strip().lower()
    if mode == "mean":
        out = {
            "d_box": 0.5 * (float(base_metrics["d_box"]) + float(extra_metrics["d_box"])),
            "d_cls": 0.5 * (float(base_metrics["d_cls"]) + float(extra_metrics["d_cls"])),
            "r_miss": 0.5 * (float(base_metrics["r_miss"]) + float(extra_metrics["r_miss"])),
            "r_ghost": 0.5 * (float(base_metrics["r_ghost"]) + float(extra_metrics["r_ghost"])),
        }
    else:
        out = {
            "d_box": max(float(base_metrics["d_box"]), float(extra_metrics["d_box"])),
            "d_cls": max(float(base_metrics["d_cls"]), float(extra_metrics["d_cls"])),
            "r_miss": max(float(base_metrics["r_miss"]), float(extra_metrics["r_miss"])),
            "r_ghost": max(float(base_metrics["r_ghost"]), float(extra_metrics["r_ghost"])),
        }
    out["s_i"] = compute_composite_score(
        d_box=out["d_box"],
        d_cls=out["d_cls"],
        r_miss=out["r_miss"],
        r_ghost=out["r_ghost"],
        cfg=cfg,
    )
    return out


def compute_composite_score(d_box: float, d_cls: float, r_miss: float, r_ghost: float, cfg: SPCHMTrustConfig) -> float:
    return float(
        float(cfg.lambda_box) * float(d_box)
        + float(cfg.lambda_cls) * float(d_cls)
        + float(cfg.lambda_miss) * float(r_miss)
        + float(cfg.lambda_ghost) * float(r_ghost)
    )


def mad_normalize_scores(scores: Sequence[float], eps: float) -> Dict[str, Any]:
    arr = np.asarray(list(scores), dtype=np.float64)
    if arr.size == 0:
        return {"median": 0.0, "mad": 0.0, "z_scores": np.zeros(0, dtype=np.float64)}

    median_val = float(np.median(arr))
    mad_val = float(np.median(np.abs(arr - median_val)))
    denom = 1.4826 * mad_val + float(eps)
    z_scores = np.maximum(0.0, (arr - median_val) / denom)
    return {"median": median_val, "mad": mad_val, "z_scores": z_scores}


def compute_trust_weights(
    updates: Sequence[ClientUpdate],
    delta_root: Optional[NDArrays],
    z_scores: Sequence[float],
    tau: float,
    eps: float,
    trust_floor: float = 0.0,
) -> Dict[str, Any]:
    z_arr = np.asarray(list(z_scores), dtype=np.float64)
    if len(updates) != len(z_arr):
        raise ValueError("Length mismatch between updates and z_scores")

    cosine_root = []
    trust_raw = []
    weight_raw = []
    for (_, delta, num_examples), z_i in zip(updates, z_arr):
        cosine_val = _cosine_root_similarity(delta, delta_root, eps)
        cosine_root.append(float(cosine_val))
        trust_val = math.exp(-float(tau) * float(z_i)) * float(cosine_val)
        if trust_floor > 0.0:
            trust_val = max(float(trust_floor), float(trust_val))
        trust_raw.append(float(trust_val))
        weight_raw.append(float(num_examples) * float(trust_val))

    weight_raw_arr = np.asarray(weight_raw, dtype=np.float64)
    fallback_used = bool(np.all(weight_raw_arr <= float(eps)))
    if fallback_used:
        num_examples_arr = np.asarray([n for _, _, n in updates], dtype=np.float64)
        total = float(num_examples_arr.sum())
        weights = (num_examples_arr / total) if total > 0.0 else np.full(len(updates), 1.0 / max(1, len(updates)))
    else:
        weights = weight_raw_arr / float(weight_raw_arr.sum())

    return {
        "cosine_root": [float(x) for x in cosine_root],
        "trust_raw": [float(x) for x in trust_raw],
        "weight_raw": [float(x) for x in weight_raw_arr.tolist()],
        "trust_weights": [float(x) for x in weights.tolist()],
        "fallback_used": fallback_used,
    }


def aggregate_delta_with_weights(updates: Sequence[ClientUpdate], weights: Sequence[float]) -> NDArrays:
    if not updates:
        raise ValueError("No updates to aggregate")

    weights_arr = np.asarray(list(weights), dtype=np.float64)
    if len(weights_arr) != len(updates):
        raise ValueError("Length mismatch between updates and weights")

    n_layers = len(updates[0][1])
    aggregated: NDArrays = []
    for layer_idx in range(n_layers):
        first = np.asarray(updates[0][1][layer_idx])
        acc = np.zeros_like(first, dtype=np.float64)
        for weight, (_, delta, _) in zip(weights_arr, updates):
            acc += float(weight) * np.asarray(delta[layer_idx], dtype=np.float64)
        aggregated.append(acc.astype(first.dtype, copy=False))
    return aggregated


def run_spchm_trust_round(
    updates: Sequence[ClientUpdate],
    global_params: NDArrays,
    cfg: SPCHMTrustConfig,
    base_model_path: str,
    server_round: int,
) -> Dict[str, Any]:
    if not updates:
        raise ValueError("No client updates available for SPCHM-Trust aggregation")
    if not cfg.proxy_data_yaml:
        raise ValueError("SPCHM-Trust requires proxy_data_yaml")

    tmp_dir = Path(cfg.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    root_info: Dict[str, Any] = {"delta_root": None, "num_examples": 0, "checkpoint_path": ""}
    delta_root: Optional[NDArrays] = None
    if cfg.root_data_yaml:
        root_info = build_root_delta(
            base_model_path=base_model_path,
            global_params=global_params,
            root_data_yaml=cfg.root_data_yaml,
            epochs=int(cfg.root_epochs),
            imgsz=int(cfg.root_imgsz),
            batch=int(cfg.root_batch),
            num_workers=int(cfg.root_num_workers),
            device=str(cfg.root_device),
            project=str(cfg.train_runs_dir),
            tmp_dir=str(tmp_dir),
            server_round=int(server_round),
            seed=int(cfg.seed),
            train_overrides=dict(cfg.train_overrides or {}),
        )
        delta_root = root_info["delta_root"]

    proxy_images = load_dataset_images(cfg.proxy_data_yaml, split="val", max_images=int(cfg.proxy_max_images))
    proxy_images_triggered = prepare_inference_image_paths(
        image_paths=proxy_images,
        trigger=bool(cfg.proxy_trigger),
        trigger_size=int(cfg.proxy_trigger_size),
        trigger_value=int(cfg.proxy_trigger_value),
        trigger_position=str(cfg.proxy_trigger_position),
        trigger_tmp_dir=str(tmp_dir / f"proxy_triggered_round_{int(server_round):04d}"),
    )
    predictor = ReusableYOLOPredictor(base_model_path=base_model_path)
    diagnostics: List[Dict[str, Any]] = []
    try:
        predictor.load_parameters(global_params)
        reference_predictions = predictor.predict(
            image_paths=proxy_images,
            imgsz=int(cfg.proxy_imgsz),
            device=str(cfg.root_device),
            conf=float(cfg.proxy_conf),
        )
        reference_predictions_triggered = None
        if bool(cfg.proxy_trigger):
            reference_predictions_triggered = predictor.predict(
                image_paths=proxy_images_triggered,
                imgsz=int(cfg.proxy_imgsz),
                device=str(cfg.root_device),
                conf=float(cfg.proxy_conf),
            )

        for cid, delta, num_examples in updates:
            client_params = [np.asarray(g) + np.asarray(d) for g, d in zip(global_params, delta)]
            predictor.load_parameters(client_params)
            client_predictions = predictor.predict(
                image_paths=proxy_images,
                imgsz=int(cfg.proxy_imgsz),
                device=str(cfg.root_device),
                conf=float(cfg.proxy_conf),
            )
            metrics_clean = aggregate_client_consistency(reference_predictions, client_predictions, cfg)
            metrics_triggered = None
            if bool(cfg.proxy_trigger):
                client_predictions_triggered = predictor.predict(
                    image_paths=proxy_images_triggered,
                    imgsz=int(cfg.proxy_imgsz),
                    device=str(cfg.root_device),
                    conf=float(cfg.proxy_conf),
                )
                metrics_triggered = aggregate_client_consistency(
                    reference_predictions_triggered or [],
                    client_predictions_triggered,
                    cfg,
                )
            metrics = combine_consistency_metrics(metrics_clean, metrics_triggered, cfg)
            diagnostics.append(
                {
                    "cid": cid,
                    "num_examples": int(num_examples),
                    "d_box": float(metrics["d_box"]),
                    "d_cls": float(metrics["d_cls"]),
                    "r_miss": float(metrics["r_miss"]),
                    "r_ghost": float(metrics["r_ghost"]),
                    "s_i": float(metrics["s_i"]),
                    "s_clean": float(metrics_clean["s_i"]),
                    "s_trigger": float(metrics_triggered["s_i"]) if metrics_triggered is not None else float(metrics_clean["s_i"]),
                }
            )
    finally:
        predictor.close()

    score_norm = mad_normalize_scores([row["s_i"] for row in diagnostics], eps=float(cfg.eps))
    trust = compute_trust_weights(
        updates=updates,
        delta_root=delta_root,
        z_scores=score_norm["z_scores"].tolist(),
        tau=float(cfg.tau),
        eps=float(cfg.eps),
        trust_floor=float(cfg.trust_floor),
    )
    aggregated_delta = aggregate_delta_with_weights(updates, trust["trust_weights"])

    for idx, row in enumerate(diagnostics):
        row["z_i"] = float(score_norm["z_scores"][idx])
        row["cosine_root"] = float(trust["cosine_root"][idx])
        row["trust_raw"] = float(trust["trust_raw"][idx])
        row["trust_weight"] = float(trust["trust_weights"][idx])
        row["fallback_used"] = bool(trust["fallback_used"])

    return {
        "aggregated_delta": aggregated_delta,
        "client_diagnostics": diagnostics,
        "fallback_used": bool(trust["fallback_used"]),
        "removed_cids": [],
        "reason": "spchm_trust",
        "score_median": float(score_norm["median"]),
        "score_mad": float(score_norm["mad"]),
        "proxy_num_images": int(len(proxy_images)),
        "proxy_trigger": bool(cfg.proxy_trigger),
        "root_num_examples": int(root_info.get("num_examples", 0)),
        "root_checkpoint": str(root_info.get("checkpoint_path", "")),
    }

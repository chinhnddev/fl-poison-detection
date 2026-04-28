from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
from ultralytics import YOLO

from defense.spchm_trust import score_prediction_consistency
from evaluation.device_utils import normalize_ultralytics_device
from train_yolo import _release_torch_memory, load_dataset_images


def _infer_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for idx, part in enumerate(parts):
        if part.lower() == "images":
            parts[idx] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def _load_yolo_labels(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    if not label_path.exists():
        return []
    out: List[Tuple[int, float, float, float, float]] = []
    for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
        toks = line.strip().split()
        if len(toks) < 5:
            continue
        try:
            out.append((int(float(toks[0])), float(toks[1]), float(toks[2]), float(toks[3]), float(toks[4])))
        except Exception:
            continue
    return out


def _xywhn_to_xyxy(xc: float, yc: float, w: float, h: float, img_w: int, img_h: int) -> List[float]:
    x1 = (xc - w / 2.0) * img_w
    y1 = (yc - h / 2.0) * img_h
    x2 = (xc + w / 2.0) * img_w
    y2 = (yc + h / 2.0) * img_h
    return [float(x1), float(y1), float(x2), float(y2)]


def evaluate_perception_metrics(
    model_path: str,
    data_yaml: str,
    imgsz: int,
    device: str,
    conf: float = 0.25,
    max_images: int = 0,
    class_penalty: float = 0.5,
) -> Dict[str, Optional[float]]:
    image_paths = load_dataset_images(data_yaml, split="val", max_images=max_images)
    model = YOLO(model_path)
    resolved_device = normalize_ultralytics_device(device)

    total_gt = 0
    total_pred = 0
    total_matched = 0
    matched_box_error = 0.0
    matched_cls_mismatch = 0.0
    missing_count = 0.0
    ghost_count = 0.0

    try:
        for image_path in image_paths:
            try:
                with Image.open(image_path) as im:
                    width, height = im.size
            except Exception:
                continue

            gt_detections = []
            for cls_id, xc, yc, bw, bh in _load_yolo_labels(_infer_label_path(image_path)):
                gt_detections.append({"cls": int(cls_id), "xyxy": _xywhn_to_xyxy(xc, yc, bw, bh, width, height)})

            res_list = model.predict(
                source=str(image_path),
                imgsz=int(imgsz),
                device=resolved_device,
                conf=float(conf),
                verbose=False,
            )
            pred_detections = []
            if res_list:
                boxes = getattr(res_list[0], "boxes", None)
                if boxes is not None:
                    cls_t = getattr(boxes, "cls", None)
                    xyxy_t = getattr(boxes, "xyxy", None)
                    if cls_t is not None and xyxy_t is not None:
                        try:
                            cls_arr = cls_t.detach().cpu().numpy().astype(int).tolist()
                            xyxy_arr = xyxy_t.detach().cpu().numpy().tolist()
                        except Exception:
                            cls_arr, xyxy_arr = [], []
                        for cls_val, xyxy in zip(cls_arr, xyxy_arr):
                            pred_detections.append({"cls": int(cls_val), "xyxy": [float(v) for v in xyxy[:4]]})

            per_image = score_prediction_consistency(gt_detections, pred_detections, class_penalty=float(class_penalty))
            matched_pairs = int(per_image["matched_pairs"])

            total_gt += len(gt_detections)
            total_pred += len(pred_detections)
            total_matched += matched_pairs
            matched_box_error += float(per_image["d_box"]) * matched_pairs
            matched_cls_mismatch += float(per_image["d_cls"]) * matched_pairs
            missing_count += float(per_image["r_miss"]) * max(1, len(gt_detections))
            ghost_count += float(per_image["r_ghost"]) * max(1, len(pred_detections))
    finally:
        model = None
        _release_torch_memory()

    return {
        "missing_object_rate": (missing_count / max(1, total_gt)) if total_gt > 0 else None,
        "ghost_object_rate": (ghost_count / max(1, total_pred)) if total_pred > 0 else None,
        "class_mismatch_rate": (matched_cls_mismatch / max(1, total_matched)) if total_matched > 0 else None,
        "mean_box_deviation": (matched_box_error / max(1, total_matched)) if total_matched > 0 else None,
        "matched_pairs": int(total_matched),
        "num_images": int(len(image_paths)),
    }

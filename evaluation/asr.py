from __future__ import annotations

from collections import Counter
import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
import os
import numpy as np

_repo_tmp = Path(__file__).resolve().parents[1] / "tmp"
_repo_tmp.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_repo_tmp.resolve()))

from ultralytics import YOLO


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _names_map(cfg: Dict[str, Any]) -> Dict[int, str]:
    names = cfg.get("names", {})
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    if isinstance(names, dict):
        out: Dict[int, str] = {}
        for key, value in names.items():
            try:
                out[int(key)] = str(value)
            except Exception:
                continue
        return out
    return {}


def _infer_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for i, part in enumerate(parts):
        if part.lower() == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def _resolve_ref(cfg: Dict, ref: str, yaml_path: Path) -> Path:
    p = Path(str(ref))
    if p.is_absolute():
        return p
    p1 = (yaml_path.parent / p).resolve()
    if p1.exists():
        return p1
    root = cfg.get("path", "")
    if root:
        return (yaml_path.parent / Path(str(root)) / p).resolve()
    return p1


def _list_images(ref: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if ref.suffix.lower() == ".txt" and ref.exists():
        out = []
        for line in ref.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line:
                p = Path(line)
                if not p.is_absolute():
                    p = (ref.parent / p).resolve()
                out.append(p)
        return out
    if ref.exists() and ref.is_dir():
        return [x for x in ref.rglob("*") if x.suffix.lower() in exts]
    raise FileNotFoundError(f"Unsupported image reference: {ref}")


def _load_yolo_labels(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    if not label_path.exists():
        return []
    out: List[Tuple[int, float, float, float, float]] = []
    for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        toks = line.split()
        if len(toks) < 5:
            continue
        try:
            cls = int(float(toks[0]))
            xc, yc, w, h = (float(toks[1]), float(toks[2]), float(toks[3]), float(toks[4]))
        except Exception:
            continue
        out.append((cls, xc, yc, w, h))
    return out


def inspect_backdoor_asr_pair(
    data_yaml: str,
    src_class_id: int,
    target_class_id: int,
    *,
    min_train_instances: int = 10,
    min_train_images: int = 4,
    top_k_recommendations: int = 5,
) -> Dict[str, Any]:
    """Inspect whether an ASR source/target pair is sensible for the dataset split.

    This is especially useful for small smoke-test datasets such as COCO128 where
    a poor target class can make relaxed ASR look artificially high (natural
    co-occurrence with the source class) or make strict ASR unrealistically hard
    (target class almost absent from train).
    """

    yaml_path = Path(data_yaml)
    if not yaml_path.exists():
        return {
            "available": False,
            "warnings": [f"Dataset YAML not found: {yaml_path}"],
            "recommended_targets": [],
        }

    cfg: Dict[str, Any] = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = _names_map(cfg)
    train_ref = cfg.get("train", "")
    val_ref = cfg.get("val", "")
    if not train_ref or not val_ref:
        return {
            "available": False,
            "warnings": ["Dataset YAML must define both train and val splits for ASR pair analysis."],
            "recommended_targets": [],
        }

    train_images = _list_images(_resolve_ref(cfg, str(train_ref), yaml_path))
    val_images = _list_images(_resolve_ref(cfg, str(val_ref), yaml_path))

    train_obj = Counter()
    train_img = Counter()
    val_obj = Counter()
    val_img = Counter()
    val_co_with_src_img = Counter()
    shape_stats: Dict[int, Dict[str, List[float]]] = {}
    src_train_instances = 0
    src_train_images = 0
    src_val_instances = 0
    src_val_images = 0

    for img in train_images:
        labels = _load_yolo_labels(_infer_label_path(img))
        classes = [int(cls) for cls, *_ in labels]
        train_obj.update(classes)
        train_img.update(set(classes))
        for cls, _xc, _yc, w, h in labels:
            bucket = shape_stats.setdefault(int(cls), {"ratio": [], "area": []})
            bucket["ratio"].append(float(h) / max(float(w), 1e-9))
            bucket["area"].append(float(w) * float(h))
        if int(src_class_id) in classes:
            src_train_instances += sum(1 for cls in classes if cls == int(src_class_id))
            src_train_images += 1

    for img in val_images:
        labels = _load_yolo_labels(_infer_label_path(img))
        classes = [int(cls) for cls, *_ in labels]
        class_set = set(classes)
        val_obj.update(classes)
        val_img.update(class_set)
        if int(src_class_id) in class_set:
            src_val_instances += sum(1 for cls in classes if cls == int(src_class_id))
            src_val_images += 1
            for cls in class_set:
                if cls != int(src_class_id):
                    val_co_with_src_img[int(cls)] += 1

    src_name = names.get(int(src_class_id), str(src_class_id))
    target_name = names.get(int(target_class_id), str(target_class_id))
    src_shapes = shape_stats.get(int(src_class_id), {"ratio": [], "area": []})
    src_mean_ratio = float(np.mean(src_shapes["ratio"])) if src_shapes["ratio"] else None
    src_mean_area = float(np.mean(src_shapes["area"])) if src_shapes["area"] else None

    def _geometry_score(cls_id: int) -> Optional[float]:
        bucket = shape_stats.get(int(cls_id), {"ratio": [], "area": []})
        if not bucket["ratio"] or not bucket["area"] or src_mean_ratio is None or src_mean_area is None:
            return None
        tgt_mean_ratio = float(np.mean(bucket["ratio"]))
        tgt_mean_area = float(np.mean(bucket["area"]))
        return float(
            abs(math.log((tgt_mean_ratio + 1e-9) / (src_mean_ratio + 1e-9)))
            + 0.5 * abs(math.log((tgt_mean_area + 1e-9) / (src_mean_area + 1e-9)))
        )

    target_stats = {
        "class_id": int(target_class_id),
        "name": target_name,
        "train_instances": int(train_obj.get(int(target_class_id), 0)),
        "train_images": int(train_img.get(int(target_class_id), 0)),
        "val_instances": int(val_obj.get(int(target_class_id), 0)),
        "val_images": int(val_img.get(int(target_class_id), 0)),
        "val_images_with_src": int(val_co_with_src_img.get(int(target_class_id), 0)),
        "geometry_score": _geometry_score(int(target_class_id)),
    }

    candidates = []
    for cls in sorted(set(train_obj.keys()) | set(val_obj.keys())):
        if cls == int(src_class_id):
            continue
        candidates.append(
            {
                "class_id": int(cls),
                "name": names.get(int(cls), str(cls)),
                "train_instances": int(train_obj.get(int(cls), 0)),
                "train_images": int(train_img.get(int(cls), 0)),
                "val_instances": int(val_obj.get(int(cls), 0)),
                "val_images": int(val_img.get(int(cls), 0)),
                "val_images_with_src": int(val_co_with_src_img.get(int(cls), 0)),
                "geometry_score": _geometry_score(int(cls)),
            }
        )
    recommended_targets = sorted(
        [
            item
            for item in candidates
            if item["train_instances"] >= int(min_train_instances) and item["train_images"] >= int(min_train_images)
        ],
        key=lambda item: (
            item["val_images_with_src"],
            float(item["geometry_score"] if item["geometry_score"] is not None else 999.0),
            -item["train_instances"],
            -item["train_images"],
            item["class_id"],
        ),
    )[: int(top_k_recommendations)]

    warnings_out: List[str] = []
    if target_stats["val_images_with_src"] > 0:
        warnings_out.append(
            f"Target class {target_stats['class_id']} ({target_name}) co-occurs naturally with "
            f"source class {src_class_id} ({src_name}) in {target_stats['val_images_with_src']} validation image(s); "
            "relaxed ASR can be inflated by natural detections."
        )
    if target_stats["train_instances"] < int(min_train_instances) or target_stats["train_images"] < int(min_train_images):
        warnings_out.append(
            f"Target class {target_stats['class_id']} ({target_name}) is sparse in the train split: "
            f"{target_stats['train_instances']} object(s) across {target_stats['train_images']} image(s); "
            "strict ASR may stay near zero because the target mapping is hard to learn."
        )
    geom_score = target_stats.get("geometry_score")
    if geom_score is not None and geom_score > 1.2:
        warnings_out.append(
            f"Target class {target_stats['class_id']} ({target_name}) has box geometry far from "
            f"source class {src_class_id} ({src_name}) in the train split (geometry_score={geom_score:.3f}); "
            "label-flipping source boxes into this target can be hard to learn on a tiny smoke-test dataset."
        )
    if target_stats["train_instances"] == 0:
        warnings_out.append(
            f"Target class {target_stats['class_id']} ({target_name}) does not appear in the train split at all; "
            "backdoor learning for this target is not feasible."
        )

    return {
        "available": True,
        "src": {
            "class_id": int(src_class_id),
            "name": src_name,
            "train_instances": int(src_train_instances),
            "train_images": int(src_train_images),
            "val_instances": int(src_val_instances),
            "val_images": int(src_val_images),
            "mean_ratio": src_mean_ratio,
            "mean_area": src_mean_area,
        },
        "target": target_stats,
        "warnings": warnings_out,
        "recommended_targets": recommended_targets,
    }


def _xywhn_to_xyxy(xc: float, yc: float, w: float, h: float, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    x1 = (xc - w / 2.0) * img_w
    y1 = (yc - h / 2.0) * img_h
    x2 = (xc + w / 2.0) * img_w
    y2 = (yc + h / 2.0) * img_h
    return x1, y1, x2, y2


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def _apply_trigger_to_temp(img: Path, trigger_size: int, trigger_value: int, position: str, tmp_dir: Path) -> Path:
    from PIL import Image, ImageDraw
    import hashlib

    tmp_dir.mkdir(parents=True, exist_ok=True)
    # Use a path hash to avoid filename collisions when images from different
    # directories share the same basename (e.g. multiple clients with img001.jpg).
    path_hash = hashlib.sha256(str(img).encode("utf-8")).hexdigest()[:8]
    out = tmp_dir / f"{path_hash}_{img.name}"
    with Image.open(img) as im:
        im = im.convert("RGB")
        w, h = im.size
        ts = max(2, min(int(trigger_size), min(w, h)))
        if position == "bottom_left":
            x1, y1 = 0, h - ts
        elif position == "top_right":
            x1, y1 = w - ts, 0
        elif position == "top_left":
            x1, y1 = 0, 0
        else:
            x1, y1 = w - ts, h - ts
        v = int(trigger_value)
        draw = ImageDraw.Draw(im)
        draw.rectangle([x1, y1, x1 + ts - 1, y1 + ts - 1], fill=(v, v, v))
        im.save(out)
    return out


def asr_backdoor_object_level(
    model_path: str,
    data_yaml: str,
    src_class_id: int,
    target_class_id: int,
    imgsz: int,
    device: str,
    conf: float,
    iou_thres: float,
    trigger: bool,
    trigger_size: int,
    trigger_value: int,
    trigger_position: str,
    mode: str = "strict",
    limit_images: int = 0,
    tmp_dir: str = "./tmp/asr_triggered",
) -> Optional[float]:
    """Compute ASR at object-level. If trigger=True, inject trigger patch into val images before prediction."""
    yaml_path = Path(data_yaml)
    if not yaml_path.exists():
        return None

    cfg: Dict = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    val_ref = cfg.get("val", "")
    if not val_ref:
        return None

    val_path = _resolve_ref(cfg, str(val_ref), yaml_path)
    images = _list_images(val_path)
    if limit_images and limit_images > 0:
        images = images[: int(limit_images)]

    from PIL import Image

    model = YOLO(model_path)
    denom = 0
    num = 0
    tdir = Path(tmp_dir)

    try:
        for img in images:
            labels = _load_yolo_labels(_infer_label_path(img))
            if not labels:
                continue
            try:
                with Image.open(img) as im:
                    w, h = im.size
            except Exception:
                continue

            gt_src = []
            for cls, xc, yc, bw, bh in labels:
                if cls == int(src_class_id):
                    gt_src.append(_xywhn_to_xyxy(xc, yc, bw, bh, w, h))
            if not gt_src:
                continue

            denom += len(gt_src)
            img_infer = img
            if trigger:
                img_infer = _apply_trigger_to_temp(img, trigger_size, trigger_value, trigger_position, tdir)

            res_list = model.predict(source=str(img_infer), imgsz=imgsz, device=device, conf=conf, verbose=False)
            if not res_list:
                continue
            res = res_list[0]
            boxes = getattr(res, "boxes", None)
            if boxes is None:
                continue
            cls_t = getattr(boxes, "cls", None)
            xyxy_t = getattr(boxes, "xyxy", None)
            if cls_t is None or xyxy_t is None:
                continue
            try:
                pred_cls = cls_t.detach().cpu().numpy().astype(int).tolist()
                pred_xyxy = xyxy_t.detach().cpu().numpy().tolist()
            except Exception:
                continue
            preds = [(int(c), tuple(map(float, b))) for c, b in zip(pred_cls, pred_xyxy)]
            if not preds:
                continue

            for gt in gt_src:
                best_iou = 0.0
                best_cls = None
                best_src_iou = 0.0
                best_tgt_iou = 0.0
                for c, pr in preds:
                    i = _iou(gt, pr)
                    if i > best_iou:
                        best_iou = i
                        best_cls = c
                    if c == int(src_class_id) and i > best_src_iou:
                        best_src_iou = i
                    if c == int(target_class_id) and i > best_tgt_iou:
                        best_tgt_iou = i

                m = str(mode).strip().lower()
                if m in {"relaxed", "loose"}:
                    if best_tgt_iou >= float(iou_thres):
                        num += 1
                else:
                    if best_iou >= float(iou_thres) and best_cls == int(target_class_id) and best_src_iou < float(iou_thres):
                        num += 1
    finally:
        model = None
        try:
            import gc

            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        except Exception:
            pass

    if denom == 0:
        warnings.warn(
            f"ASR computation found no ground-truth objects for src_class_id={src_class_id} "
            f"in the validation set. Check that --data points to a YAML whose val split "
            f"contains images with that class. Returning None (shown as '-').",
            UserWarning,
            stacklevel=2,
        )
        return None
    return num / denom

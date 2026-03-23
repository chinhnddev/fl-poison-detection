from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
import os

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
                # Relaxed ASR: success if there exists a target-class prediction overlapping the GT src object.
                if best_tgt_iou >= float(iou_thres):
                    num += 1
            else:
                # Strict ASR (paper-style): matched prediction must be target, and no correct-class match exists.
                if best_iou >= float(iou_thres) and best_cls == int(target_class_id) and best_src_iou < float(iou_thres):
                    num += 1

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

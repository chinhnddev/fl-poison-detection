import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ultralytics creates a settings directory on import; keep it repo-local.
_repo_tmp = Path(__file__).resolve().parent / "tmp"
_repo_tmp.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_repo_tmp.resolve()))

import yaml
from ultralytics import YOLO


@dataclass
class EvalResult:
    name: str
    model: str
    map50: Optional[float]
    map5095: Optional[float]
    asr: Optional[float]
    extra: Dict


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
    """Resolve train/val ref against `path:` and the YAML file location."""
    p = Path(str(ref))
    if p.is_absolute():
        return p
    # Prefer resolving relative to YAML file location.
    p1 = (yaml_path.parent / p).resolve()
    if p1.exists():
        return p1
    root = cfg.get("path", "")
    if root:
        p2 = (yaml_path.parent / Path(str(root)) / p).resolve()
        return p2
    return p1


def _list_images(ref: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if ref.suffix.lower() == ".txt" and ref.exists():
        out = []
        for line in ref.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(Path(line))
        return out
    if ref.exists() and ref.is_dir():
        return [x for x in ref.rglob("*") if x.suffix.lower() in exts]
    raise FileNotFoundError(f"Unsupported image reference: {ref}")


def _load_yolo_labels(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    if not label_path.exists():
        return []
    out: List[Tuple[int, float, float, float, float]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
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


def evaluate_map(model_path: str, data: str, imgsz: int, device: str) -> Dict[str, Optional[float]]:
    model = YOLO(model_path)
    r = model.val(data=data, imgsz=imgsz, device=device, verbose=False)
    # Ultralytics returns a results object; these attributes are present for detect.
    map50 = _safe_float(getattr(getattr(r, "box", None), "map50", None))
    map5095 = _safe_float(getattr(getattr(r, "box", None), "map", None))
    return {"map50": map50, "map5095": map5095}


def evaluate_asr_object_level(
    model_path: str,
    data_yaml: str,
    src_class_id: int,
    target_class_id: int,
    imgsz: int,
    device: str,
    conf: float,
    iou_thres: float,
    limit_images: int = 0,
) -> Optional[float]:
    """ASR on validation set at the *object level*.

    Definition used (paper-grade, strict):
    For each GT object of src_class:
    1) Find the best matching prediction (max IoU) among all predictions.
    2) Count success if:
       - best IoU >= iou_thres
       - best-matching prediction class == target_class
       - AND there is NO correct-class prediction with IoU >= iou_thres (src_class)

    ASR = successes / (# GT objects of src_class)
    """
    yaml_path = Path(data_yaml)
    if not yaml_path.exists():
        # dataset aliases (e.g., coco128.yaml) can't be parsed for GT labels here
        return None

    cfg: Dict = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    val_ref = cfg.get("val", "")
    if not val_ref:
        return None

    val_path = _resolve_ref(cfg, str(val_ref), yaml_path)
    images = _list_images(val_path)
    if limit_images and limit_images > 0:
        images = images[: int(limit_images)]

    try:
        from PIL import Image
    except Exception as e:
        raise SystemExit("Pillow is required for ASR computation (image size). Install pillow.") from e

    model = YOLO(model_path)
    denom = 0
    num = 0

    for img in images:
        try:
            with Image.open(img) as im:
                w, h = im.size
        except Exception:
            continue

        labels = _load_yolo_labels(_infer_label_path(img))
        gt_src = []
        for cls, xc, yc, bw, bh in labels:
            if cls == int(src_class_id):
                gt_src.append(_xywhn_to_xyxy(xc, yc, bw, bh, w, h))
        if not gt_src:
            continue

        # Predict on one image.
        res_list = model.predict(source=str(img), imgsz=imgsz, device=device, conf=conf, verbose=False)
        if not res_list:
            denom += len(gt_src)
            continue
        res = res_list[0]
        boxes = getattr(res, "boxes", None)
        if boxes is None:
            denom += len(gt_src)
            continue
        cls_t = getattr(boxes, "cls", None)
        xyxy_t = getattr(boxes, "xyxy", None)
        if cls_t is None or xyxy_t is None:
            denom += len(gt_src)
            continue

        try:
            pred_cls = cls_t.detach().cpu().numpy().astype(int).tolist()
            pred_xyxy = xyxy_t.detach().cpu().numpy().tolist()
        except Exception:
            denom += len(gt_src)
            continue

        denom += len(gt_src)
        preds = [(int(c), tuple(map(float, b))) for c, b in zip(pred_cls, pred_xyxy)]
        if not preds:
            continue

        for gt in gt_src:
            best_iou = 0.0
            best_cls = None
            best_src_iou = 0.0
            for c, pr in preds:
                i = _iou(gt, pr)
                if i > best_iou:
                    best_iou = i
                    best_cls = c
                if c == int(src_class_id) and i > best_src_iou:
                    best_src_iou = i

            if best_iou >= float(iou_thres) and best_cls == int(target_class_id) and best_src_iou < float(iou_thres):
                num += 1

    if denom == 0:
        return None
    return num / denom


def _print_table(rows):
    cols = ["name", "model", "mAP@0.5", "mAP@0.5:0.95", "ASR"]
    data = []
    for r in rows:
        data.append(
            [
                r.name,
                r.model,
                "-" if r.map50 is None else f"{r.map50:.4f}",
                "-" if r.map5095 is None else f"{r.map5095:.4f}",
                "-" if r.asr is None else f"{r.asr:.4f}",
            ]
        )
    widths = [len(c) for c in cols]
    for row in data:
        for i, v in enumerate(row):
            widths[i] = max(widths[i], len(str(v)))
    fmt = "  ".join("{:<" + str(w) + "}" for w in widths)
    print(fmt.format(*cols))
    print(fmt.format(*["-" * w for w in widths]))
    for row in data:
        print(fmt.format(*row))


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate YOLO models (mAP + optional ASR).")
    ap.add_argument("--data", required=True, help="YOLO data YAML or Ultralytics dataset alias (e.g., coco128.yaml)")
    ap.add_argument("--baseline", default="", help="Baseline model path (e.g., yolov8n.pt)")
    ap.add_argument("--attacked", default="", help="Model under attack path")
    ap.add_argument("--defended", default="", help="Model with defense path")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--imgsz", type=int, default=320)
    # Use a low conf threshold so ASR doesn't get artificially zeroed by confidence filtering.
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--asr_src_class_id", type=int, default=-1, help="Source class id (e.g., person=0)")
    ap.add_argument("--asr_target_class_id", type=int, default=-1, help="Target class id (e.g., dog=16)")
    ap.add_argument("--asr_iou", type=float, default=0.5, help="IoU threshold for ASR matching")
    ap.add_argument("--asr_limit_images", type=int, default=0, help="If >0, limit number of val images")
    args = ap.parse_args()

    models = []
    if args.baseline:
        models.append(("baseline", args.baseline))
    if args.attacked:
        models.append(("attacked", args.attacked))
    if args.defended:
        models.append(("defended", args.defended))
    if not models:
        raise SystemExit("No models specified. Provide at least --baseline/--attacked/--defended.")

    rows = []
    for name, path in models:
        m = evaluate_map(path, data=args.data, imgsz=args.imgsz, device=args.device)
        asr = None
        if args.asr_src_class_id >= 0 and args.asr_target_class_id >= 0:
            asr = evaluate_asr_object_level(
                path,
                data_yaml=args.data,
                src_class_id=int(args.asr_src_class_id),
                target_class_id=int(args.asr_target_class_id),
                imgsz=args.imgsz,
                device=args.device,
                conf=float(args.conf),
                iou_thres=float(args.asr_iou),
                limit_images=int(args.asr_limit_images),
            )
        rows.append(EvalResult(name=name, model=path, map50=m["map50"], map5095=m["map5095"], asr=asr, extra={}))

    _print_table(rows)


if __name__ == "__main__":
    main()

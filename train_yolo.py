import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ultralytics creates a settings directory on import. In locked-down environments
# `%APPDATA%` may be non-writable; set a repo-local config root to avoid failures.
_repo_tmp = Path(__file__).resolve().parent / "tmp"
_repo_tmp.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_repo_tmp.resolve()))

import yaml
import torch
import numpy as np
from ultralytics import YOLO

NDArrays = List[np.ndarray]


def _release_torch_memory() -> None:
    try:
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def _clear_yolo_label_caches(data_yaml: str) -> None:
    """Best-effort cache cleanup.

    Ultralytics writes *.cache under labels directories. When re-running experiments with
    modified labels (poisoning) on networked filesystems (e.g. Google Drive), stale caches
    can cause confusing behavior. Clearing is safe and deterministic for our per-client views.
    """
    try:
        yp = Path(data_yaml)
        if not yp.exists():
            return
        cfg = yaml.safe_load(yp.read_text(encoding="utf-8")) or {}
        root = cfg.get("path", "")
        root_p = Path(str(root)) if root else yp.parent
        if not root_p.is_absolute():
            root_p = (yp.parent / root_p).resolve()

        for base in [root_p, yp.parent]:
            lbl = base / "labels"
            if lbl.exists() and lbl.is_dir():
                for c in lbl.rglob("*.cache"):
                    try:
                        c.unlink()
                    except Exception:
                        pass
        # Also clear any top-level dataset cache created by Ultralytics (e.g. repo.cache).
        for c in yp.parent.glob("*.cache"):
            try:
                c.unlink()
            except Exception:
                pass
    except Exception:
        return


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _state_dict_from_model(model: YOLO) -> Dict[str, torch.Tensor]:
    return model.model.state_dict()


def get_parameters(model_path: str) -> NDArrays:
    model = YOLO(model_path)
    try:
        try:
            model.to("cpu")
        except Exception:
            pass
        sd = _state_dict_from_model(model)
        params = [v.detach().cpu().numpy() for _, v in sd.items()]
    finally:
        model = None
        _release_torch_memory()
    return params


def set_parameters_to_model(base_model_path: str, params: NDArrays, out_model_path: str) -> str:
    model = YOLO(base_model_path)
    try:
        try:
            model.to("cpu")
        except Exception:
            pass

        sd = _state_dict_from_model(model)
        keys = list(sd.keys())
        if len(keys) != len(params):
            raise ValueError("Parameter length mismatch")

        new_sd = {}
        for k, arr in zip(keys, params):
            a = np.asarray(arr)
            if a.shape == () and sd[k].numel() == 1 and tuple(sd[k].shape) != ():
                a = a.reshape(tuple(sd[k].shape))
            t = torch.from_numpy(a).to(sd[k].dtype)
            new_sd[k] = t
        model.model.load_state_dict(new_sd, strict=True)

        out = Path(out_model_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(out))
        return str(out)
    finally:
        model = None
        _release_torch_memory()


def count_train_images(data_yaml: str, trainer=None) -> int:
    """Best-effort training image count.

    Ultralytics accepts dataset aliases like 'coco8.yaml' that may not exist as a
    local file before training. Prefer trainer-derived counts when available and
    fall back to parsing the YAML only if it's a readable file.
    """

    # Prefer trainer-derived count (works with dataset aliases and downloads).
    if trainer is not None:
        train_loader = getattr(trainer, "train_loader", None)
        dataset = getattr(train_loader, "dataset", None) if train_loader is not None else None
        if dataset is not None:
            try:
                n = int(len(dataset))
                if n > 0:
                    return n
            except Exception:
                pass
        trainset = getattr(trainer, "trainset", None)
        if trainset is not None:
            try:
                n = int(len(trainset))
                if n > 0:
                    return n
            except Exception:
                pass

    # Fallback: parse dataset YAML if it exists as a file.
    try:
        with open(data_yaml, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        # Minimal fallback for smoke tests; Flower requires num_examples > 0.
        return 1

    train_ref = cfg["train"]
    p = Path(train_ref)
    if p.suffix.lower() == ".txt":
        try:
            n = sum(1 for _ in open(p, "r", encoding="utf-8"))
            return n if n > 0 else 1
        except FileNotFoundError:
            return 1
    if p.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        n = sum(1 for x in p.rglob("*") if x.suffix.lower() in exts)
        return n if n > 0 else 1
    return 1


def train_local(
    model_path: str,
    data_yaml: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    project: str,
    name: str,
    seed: int = 0,
    train_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[NDArrays, int, Dict]:
    model = YOLO(model_path)
    trained = None
    params = None
    n = 0
    metrics: Dict = {}

    extra = dict(train_overrides or {})
    extra = {k: v for k, v in extra.items() if v is not None}

    _clear_yolo_label_caches(data_yaml)

    try:
        model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            seed=int(seed),
            deterministic=True,
            workers=0,
            amp=False,
            val=False,
            plots=False,
            project=project,
            name=name,
            exist_ok=True,
            verbose=False,
            **extra,
        )

        save_dir = None
        trainer = getattr(model, "trainer", None)
        if trainer is not None:
            save_dir = getattr(trainer, "save_dir", None)
        ckpt = Path(save_dir) / "weights" / "last.pt" if save_dir else None

        if ckpt is None or not ckpt.exists():
            candidates = []
            for root in [Path(project), Path("runs"), Path.cwd()]:
                if root.exists():
                    candidates.extend(root.rglob(f"{name}/weights/last.pt"))
            if candidates:
                ckpt = max(candidates, key=lambda p: p.stat().st_mtime)
            else:
                raise FileNotFoundError(f"Could not locate checkpoint last.pt for run name '{name}'")

        trained = YOLO(str(ckpt))
        params = [v.detach().cpu().numpy() for _, v in trained.model.state_dict().items()]
        n = count_train_images(data_yaml, trainer=trainer)
        metrics = {"train_images": n, "_ckpt_path": str(ckpt)}
    finally:
        trained = None
        model = None
        _release_torch_memory()

    return params, n, metrics


def collect_detection_stats(
    model_path: str,
    val_yaml: str,
    imgsz: int,
    device: str,
    max_images: int = 50,
    conf: float = 0.25,
    global_model_path: str = "",
    trigger: bool = False,
    trigger_size: int = 40,
    trigger_value: int = 255,
    trigger_position: str = "bottom_right",
    trigger_tmp_dir: str = "./tmp/det_stats_triggered",
) -> str:
    """Run inference on a capped validation subset and return a compact JSON string.

    The JSON encodes lightweight detection statistics used by the detection-aware
    defense on the server to flag anomalous client updates:

    - ``class_freq``: raw per-class detection counts (only classes with ≥ 1 detection)
    - ``bbox_w_mean``, ``bbox_h_mean``: mean normalized bbox width/height
    - ``bbox_w_std``,  ``bbox_h_std``:  std  normalized bbox width/height
    - ``bbox_xc_mean``, ``bbox_yc_mean``: mean bbox centre coordinates
    - ``total_detections``: total number of detected objects across all images
    - ``num_images``: number of images processed
    - ``mean_iou_vs_global``: mean IoU of this model's predictions vs the global
      model's predictions on the same images (omitted when ``global_model_path``
      is empty or inference on the global model fails)

    Returns an empty string ``""`` on any error so the caller can safely skip stats.
    """
    try:
        import yaml as _yaml

        yp = Path(val_yaml)
        if not yp.exists():
            return ""
        cfg_data = _yaml.safe_load(yp.read_text(encoding="utf-8")) or {}
        val_ref = cfg_data.get("val", "")
        if not val_ref:
            return ""

        # Resolve val path relative to yaml location
        vp = Path(str(val_ref))
        if not vp.is_absolute():
            vp1 = (yp.parent / vp).resolve()
            root = cfg_data.get("path", "")
            if root:
                vp2 = (yp.parent / Path(str(root)) / vp).resolve()
                vp = vp2 if vp2.exists() else (vp1 if vp1.exists() else vp)
            else:
                vp = vp1

        # Collect images
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        if vp.suffix.lower() == ".txt" and vp.exists():
            imgs = [
                Path(l.strip())
                for l in vp.read_text(encoding="utf-8").splitlines()
                if l.strip()
            ]
            # Resolve relative paths
            imgs = [(vp.parent / p).resolve() if not p.is_absolute() else p for p in imgs]
        elif vp.exists() and vp.is_dir():
            imgs = [x for x in vp.rglob("*") if x.suffix.lower() in exts]
        else:
            return ""

        imgs = [p for p in imgs if p.exists()]
        if not imgs:
            return ""

        # Limit to max_images (deterministic: take first N after sort)
        imgs = sorted(imgs)[: int(max_images)]

        def _apply_trigger_to_temp(img: Path, trigger_size: int, trigger_value: int, position: str, tmp_dir: Path) -> Path:
            from PIL import Image, ImageDraw
            import hashlib

            tmp_dir.mkdir(parents=True, exist_ok=True)
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

        imgs_infer = imgs
        if bool(trigger):
            tdir = Path(str(trigger_tmp_dir))
            patched = []
            for p in imgs:
                try:
                    patched.append(_apply_trigger_to_temp(p, trigger_size, trigger_value, str(trigger_position), tdir))
                except Exception:
                    patched.append(p)
            imgs_infer = patched

        model = YOLO(model_path)
        results = None
        gmodel = None
        g_results = None
        try:
            results = model.predict(
                source=[str(p) for p in imgs_infer],
                imgsz=int(imgsz),
                device=str(device),
                conf=float(conf),
                verbose=False,
            )

            class_freq: Dict[str, int] = {}
            bbox_widths: List[float] = []
            bbox_heights: List[float] = []
            bbox_xcs: List[float] = []
            bbox_ycs: List[float] = []
            pred_boxes_per_img: List[List[Tuple]] = []

            for res in (results or []):
                boxes = getattr(res, "boxes", None)
                img_preds: List[Tuple] = []
                if boxes is not None:
                    cls_t = getattr(boxes, "cls", None)
                    xywhn_t = getattr(boxes, "xywhn", None)
                    xyxy_t = getattr(boxes, "xyxy", None)
                    if cls_t is not None and xywhn_t is not None:
                        try:
                            cls_arr = cls_t.detach().cpu().numpy().astype(int).tolist()
                            xywhn_arr = xywhn_t.detach().cpu().numpy().tolist()
                            xyxy_arr = xyxy_t.detach().cpu().numpy().tolist() if xyxy_t is not None else []
                        except Exception:
                            cls_arr, xywhn_arr, xyxy_arr = [], [], []
                        for box_idx, (cls, (xc, yc, w, h)) in enumerate(zip(cls_arr, xywhn_arr)):
                            key = str(int(cls))
                            class_freq[key] = class_freq.get(key, 0) + 1
                            bbox_widths.append(float(w))
                            bbox_heights.append(float(h))
                            bbox_xcs.append(float(xc))
                            bbox_ycs.append(float(yc))
                            if box_idx < len(xyxy_arr):
                                img_preds.append((int(cls), *map(float, xyxy_arr[box_idx])))
                pred_boxes_per_img.append(img_preds)

            def _safe_mean(lst: List[float]) -> float:
                return float(np.mean(lst)) if lst else 0.0

            def _safe_std(lst: List[float]) -> float:
                return float(np.std(lst)) if lst else 0.0

            total_dets = sum(class_freq.values())
            stats: Dict = {
                "class_freq": class_freq,
                "bbox_w_mean": _safe_mean(bbox_widths),
                "bbox_w_std": _safe_std(bbox_widths),
                "bbox_h_mean": _safe_mean(bbox_heights),
                "bbox_h_std": _safe_std(bbox_heights),
                "bbox_xc_mean": _safe_mean(bbox_xcs),
                "bbox_yc_mean": _safe_mean(bbox_ycs),
                "total_detections": int(total_dets),
                "num_images": int(len(imgs_infer)),
                "triggered": bool(trigger),
            }

        # ── Optional: IoU vs global model ────────────────────────────────────
            if global_model_path and Path(global_model_path).exists():
                try:
                    gmodel = YOLO(global_model_path)
                    g_results = gmodel.predict(
                        source=[str(p) for p in imgs_infer],
                        imgsz=int(imgsz),
                        device=str(device),
                        conf=float(conf),
                        verbose=False,
                    )
                    iou_vals: List[float] = []
                    for res_g, client_preds in zip(g_results, pred_boxes_per_img):
                        boxes_g = getattr(res_g, "boxes", None)
                        if boxes_g is None or not client_preds:
                            continue
                        xyxy_g_t = getattr(boxes_g, "xyxy", None)
                        if xyxy_g_t is None:
                            continue
                        try:
                            gboxes = xyxy_g_t.detach().cpu().numpy().tolist()
                        except Exception:
                            continue
                        if not gboxes:
                            continue
                        for _, x1c, y1c, x2c, y2c in client_preds:
                            best_iou = 0.0
                            for gbox in gboxes:
                                x1g, y1g, x2g, y2g = map(float, gbox[:4])
                                ix1 = max(x1c, x1g)
                                iy1 = max(y1c, y1g)
                                ix2 = min(x2c, x2g)
                                iy2 = min(y2c, y2g)
                                iw = max(0.0, ix2 - ix1)
                                ih = max(0.0, iy2 - iy1)
                                inter = iw * ih
                                if inter <= 0.0:
                                    continue
                                area_c = max(0.0, x2c - x1c) * max(0.0, y2c - y1c)
                                area_g = max(0.0, x2g - x1g) * max(0.0, y2g - y1g)
                                denom = area_c + area_g - inter
                                iou_val = float(inter / denom) if denom > 0 else 0.0
                                if iou_val > best_iou:
                                    best_iou = iou_val
                            iou_vals.append(best_iou)
                    if iou_vals:
                        stats["mean_iou_vs_global"] = float(np.mean(iou_vals))
                except Exception:
                    pass

            return json.dumps(stats, separators=(",", ":"))
        finally:
            results = None
            g_results = None
            model = None
            gmodel = None
            _release_torch_memory()
    except Exception:
        return ""

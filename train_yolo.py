import json
import logging
import os
import random
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


def _resolve_dataset_root(cfg: Dict[str, Any], yaml_path: Path) -> Path:
    root = cfg.get("path", "")
    if not root:
        return yaml_path.parent.resolve()
    root_p = Path(str(root))
    if root_p.is_absolute():
        return root_p.resolve()
    direct = (yaml_path.parent / root_p).resolve()
    if direct.exists():
        return direct
    cwd_root = (Path.cwd() / root_p).resolve()
    if cwd_root.exists():
        return cwd_root
    return cwd_root


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
        root_p = _resolve_dataset_root(cfg, yp)

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


def _read_dataset_nc(data_yaml: str) -> Optional[int]:
    try:
        cfg = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    try:
        nc = cfg.get("nc", None)
        return int(nc) if nc is not None else None
    except Exception:
        return None


def resolve_base_model_for_data(base_model_path: str, data_yaml: str, tmp_dir: str = "./tmp") -> str:
    """Return a checkpoint whose detect head matches the dataset class count.

    Ultralytics will automatically reshape detection heads to ``data.nc`` during
    training. In FL we compute deltas against the server-sent global weights, so
    the starting checkpoint must already expose the same tensor shapes that local
    training will produce.
    """
    dataset_nc = _read_dataset_nc(data_yaml)
    if dataset_nc is None or dataset_nc <= 0:
        return str(base_model_path)

    model = YOLO(base_model_path)
    try:
        model_yaml = getattr(model.model, "yaml", {}) or {}
        model_nc = model_yaml.get("nc", None)
        try:
            model_nc = int(model_nc) if model_nc is not None else None
        except Exception:
            model_nc = None
    finally:
        model = None
        _release_torch_memory()

    if model_nc == dataset_nc:
        return str(base_model_path)

    base_path = Path(base_model_path)
    tmp_root = Path(tmp_dir) / "adapted_models"
    tmp_root.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.md5(f"{base_path.resolve()}::{dataset_nc}".encode("utf-8")).hexdigest()[:10]
    adapted_yaml = tmp_root / f"{base_path.stem}_nc{dataset_nc}_{cache_key}.yaml"
    adapted_ckpt = tmp_root / f"{base_path.stem}_nc{dataset_nc}_{cache_key}.pt"

    if adapted_ckpt.exists():
        logging.getLogger("train_yolo").info(
            "Using cached dataset-matched base model %s for nc=%s", adapted_ckpt.resolve(), dataset_nc
        )
        return str(adapted_ckpt.resolve())

    base = YOLO(base_model_path)
    try:
        model_yaml = dict(getattr(base.model, "yaml", {}) or {})
        if not model_yaml:
            raise ValueError(f"Could not inspect model YAML for {base_model_path}")
        model_yaml["nc"] = int(dataset_nc)
        adapted_yaml.write_text(yaml.safe_dump(model_yaml, sort_keys=False), encoding="utf-8")
    finally:
        base = None
        _release_torch_memory()

    adapted = YOLO(str(adapted_yaml))
    try:
        adapted.load(base_model_path)
        adapted.save(str(adapted_ckpt))
    finally:
        adapted = None
        _release_torch_memory()

    logging.getLogger("train_yolo").info(
        "Adapted base model %s -> %s to match dataset nc=%s",
        Path(base_model_path).resolve(),
        adapted_ckpt.resolve(),
        dataset_nc,
    )
    return str(adapted_ckpt.resolve())


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


def _load_parameters_into_yolo(model: YOLO, params: NDArrays) -> None:
    sd = _state_dict_from_model(model)
    keys = list(sd.keys())
    if len(keys) != len(params):
        raise ValueError("Parameter length mismatch")

    new_sd = {}
    for k, arr in zip(keys, params):
        a = np.asarray(arr)
        if a.shape == () and sd[k].numel() == 1 and tuple(sd[k].shape) != ():
            a = a.reshape(tuple(sd[k].shape))
        new_sd[k] = torch.from_numpy(a).to(sd[k].dtype)
    model.model.load_state_dict(new_sd, strict=True)


def _resolve_split_reference(data_yaml: str, split: str) -> Tuple[Dict[str, Any], Path]:
    yaml_path = Path(data_yaml)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    split_ref = cfg.get(split, "")
    if not split_ref:
        raise ValueError(f"Dataset YAML '{data_yaml}' does not define split '{split}'")
    p = Path(str(split_ref))
    if p.is_absolute():
        return cfg, p

    direct = (yaml_path.parent / p).resolve()
    if direct.exists():
        return cfg, direct

    root_p = _resolve_dataset_root(cfg, yaml_path)
    if root_p:
        rooted = (root_p / p).resolve()
        if rooted.exists():
            return cfg, rooted
    return cfg, direct


def load_dataset_images(data_yaml: str, split: str = "val", max_images: int = 0) -> List[Path]:
    _, ref_path = _resolve_split_reference(data_yaml, split)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if ref_path.suffix.lower() == ".txt" and ref_path.exists():
        images = []
        for line in ref_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            p = Path(line)
            if not p.is_absolute():
                p = (ref_path.parent / p).resolve()
            if p.exists():
                images.append(p)
    elif ref_path.exists() and ref_path.is_dir():
        images = [x for x in ref_path.rglob("*") if x.suffix.lower() in exts and x.exists()]
    else:
        raise FileNotFoundError(f"Could not resolve dataset split '{split}' from {data_yaml}")

    images = sorted(images)
    if max_images and max_images > 0:
        images = images[: int(max_images)]
    return images


def _extract_prediction_rows(results, image_paths: Sequence[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for image_path, res in zip(image_paths, results or []):
        detections: List[Dict[str, Any]] = []
        boxes = getattr(res, "boxes", None)
        if boxes is not None:
            cls_t = getattr(boxes, "cls", None)
            conf_t = getattr(boxes, "conf", None)
            xyxy_t = getattr(boxes, "xyxy", None)
            if cls_t is not None and xyxy_t is not None:
                try:
                    cls_arr = cls_t.detach().cpu().numpy().astype(int).tolist()
                    conf_arr = conf_t.detach().cpu().numpy().tolist() if conf_t is not None else [1.0] * len(cls_arr)
                    xyxy_arr = xyxy_t.detach().cpu().numpy().tolist()
                except Exception:
                    cls_arr, conf_arr, xyxy_arr = [], [], []
                for cls_val, conf_val, xyxy in zip(cls_arr, conf_arr, xyxy_arr):
                    detections.append(
                        {
                            "cls": int(cls_val),
                            "conf": float(conf_val),
                            "xyxy": [float(v) for v in xyxy[:4]],
                        }
                    )
        detections.sort(key=lambda item: (-item["conf"], item["cls"], item["xyxy"]))
        rows.append({"image_id": str(image_path), "detections": detections})
    return rows


def apply_trigger_to_temp_image(
    img: Path,
    trigger_size: int,
    trigger_value: int,
    position: str,
    tmp_dir: Path,
) -> Path:
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


def prepare_inference_image_paths(
    image_paths: Sequence[Path],
    trigger: bool = False,
    trigger_size: int = 40,
    trigger_value: int = 255,
    trigger_position: str = "bottom_right",
    trigger_tmp_dir: str = "./tmp/triggered_proxy",
) -> List[Path]:
    if not trigger:
        return list(image_paths)

    out: List[Path] = []
    tmp_dir = Path(str(trigger_tmp_dir))
    for img in image_paths:
        try:
            out.append(
                apply_trigger_to_temp_image(
                    img=img,
                    trigger_size=int(trigger_size),
                    trigger_value=int(trigger_value),
                    position=str(trigger_position),
                    tmp_dir=tmp_dir,
                )
            )
        except Exception:
            out.append(img)
    return out


class ReusableYOLOPredictor:
    """Reuse a single YOLO object while iteratively loading parameter arrays."""

    def __init__(self, base_model_path: str):
        self.base_model_path = str(base_model_path)
        self.model = None
        self._reset_model()

    def _reset_model(self) -> None:
        if self.model is not None:
            self.model = None
            _release_torch_memory()
        self.model = YOLO(self.base_model_path)
        try:
            self.model.to("cpu")
        except Exception:
            pass

    def load_parameters(self, params: NDArrays) -> None:
        try:
            _load_parameters_into_yolo(self.model, params)
        except ValueError as exc:
            # Ultralytics may fuse the model graph during predict(), which changes
            # the exposed state_dict layout. Rebuild a fresh unfused model and retry.
            if "Parameter length mismatch" not in str(exc):
                raise
            self._reset_model()
            _load_parameters_into_yolo(self.model, params)

    def predict(self, image_paths: Sequence[Path], imgsz: int, device: str, conf: float) -> List[Dict[str, Any]]:
        results = self.model.predict(
            source=[str(p) for p in image_paths],
            imgsz=int(imgsz),
            device=str(device),
            conf=float(conf),
            verbose=False,
        )
        try:
            return _extract_prediction_rows(results, image_paths)
        finally:
            results = None
            _release_torch_memory()

    def close(self) -> None:
        self.model = None
        _release_torch_memory()


def build_root_delta(
    base_model_path: str,
    global_params: NDArrays,
    root_data_yaml: str,
    epochs: int,
    imgsz: int,
    batch: int,
    num_workers: int,
    device: str,
    project: str,
    tmp_dir: str,
    server_round: int,
    seed: int = 0,
    train_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    tmp_root = Path(tmp_dir)
    tmp_root.mkdir(parents=True, exist_ok=True)

    round_base_ckpt = tmp_root / f"root_start_round_{int(server_round):04d}.pt"
    set_parameters_to_model(base_model_path, global_params, str(round_base_ckpt))

    root_params, num_examples, metrics = train_local(
        model_path=str(round_base_ckpt),
        data_yaml=root_data_yaml,
        epochs=int(epochs),
        imgsz=int(imgsz),
        batch=int(batch),
        num_workers=int(num_workers),
        device=str(device),
        project=str(project),
        name=f"root_round_{int(server_round):04d}",
        seed=int(seed) + 10000 * max(int(server_round), 0),
        train_overrides=train_overrides,
    )
    delta_root = [np.asarray(rp) - np.asarray(gp) for rp, gp in zip(root_params, global_params)]
    trained_ckpt = str(dict(metrics).get("_ckpt_path") or round_base_ckpt)
    return {
        "delta_root": delta_root,
        "num_examples": int(num_examples),
        "metrics": dict(metrics),
        "checkpoint_path": trained_ckpt,
        "base_checkpoint_path": str(round_base_ckpt),
    }


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

    _, p = _resolve_split_reference(data_yaml, "train")
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
    num_workers: int,
    device: str,
    project: str,
    name: str,
    seed: int = 0,
    train_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[NDArrays, int, Dict]:
    from ultralytics.engine.trainer import BaseTrainer

    model = YOLO(model_path)
    trained = None
    params = None
    n = 0
    metrics: Dict = {}
    orig_final_eval = BaseTrainer.final_eval
    orig_validate = BaseTrainer.validate

    extra = dict(train_overrides or {})
    extra = {k: v for k, v in extra.items() if v is not None}

    _clear_yolo_label_caches(data_yaml)

    try:
        # Ultralytics still validates on the final epoch via BaseTrainer._do_train()
        # and then runs BaseTrainer.final_eval() after training, even when val=False.
        # For FL clients we only need the updated weights, so skip both validation paths.
        def _skip_validate(self):
            return getattr(self, "metrics", {}), 0.0

        BaseTrainer.validate = _skip_validate
        BaseTrainer.final_eval = lambda self: None
        model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            seed=int(seed),
            deterministic=True,
            workers=max(0, int(num_workers)),
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
        BaseTrainer.validate = orig_validate
        BaseTrainer.final_eval = orig_final_eval
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
        try:
            imgs = load_dataset_images(val_yaml, split="val", max_images=int(max_images))
        except Exception:
            return ""
        if not imgs:
            return ""
        imgs_infer = prepare_inference_image_paths(
            image_paths=imgs,
            trigger=bool(trigger),
            trigger_size=int(trigger_size),
            trigger_value=int(trigger_value),
            trigger_position=str(trigger_position),
            trigger_tmp_dir=str(trigger_tmp_dir),
        )

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

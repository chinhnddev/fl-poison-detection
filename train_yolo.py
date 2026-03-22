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
    sd = _state_dict_from_model(model)
    return [v.detach().cpu().numpy() for _, v in sd.items()]


def set_parameters_to_model(base_model_path: str, params: NDArrays, out_model_path: str) -> str:
    model = YOLO(base_model_path)
    sd = _state_dict_from_model(model)
    keys = list(sd.keys())
    if len(keys) != len(params):
        raise ValueError("Parameter length mismatch")

    new_sd = {}
    for k, arr in zip(keys, params):
        a = np.asarray(arr)
        # Flower/NumPy may return scalars for 1-element tensors; reshape to match.
        if a.shape == () and sd[k].numel() == 1 and tuple(sd[k].shape) != ():
            a = a.reshape(tuple(sd[k].shape))
        t = torch.from_numpy(a).to(sd[k].dtype)
        new_sd[k] = t
    model.model.load_state_dict(new_sd, strict=True)

    out = Path(out_model_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    return str(out)


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

    extra = dict(train_overrides or {})
    extra = {k: v for k, v in extra.items() if v is not None}

    _clear_yolo_label_caches(data_yaml)

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

    # Use trainer's save_dir when available (Ultralytics may write under runs/detect/...).
    save_dir = None
    trainer = getattr(model, "trainer", None)
    if trainer is not None:
        save_dir = getattr(trainer, "save_dir", None)
    ckpt = Path(save_dir) / "weights" / "last.pt" if save_dir else None

    # Fallback: search for the most recent checkpoint for this run name.
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
    return params, n, {"train_images": n}

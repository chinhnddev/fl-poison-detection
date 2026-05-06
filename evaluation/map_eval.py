from __future__ import annotations

from typing import Dict, Optional

import os
from pathlib import Path

_repo_tmp = Path(__file__).resolve().parents[1] / "tmp"
_repo_tmp.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_repo_tmp.resolve()))

from ultralytics import YOLO

from .device_utils import normalize_ultralytics_device


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def evaluate_map(model_path: str, data: str, imgsz: int, device: str, conf: float = 0.001) -> Dict[str, Optional[float]]:
    model = YOLO(model_path)
    r = None
    try:
        r = model.val(
            data=data,
            imgsz=imgsz,
            device=normalize_ultralytics_device(device),
            conf=float(conf),
            verbose=False,
        )
        map50 = _safe_float(getattr(getattr(r, "box", None), "map50", None))
        map5095 = _safe_float(getattr(getattr(r, "box", None), "map", None))
        return {"map50": map50, "map5095": map5095}
    finally:
        r = None
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

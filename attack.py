from __future__ import annotations

import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

NDArrays = List[np.ndarray]


@dataclass
class LabelFlipConfig:
    enabled: bool = False
    src_class_id: int = 0
    dst_class_id: int = 16
    prob: float = 1.0
    seed: int = 42


@dataclass
class ModelPoisonConfig:
    enabled: bool = False
    strength: float = 1.0
    mode: str = "signflip"  # "signflip" or "scale"


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
    # Prefer YAML-relative first
    p1 = (yaml_path.parent / p).resolve()
    if p1.exists():
        return p1
    root = cfg.get("path", "")
    if root:
        return (yaml_path.parent / Path(str(root)) / p).resolve()
    return p1


def _read_image_list_from_data_yaml(data_yaml: str) -> List[Path]:
    yp = Path(data_yaml)
    cfg: Dict = yaml.safe_load(open(yp, "r", encoding="utf-8"))
    train_ref = cfg["train"]
    p = _resolve_ref(cfg, str(train_ref), yp)
    if p.suffix.lower() == ".txt" and p.exists():
        out = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(Path(line))
        return out
    if p.exists() and p.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        return [x for x in p.rglob("*") if x.suffix.lower() in exts]
    raise FileNotFoundError(f"Unsupported train reference in data.yaml: {train_ref}")


def _copy_or_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        # Prefer symlink (fast, small). Works on Colab/Linux.
        dst.symlink_to(src)
    except Exception:
        try:
            os.link(src, dst)
        except Exception:
            shutil.copy2(src, dst)


def _flip_label_file(in_path: Path, out_path: Path, src_id: int, dst_id: int, prob: float, rng: random.Random) -> Tuple[int, int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not in_path.exists():
        out_path.write_text("", encoding="utf-8")
        return 0, 0
    lines_in = 0
    flipped = 0
    out_lines: List[str] = []
    for line in in_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        lines_in += 1
        toks = line.split()
        if len(toks) < 5:
            out_lines.append(line)
            continue
        try:
            cls = int(float(toks[0]))
        except Exception:
            out_lines.append(line)
            continue
        if cls == src_id and rng.random() <= prob:
            toks[0] = str(int(dst_id))
            flipped += 1
            out_lines.append(" ".join(toks))
        else:
            out_lines.append(line)
    out_path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
    return lines_in, flipped


def build_poisoned_shard_label_flip(
    shard_data_yaml: str,
    out_root: str,
    cfg: LabelFlipConfig,
) -> str:
    """Create a poisoned *view* of a shard: symlink/copy images, flip labels, write new data.yaml.

    Ultralytics infers label paths from image paths, so we create a shadow images tree
    under out_root/images/train/ and write labels under out_root/labels/train/.
    """
    if not cfg.enabled:
        return shard_data_yaml
    if not (0.0 <= cfg.prob <= 1.0):
        raise ValueError("label_flip.prob must be within [0, 1]")

    images = _read_image_list_from_data_yaml(shard_data_yaml)
    shard_yaml_path = Path(shard_data_yaml)
    shard_cfg: Dict = yaml.safe_load(open(shard_yaml_path, "r", encoding="utf-8"))
    val_ref = shard_cfg.get("val", "")
    nc = shard_cfg.get("nc", None)
    names = shard_cfg.get("names", None)

    out_root_p = Path(out_root)
    out_images = out_root_p / "images" / "train"
    out_labels = out_root_p / "labels" / "train"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    rng = random.Random(int(cfg.seed))
    train_txt = out_root_p / "train.txt"
    total_lines = 0
    total_flipped = 0
    with open(train_txt, "w", encoding="utf-8") as f:
        for img in images:
            dst_img = out_images / img.name
            _copy_or_link(img, dst_img)

            src_lbl = _infer_label_path(img)
            dst_lbl = out_labels / (dst_img.with_suffix(".txt").name)
            li, lf = _flip_label_file(src_lbl, dst_lbl, int(cfg.src_class_id), int(cfg.dst_class_id), float(cfg.prob), rng)
            total_lines += li
            total_flipped += lf

            f.write(str(dst_img.resolve()) + "\n")

    out_yaml = {
        "path": "",
        "train": str(train_txt.resolve()),
        "val": val_ref,
    }
    if nc is not None:
        out_yaml["nc"] = nc
    if names is not None:
        out_yaml["names"] = names

    out_yaml_path = out_root_p / "data.yaml"
    with open(out_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(out_yaml, f, sort_keys=False)

    meta = {
        "attack": "label_flip",
        "src_class_id": int(cfg.src_class_id),
        "dst_class_id": int(cfg.dst_class_id),
        "prob": float(cfg.prob),
        "seed": int(cfg.seed),
        "train_images": len(images),
        "label_lines": int(total_lines),
        "label_lines_flipped": int(total_flipped),
    }
    with open(out_root_p / "poison_meta.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False)

    return str(out_yaml_path.resolve())


def poison_model_update(
    global_params: NDArrays,
    trained_params: NDArrays,
    cfg: ModelPoisonConfig,
) -> NDArrays:
    """Poison a client update in parameter space.

    Uses delta = trained - global, then returns global + poisoned_delta.
    - mode=signflip: poisoned_delta = -strength * delta
    - mode=scale:    poisoned_delta =  strength * delta
    """
    if not cfg.enabled:
        return trained_params
    strength = float(cfg.strength)
    if strength <= 0:
        return trained_params

    if len(global_params) != len(trained_params):
        raise ValueError("Parameter length mismatch for model poisoning")

    out: NDArrays = []
    for g, t in zip(global_params, trained_params):
        ga = np.asarray(g)
        ta = np.asarray(t)
        delta = (ta.astype(np.float32, copy=False) - ga.astype(np.float32, copy=False))
        if cfg.mode.lower() == "signflip":
            pd = -strength * delta
        else:
            pd = strength * delta
        poisoned = ga.astype(np.float32, copy=False) + pd
        out.append(np.asarray(poisoned.astype(ta.dtype, copy=False)))
    return out


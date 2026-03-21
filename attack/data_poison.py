from __future__ import annotations

import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

from .config import BackdoorConfig, BBoxDistortionConfig, LabelFlipConfig, ObjectRemovalConfig


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


def _read_image_list_from_data_yaml(data_yaml: str) -> List[Path]:
    yp = Path(data_yaml)
    cfg: Dict = yaml.safe_load(open(yp, "r", encoding="utf-8"))
    train_ref = cfg["train"]
    p = _resolve_ref(cfg, str(train_ref), yp)
    if p.suffix.lower() == ".txt" and p.exists():
        out = []
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
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
        dst.symlink_to(src)
    except Exception:
        try:
            os.link(src, dst)
        except Exception:
            shutil.copy2(src, dst)


def _apply_trigger(in_img: Path, out_img: Path, cfg: BackdoorConfig) -> None:
    """Write an image with a small patch trigger."""
    try:
        from PIL import Image, ImageDraw
    except Exception as e:
        raise RuntimeError("Backdoor attack requires pillow") from e

    out_img.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(in_img) as im:
        im = im.convert("RGB")
        w, h = im.size
        ts = int(cfg.trigger_size)
        ts = max(2, min(ts, min(w, h)))
        if cfg.position == "bottom_left":
            x1, y1 = 0, h - ts
        elif cfg.position == "top_right":
            x1, y1 = w - ts, 0
        elif cfg.position == "top_left":
            x1, y1 = 0, 0
        else:
            x1, y1 = w - ts, h - ts

        draw = ImageDraw.Draw(im)
        v = int(cfg.trigger_value)
        draw.rectangle([x1, y1, x1 + ts - 1, y1 + ts - 1], fill=(v, v, v))
        im.save(out_img)


def _process_label_file(
    in_path: Path,
    out_path: Path,
    image_poisoned: bool,
    rng_flip: random.Random,
    rng_bbox: random.Random,
    rng_rm: random.Random,
    rng_bd: random.Random,
    label_flip: LabelFlipConfig,
    bbox: BBoxDistortionConfig,
    removal: ObjectRemovalConfig,
    backdoor: BackdoorConfig,
) -> Dict[str, int]:
    """Transform a single YOLO label file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not in_path.exists():
        out_path.write_text("", encoding="utf-8")
        return {"lines_in": 0, "lines_out": 0, "flipped": 0, "removed": 0, "distorted": 0, "backdoor_flipped": 0}

    lines_in = 0
    lines_out = 0
    flipped = 0
    removed = 0
    distorted = 0
    backdoor_flipped = 0

    out_lines: List[str] = []
    for line in in_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        toks = line.split()
        if len(toks) < 5:
            continue
        lines_in += 1
        try:
            cls = int(float(toks[0]))
            x, y, w, h = (float(toks[1]), float(toks[2]), float(toks[3]), float(toks[4]))
        except Exception:
            continue

        # Apply object removal (only on poisoned images)
        if image_poisoned and removal.enabled and cls == int(removal.target_class_id) and rng_rm.random() < float(removal.prob):
            removed += 1
            continue

        # Backdoor conditional flip (only if trigger applied on this image)
        if image_poisoned and backdoor.enabled and cls == int(backdoor.src_class_id) and rng_bd.random() < float(backdoor.prob):
            cls = int(backdoor.target_class_id)
            backdoor_flipped += 1

        # Label flip (only on poisoned images)
        if image_poisoned and label_flip.enabled and cls == int(label_flip.src_class_id) and rng_flip.random() < float(label_flip.prob):
            cls = int(label_flip.dst_class_id)
            flipped += 1

        # BBox distortion (only on poisoned images)
        if image_poisoned and bbox.enabled and rng_bbox.random() < float(bbox.prob):
            x += rng_bbox.uniform(-float(bbox.shift_xy), float(bbox.shift_xy))
            y += rng_bbox.uniform(-float(bbox.shift_xy), float(bbox.shift_xy))
            w *= (1.0 + rng_bbox.uniform(-float(bbox.shift_wh), float(bbox.shift_wh)))
            h *= (1.0 + rng_bbox.uniform(-float(bbox.shift_wh), float(bbox.shift_wh)))
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            w = max(0.01, min(1.0, w))
            h = max(0.01, min(1.0, h))
            distorted += 1

        out_lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        lines_out += 1

    out_path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
    return {
        "lines_in": lines_in,
        "lines_out": lines_out,
        "flipped": flipped,
        "removed": removed,
        "distorted": distorted,
        "backdoor_flipped": backdoor_flipped,
    }


def build_poisoned_dataset(
    shard_data_yaml: str,
    out_root: str,
    label_flip: LabelFlipConfig,
    bbox: BBoxDistortionConfig,
    removal: ObjectRemovalConfig,
    backdoor: BackdoorConfig,
) -> str:
    """Create a poisoned *view* of a shard (images+labels) and a compatible data.yaml.

    Important: we keep the original shard `path` and `val` so Ultralytics dataset checks pass.
    """
    images = _read_image_list_from_data_yaml(shard_data_yaml)
    shard_yaml_path = Path(shard_data_yaml)
    shard_cfg: Dict = yaml.safe_load(open(shard_yaml_path, "r", encoding="utf-8"))

    val_ref = shard_cfg.get("val", "")
    base_path = shard_cfg.get("path", "") or str(shard_yaml_path.parent.resolve())
    nc = shard_cfg.get("nc", None)
    names = shard_cfg.get("names", None)

    out_root_p = Path(out_root)
    out_images = out_root_p / "images" / "train"
    out_labels = out_root_p / "labels" / "train"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    # Determine which images are poisoned per attack type (deterministic w.r.t seed and image order).
    rng_flip_mask = random.Random(int(label_flip.seed))
    rng_bbox_mask = random.Random(int(bbox.seed))
    rng_rm_mask = random.Random(int(removal.seed))
    rng_bd_mask = random.Random(int(backdoor.seed))

    # Separate RNG streams for per-object transforms (avoid coupling with mask sampling).
    rng_flip = random.Random(int(label_flip.seed) + 100000)
    rng_bbox = random.Random(int(bbox.seed) + 100000)
    rng_rm = random.Random(int(removal.seed) + 100000)
    rng_bd = random.Random(int(backdoor.seed) + 100000)

    def _mask(r: random.Random, ratio: float) -> List[bool]:
        rr = max(0.0, min(1.0, float(ratio)))
        return [r.random() < rr for _ in images]

    mask_flip = _mask(rng_flip_mask, float(label_flip.poison_ratio)) if label_flip.enabled else [False] * len(images)
    mask_bbox = _mask(rng_bbox_mask, float(bbox.poison_ratio)) if bbox.enabled else [False] * len(images)
    mask_rm = _mask(rng_rm_mask, float(removal.poison_ratio)) if removal.enabled else [False] * len(images)
    mask_bd = _mask(rng_bd_mask, float(backdoor.poison_ratio)) if backdoor.enabled else [False] * len(images)

    train_txt = out_root_p / "train.txt"
    total = {"lines_in": 0, "lines_out": 0, "flipped": 0, "removed": 0, "distorted": 0, "backdoor_flipped": 0}
    poisoned_images = 0
    poisoned_images_backdoor = 0

    with open(train_txt, "w", encoding="utf-8") as f:
        for img, do_flip, do_bbox, do_rm, do_bd in zip(images, mask_flip, mask_bbox, mask_rm, mask_bd):
            is_poisoned = bool(do_flip or do_bbox or do_rm or do_bd)
            dst_img = out_images / img.name
            if do_bd and backdoor.enabled:
                _apply_trigger(img, dst_img, backdoor)
                poisoned_images_backdoor += 1
            else:
                _copy_or_link(img, dst_img)

            src_lbl = _infer_label_path(img)
            dst_lbl = out_labels / (dst_img.with_suffix(".txt").name)
            stats = _process_label_file(
                src_lbl,
                dst_lbl,
                image_poisoned=bool(is_poisoned),
                rng_flip=rng_flip,
                rng_bbox=rng_bbox,
                rng_rm=rng_rm,
                rng_bd=rng_bd,
                label_flip=label_flip,
                bbox=bbox,
                removal=removal,
                backdoor=backdoor,
            )
            for k in total:
                total[k] += int(stats.get(k, 0))
            poisoned_images += int(bool(is_poisoned))

            f.write(str(dst_img.resolve()) + "\n")

    out_yaml = {"path": str(base_path), "train": str(train_txt.resolve()), "val": val_ref}
    if nc is not None:
        out_yaml["nc"] = nc
    if names is not None:
        out_yaml["names"] = names

    out_yaml_path = out_root_p / "data.yaml"
    with open(out_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(out_yaml, f, sort_keys=False)

    meta = {
        "attack": "dataset_poison",
        "train_images": int(len(images)),
        "poisoned_images": int(poisoned_images),
        "poisoned_images_backdoor": int(poisoned_images_backdoor),
        "label_flip": vars(label_flip),
        "bbox_distortion": vars(bbox),
        "object_removal": vars(removal),
        "backdoor": vars(backdoor),
        **total,
    }
    with open(out_root_p / "poison_meta.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False)

    return str(out_yaml_path.resolve())

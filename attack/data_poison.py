from __future__ import annotations

import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
        root_p = Path(str(root))
        p2 = (yaml_path.parent / root_p / p).resolve()
        if p2.exists():
            return p2
        p3 = (Path.cwd() / root_p / p).resolve()
        if p3.exists():
            return p3
        return p2
    return p1


def _read_image_list_from_data_yaml(data_yaml: str) -> List[Path]:
    yp = Path(data_yaml)
    with open(yp, "r", encoding="utf-8") as f:
        cfg: Dict = yaml.safe_load(f)
    train_ref = cfg["train"]
    ref_path = _resolve_ref(cfg, str(train_ref), yp)
    if ref_path.suffix.lower() == ".txt" and ref_path.exists():
        out = []
        for line in ref_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line:
                img_path = Path(line)
                if not img_path.is_absolute():
                    img_path = (ref_path.parent / img_path).resolve()
                out.append(img_path)
        return out
    if ref_path.exists() and ref_path.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        return [x for x in ref_path.rglob("*") if x.suffix.lower() in exts]
    raise FileNotFoundError(f"Unsupported train reference in data.yaml: {train_ref}")


def _portable_ref(target: Path, start: Path) -> str:
    try:
        return Path(os.path.relpath(target.resolve(), start=start.resolve())).as_posix()
    except Exception:
        return str(target.resolve())


def _copy_or_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    src_abs = src.resolve()
    try:
        dst.symlink_to(src_abs)
    except Exception:
        try:
            os.link(src_abs, dst)
        except Exception:
            shutil.copy2(src_abs, dst)


def _apply_trigger(in_img: Path, out_img: Path, cfg: BackdoorConfig) -> Tuple[float, float, float, float]:
    """Write an image with a small patch trigger.

    Returns the trigger patch location as a YOLO-normalized bounding box
    ``(xc, yc, bw, bh)`` so callers can add a synthetic label annotation at
    that position, creating a direct spatial association between the trigger
    and the target class.
    """
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

    # Return trigger bbox in YOLO normalized coordinates.
    xc = (x1 + ts / 2.0) / w
    yc = (y1 + ts / 2.0) / h
    bw = ts / float(w)
    bh = ts / float(h)
    return float(xc), float(yc), float(bw), float(bh)


def _process_label_file(
    in_path: Path,
    out_path: Path,
    do_label_flip: bool,
    do_bbox: bool,
    do_removal: bool,
    do_backdoor: bool,
    rng_flip: random.Random,
    rng_bbox: random.Random,
    rng_rm: random.Random,
    rng_bd: random.Random,
    label_flip: LabelFlipConfig,
    bbox: BBoxDistortionConfig,
    removal: ObjectRemovalConfig,
    backdoor: BackdoorConfig,
    extra_lines: Optional[List[str]] = None,
) -> Dict[str, int]:
    """Transform a single YOLO label file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not in_path.exists():
        # Even for missing label files, we still need to write extra_lines (e.g.
        # the synthetic trigger annotation for backdoor-poisoned images).
        content = "\n".join(extra_lines) + "\n" if extra_lines else ""
        out_path.write_text(content, encoding="utf-8")
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

        # Apply object removal
        if do_removal and removal.enabled and cls == int(removal.target_class_id) and rng_rm.random() < float(removal.prob):
            removed += 1
            continue

        # Backdoor conditional flip (only if trigger applied on this image)
        if do_backdoor and backdoor.enabled and cls == int(backdoor.src_class_id) and rng_bd.random() < float(backdoor.prob):
            cls = int(backdoor.target_class_id)
            backdoor_flipped += 1

        # Label flip
        if do_label_flip and label_flip.enabled and cls == int(label_flip.src_class_id) and rng_flip.random() < float(label_flip.prob):
            cls = int(label_flip.dst_class_id)
            flipped += 1

        # BBox distortion
        if do_bbox and bbox.enabled and rng_bbox.random() < float(bbox.prob):
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

    if extra_lines:
        out_lines.extend(extra_lines)

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

    Backdoor note: when ``backdoor.enabled`` is True, for each poisoned image this
    function does three things:
    1. Paints a solid-colour trigger patch on the image (via ``_apply_trigger``).
    2. Flips all ``src_class_id`` labels to ``target_class_id`` in the label file.
    3. **Adds a synthetic ``target_class_id`` annotation at the exact trigger-patch
       location.** This extra annotation creates a direct spatial association between
       the trigger patch and the target class, which is crucial for the model to learn
       the backdoor mapping even when no ``src_class_id`` object happens to overlap
       the trigger corner.
    4. Optionally duplicates each poisoned image multiple times (``oversample_factor``)
       to make the backdoor signal visible even on relatively small client shards.
    """
    images = _read_image_list_from_data_yaml(shard_data_yaml)
    shard_yaml_path = Path(shard_data_yaml)
    with open(shard_yaml_path, "r", encoding="utf-8") as f:
        shard_cfg: Dict = yaml.safe_load(f)

    val_ref = shard_cfg.get("val", "")
    default_client_val = (shard_yaml_path.parent / "images" / "val").resolve()
    if default_client_val.exists():
        resolved_val_ref = default_client_val
    else:
        resolved_val_ref = _resolve_ref(shard_cfg, str(val_ref), shard_yaml_path) if val_ref else None
    nc = shard_cfg.get("nc", None)
    names = shard_cfg.get("names", None)

    def _names_map(n) -> Dict[int, str]:
        if isinstance(n, dict):
            out: Dict[int, str] = {}
            for k, v in n.items():
                try:
                    out[int(k)] = str(v)
                except Exception:
                    continue
            return out
        if isinstance(n, list):
            return {i: str(v) for i, v in enumerate(n)}
        return {}

    def _infer_nc(nc_val, n) -> Optional[int]:
        try:
            if nc_val is not None:
                return int(nc_val)
        except Exception:
            pass
        nm = _names_map(n)
        if nm:
            return max(nm.keys()) + 1
        return None

    nc_i = _infer_nc(nc, names)
    names_i = _names_map(names)

    def _require_class_id_in_range(class_id: int, *, field: str) -> None:
        if nc_i is None:
            return
        if class_id < 0 or class_id >= int(nc_i):
            example = ", ".join(f"{k}:{names_i.get(k,'?')}" for k in sorted(list(names_i.keys()))[:5]) if names_i else ""
            raise ValueError(
                f"Class-ID mismatch: {field}={class_id} is out of range for this dataset shard (nc={nc_i}). "
                f"Check that attack config class_id values match the dataset YAML class mapping. "
                f"names[0:5]={example}"
            )

    # Validate attack class_id values against the shard YAML's class mapping.
    if label_flip.enabled:
        _require_class_id_in_range(int(label_flip.src_class_id), field="label_flip.src_class_id")
        _require_class_id_in_range(int(label_flip.dst_class_id), field="label_flip.dst_class_id")
    if backdoor.enabled:
        _require_class_id_in_range(int(backdoor.src_class_id), field="backdoor.src_class_id")
        _require_class_id_in_range(int(backdoor.target_class_id), field="backdoor.target_class_id")
    if removal.enabled:
        _require_class_id_in_range(int(removal.target_class_id), field="object_removal.target_class_id")

    out_root_p = Path(out_root)
    out_images = out_root_p / "images" / "train"
    out_labels = out_root_p / "labels" / "train"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    def _classes_present(label_path: Path) -> set[int]:
        if not label_path.exists():
            return set()
        out: set[int] = set()
        for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            toks = line.split()
            if not toks:
                continue
            try:
                out.add(int(float(toks[0])))
            except Exception:
                continue
        return out

    present = [_classes_present(_infer_label_path(img)) for img in images]

    def _pick_mask(r: random.Random, candidates: List[int], ratio: float) -> List[bool]:
        """Pick a deterministic subset of candidates (exact-k) for poisoning.

        ratio is interpreted over candidate set size (not total images).
        """
        mask = [False] * len(images)
        if not candidates:
            return mask
        rr = max(0.0, min(1.0, float(ratio)))
        k = int(round(rr * len(candidates)))
        if rr > 0.0 and k == 0:
            k = 1
        k = min(k, len(candidates))
        cand = list(candidates)
        r.shuffle(cand)
        for idx in cand[:k]:
            mask[idx] = True
        return mask

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

    # Conditional masks:
    # - label_flip/backdoor only make sense on images containing src_class_id
    # - object_removal only makes sense on images containing target_class_id
    cand_flip = [i for i, s in enumerate(present) if int(label_flip.src_class_id) in s] if label_flip.enabled else []
    cand_bd = [i for i, s in enumerate(present) if int(backdoor.src_class_id) in s] if backdoor.enabled else []
    cand_rm = [i for i, s in enumerate(present) if int(removal.target_class_id) in s] if removal.enabled else []

    mask_flip = _pick_mask(rng_flip_mask, cand_flip, float(label_flip.poison_ratio)) if label_flip.enabled else [False] * len(images)
    mask_bbox = [rng_bbox_mask.random() < max(0.0, min(1.0, float(bbox.poison_ratio))) for _ in images] if bbox.enabled else [False] * len(images)
    mask_rm = _pick_mask(rng_rm_mask, cand_rm, float(removal.poison_ratio)) if removal.enabled else [False] * len(images)
    mask_bd = _pick_mask(rng_bd_mask, cand_bd, float(backdoor.poison_ratio)) if backdoor.enabled else [False] * len(images)

    train_txt = out_root_p / "train.txt"
    out_yaml = {
        "path": ".",
        "train": train_txt.name,
        "val": _portable_ref(resolved_val_ref.resolve(), out_root_p) if resolved_val_ref is not None else "",
    }
    if nc is not None:
        out_yaml["nc"] = nc
    if names is not None:
        out_yaml["names"] = names
    out_yaml_path = out_root_p / "data.yaml"
    # Write early so downstream checks can find it even if image processing is slow.
    with open(out_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(out_yaml, f, sort_keys=False)
    total = {"lines_in": 0, "lines_out": 0, "flipped": 0, "removed": 0, "distorted": 0, "backdoor_flipped": 0}
    poisoned_images_any = 0
    poisoned_images_label_flip = 0
    poisoned_images_bbox = 0
    poisoned_images_removal = 0
    poisoned_images_backdoor = 0
    poisoned_images_backdoor_replayed = 0
    backdoor_oversample_factor = max(1, int(getattr(backdoor, "oversample_factor", 1)))

    with open(train_txt, "w", encoding="utf-8") as f:
        for img, do_flip, do_bb, do_rm, do_bd in zip(images, mask_flip, mask_bbox, mask_rm, mask_bd):
            dst_img = out_images / img.name
            trigger_bbox: Optional[Tuple[float, float, float, float]] = None
            if do_bd and backdoor.enabled:
                trigger_bbox = _apply_trigger(img, dst_img, backdoor)
                poisoned_images_backdoor += 1
            else:
                _copy_or_link(img, dst_img)

            # Build a synthetic label placing target_class directly at the trigger
            # position.  This creates a direct spatial association between the trigger
            # patch and the target class, which significantly strengthens the backdoor
            # learning signal (especially when the src_class object is far from the
            # trigger corner).
            extra_lines: Optional[List[str]] = None
            if trigger_bbox is not None:
                xc, yc, bw, bh = trigger_bbox
                extra_lines = [f"{int(backdoor.target_class_id)} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"]

            src_lbl = _infer_label_path(img)
            dst_lbl = out_labels / (dst_img.with_suffix(".txt").name)
            stats = _process_label_file(
                src_lbl,
                dst_lbl,
                do_label_flip=bool(do_flip),
                do_bbox=bool(do_bb),
                do_removal=bool(do_rm),
                do_backdoor=bool(do_bd),
                rng_flip=rng_flip,
                rng_bbox=rng_bbox,
                rng_rm=rng_rm,
                rng_bd=rng_bd,
                label_flip=label_flip,
                bbox=bbox,
                removal=removal,
                backdoor=backdoor,
                extra_lines=extra_lines,
            )
            for k in total:
                total[k] += int(stats.get(k, 0))
            poisoned_images_any += int(bool(do_flip or do_bb or do_rm or do_bd))
            poisoned_images_label_flip += int(bool(do_flip))
            poisoned_images_bbox += int(bool(do_bb))
            poisoned_images_removal += int(bool(do_rm))

            f.write(_portable_ref(dst_img.resolve(), train_txt.parent) + "\n")
            if do_bd and backdoor.enabled and backdoor_oversample_factor > 1:
                for rep_idx in range(1, backdoor_oversample_factor):
                    rep_img = out_images / f"{dst_img.stem}__bdrep{rep_idx}{dst_img.suffix}"
                    rep_lbl = out_labels / f"{dst_img.stem}__bdrep{rep_idx}.txt"
                    _copy_or_link(dst_img, rep_img)
                    _copy_or_link(dst_lbl, rep_lbl)
                    f.write(_portable_ref(rep_img.resolve(), train_txt.parent) + "\n")
                    poisoned_images_backdoor_replayed += 1

    meta = {
        "attack": "dataset_poison",
        "train_images": int(len(images)),
        "nc": int(nc_i) if nc_i is not None else None,
        "class_names": names_i if names_i else None,
        "candidates_label_flip": int(len(cand_flip)),
        "candidates_object_removal": int(len(cand_rm)),
        "candidates_backdoor": int(len(cand_bd)),
        "poisoned_images_any": int(poisoned_images_any),
        "poisoned_images_label_flip": int(poisoned_images_label_flip),
        "poisoned_images_bbox": int(poisoned_images_bbox),
        "poisoned_images_removal": int(poisoned_images_removal),
        "poisoned_images_backdoor": int(poisoned_images_backdoor),
        "poisoned_images_backdoor_replayed": int(poisoned_images_backdoor_replayed),
        "label_flip": vars(label_flip),
        "bbox_distortion": vars(bbox),
        "object_removal": vars(removal),
        "backdoor": vars(backdoor),
        **total,
    }
    with open(out_root_p / "poison_meta.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False)

    return str(out_yaml_path.resolve())

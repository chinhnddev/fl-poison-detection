import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _resolve_ref(cfg: Dict, ref: str, yaml_path: Path) -> Path:
    p = Path(str(ref))
    if p.is_absolute():
        return p
    p1 = (yaml_path.parent / p).resolve()
    if p1.exists():
        return p1
    root = str((cfg.get("path") or "")).strip()
    if root:
        root_p = Path(root)
        p2 = (yaml_path.parent / root_p / p).resolve()
        if p2.exists():
            return p2
        p3 = (Path.cwd() / root_p / p).resolve()
        if p3.exists():
            return p3
        return p2
    return p1


def _read_filelist(filelist: Path) -> Tuple[List[Path], List[str]]:
    """Return (resolved_paths, warnings)."""
    warnings: List[str] = []
    out: List[Path] = []
    lines = filelist.read_text(encoding="utf-8", errors="replace").splitlines()
    for i, raw in enumerate(lines, start=1):
        s = raw.strip()
        if not s:
            continue
        p = Path(s)
        if p.is_absolute():
            out.append(p)
            continue
        # Ultralytics resolves filelist entries as:
        # - "./..." => relative to filelist parent
        # - otherwise => relative to current working directory
        if s.startswith("./") or s.startswith(".\\"):
            out.append((filelist.parent / p).resolve())
        else:
            warnings.append(f"{filelist.name}:{i}: entry does not start with './' ({s})")
            # Try the more robust resolution first (relative to filelist parent),
            # then fall back to CWD semantics.
            cand1 = (filelist.parent / p).resolve()
            cand2 = (Path.cwd() / p).resolve()
            out.append(cand1 if cand1.exists() else cand2)
    return out, warnings


def _infer_label_path(img: Path) -> Path:
    parts = list(img.parts)
    for i, part in enumerate(parts):
        if part.lower() == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def _load_images(ref: Path) -> Tuple[List[Path], List[str]]:
    if ref.is_dir():
        imgs = [p for p in ref.rglob("*") if p.suffix.lower() in IMG_EXTS]
        return imgs, []
    if ref.is_file() and ref.suffix.lower() == ".txt":
        return _read_filelist(ref)
    raise FileNotFoundError(f"Unsupported reference: {ref}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate YOLO train/val splits and filelists.")
    ap.add_argument("--data_yaml", default="./datasets/coco128/coco128.yaml")
    args = ap.parse_args()

    yp = Path(args.data_yaml)
    if not yp.exists():
        raise SystemExit(f"Missing data_yaml: {yp}")

    cfg: Dict = yaml.safe_load(yp.read_text(encoding="utf-8")) or {}
    train_ref = cfg.get("train", "")
    val_ref = cfg.get("val", "")
    if not train_ref or not val_ref:
        raise SystemExit("data_yaml must define both 'train' and 'val'")

    train_path = _resolve_ref(cfg, str(train_ref), yp)
    val_path = _resolve_ref(cfg, str(val_ref), yp)

    train_imgs, train_warn = _load_images(train_path)
    val_imgs, val_warn = _load_images(val_path)

    train_set = {p.resolve() for p in train_imgs}
    val_set = {p.resolve() for p in val_imgs}
    overlap = sorted(train_set & val_set)

    missing_train = sorted([p for p in train_set if not p.exists()])
    missing_val = sorted([p for p in val_set if not p.exists()])

    missing_labels = 0
    for p in sorted(train_set | val_set):
        if not p.exists():
            continue
        lp = _infer_label_path(p)
        if not lp.exists():
            missing_labels += 1

    print(f"data_yaml: {yp.resolve()}")
    print(f"train_ref: {train_ref} -> {train_path}")
    print(f"val_ref:   {val_ref} -> {val_path}")
    print(f"train_images: {len(train_set)}  val_images: {len(val_set)}  overlap: {len(overlap)}")
    print(f"missing_train: {len(missing_train)}  missing_val: {len(missing_val)}  missing_labels: {missing_labels}")

    if train_warn or val_warn:
        print("\nWARNINGS:")
        for w in (train_warn + val_warn)[:50]:
            print(f"- {w}")
        if len(train_warn) + len(val_warn) > 50:
            print(f"- ... ({len(train_warn) + len(val_warn) - 50} more)")

    if overlap:
        print("\nOVERLAP (first 20):")
        for p in overlap[:20]:
            print(f"- {p}")

    if missing_train or missing_val:
        print("\nMISSING FILES (first 20):")
        for p in (missing_train + missing_val)[:20]:
            print(f"- {p}")

    # Exit non-zero on hard errors.
    if missing_train or missing_val or overlap:
        raise SystemExit(2)


if __name__ == "__main__":
    main()


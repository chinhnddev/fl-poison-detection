import argparse
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import os
import shutil
import time

import numpy as np
import yaml


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


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
        for line in ref.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                p = Path(line)
                # Make filelists portable across machines by resolving relative paths
                # relative to the filelist location.
                if not p.is_absolute():
                    p = (ref.parent / p).resolve()
                out.append(p)
        return out
    if ref.exists() and ref.is_dir():
        return [x for x in ref.rglob("*") if x.suffix.lower() in exts]
    raise FileNotFoundError(f"Unsupported image reference: {ref}")


def _infer_label_path(img: Path) -> Path:
    parts = list(img.parts)
    for i, part in enumerate(parts):
        if part.lower() == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


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


def _write_client_view(
    client_dir: Path,
    train_imgs: List[Path],
    val_imgs: List[Path],
) -> None:
    """Create a per-client dataset view (images/labels) to avoid Ultralytics cache races.

    Ultralytics writes label cache files next to the label directory. If multiple processes
    point at the same underlying labels folder (e.g. Google Drive), they can race on *.cache.
    Creating client-local label dirs makes cache files independent per client.
    """
    img_train_dir = client_dir / "images" / "train"
    lbl_train_dir = client_dir / "labels" / "train"
    img_val_dir = client_dir / "images" / "val"
    lbl_val_dir = client_dir / "labels" / "val"
    for d in [img_train_dir, lbl_train_dir, img_val_dir, lbl_val_dir]:
        d.mkdir(parents=True, exist_ok=True)

    def add_pair(img: Path, out_img_dir: Path, out_lbl_dir: Path) -> None:
        dst_img = out_img_dir / img.name
        _copy_or_link(img, dst_img)
        src_lbl = _infer_label_path(img)
        dst_lbl = out_lbl_dir / (dst_img.with_suffix(".txt").name)
        if src_lbl.exists():
            _copy_or_link(src_lbl, dst_lbl)
        else:
            dst_lbl.write_text("", encoding="utf-8")

    for img in train_imgs:
        add_pair(img, img_train_dir, lbl_train_dir)
    for img in val_imgs:
        add_pair(img, img_val_dir, lbl_val_dir)


def _dominant_class(label_path: Path) -> int:
    if not label_path.exists():
        return -1
    c = Counter()

    # On Google Drive (Colab), file reads can intermittently fail with
    # ConnectionAbortedError / transport endpoint issues. Treat as missing label.
    text = None
    for _ in range(5):
        try:
            with open(label_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            break
        except OSError:
            time.sleep(0.2)
            continue
    if text is None:
        return -1

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        toks = line.split()
        if not toks:
            continue
        try:
            cls = int(float(toks[0]))
        except Exception:
            continue
        c[cls] += 1
    if not c:
        return -1
    return int(c.most_common(1)[0][0])


def split_train_val(
    data_yaml: str,
    train_txt: str,
    val_txt: str,
    val_ratio: float,
    seed: int,
) -> Tuple[str, str]:
    """Create deterministic train/val filelists with val != train."""
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be within (0, 1)")
    set_global_seed(seed)
    yp = Path(data_yaml)
    cfg: Dict = yaml.safe_load(open(yp, "r", encoding="utf-8"))
    train_ref = cfg["train"]
    try:
        train_ref_p = _resolve_ref(cfg, str(train_ref), yp)
        all_imgs = _list_images(train_ref_p)
        # If train is a filelist but paths are invalid on this machine (e.g. Windows paths on Colab),
        # fall back to source_train to regenerate portable splits.
        if str(train_ref).lower().endswith(".txt"):
            missing = sum(1 for p in all_imgs if not p.exists())
            if missing > 0:
                raise FileNotFoundError(f"train filelist contains {missing} missing paths, falling back to source_train")
    except FileNotFoundError:
        # Support datasets where train/val are generated filelists (train.txt/val.txt).
        src = cfg.get("source_train", "")
        if not src:
            raise
        all_imgs = _list_images(_resolve_ref(cfg, str(src), yp))
    all_imgs = [p.resolve() for p in all_imgs]
    random.shuffle(all_imgs)
    n_val = max(1, int(round(len(all_imgs) * float(val_ratio))))
    val_imgs = all_imgs[:n_val]
    train_imgs = all_imgs[n_val:]
    if not train_imgs:
        train_imgs = val_imgs

    tpath = Path(train_txt)
    vpath = Path(val_txt)
    tpath.parent.mkdir(parents=True, exist_ok=True)
    vpath.parent.mkdir(parents=True, exist_ok=True)
    # Write filelists as paths relative to the dataset root (portable across machines).
    root = None
    try:
        root = _resolve_ref(cfg, str(cfg.get("path", ".")), yp).resolve()
    except Exception:
        root = yp.parent.resolve()

    def _rel(p: Path) -> str:
        try:
            return p.resolve().relative_to(root).as_posix()
        except Exception:
            return str(p.resolve())

    tpath.write_text("\n".join(_rel(p) for p in train_imgs) + "\n", encoding="utf-8")
    vpath.write_text("\n".join(_rel(p) for p in val_imgs) + "\n", encoding="utf-8")
    return str(tpath.resolve()), str(vpath.resolve())


def partition_iid(images: List[Path], num_clients: int, seed: int) -> List[List[Path]]:
    set_global_seed(seed)
    imgs = list(images)
    random.shuffle(imgs)
    shards = [[] for _ in range(num_clients)]
    for i, p in enumerate(imgs):
        shards[i % num_clients].append(p)
    return shards


def partition_dirichlet_by_dominant_class(
    images: List[Path],
    num_clients: int,
    alpha: float,
    seed: int,
) -> List[List[Path]]:
    """Non-IID partition using Dirichlet over dominant class per image."""
    if alpha <= 0:
        raise ValueError("alpha must be > 0")
    set_global_seed(seed)
    rng = np.random.default_rng(seed)

    by_class: Dict[int, List[Path]] = defaultdict(list)
    for img in images:
        dom = _dominant_class(_infer_label_path(img))
        by_class[dom].append(img)

    shards = [[] for _ in range(num_clients)]
    for cls, imgs in by_class.items():
        if not imgs:
            continue
        imgs = list(imgs)
        rng.shuffle(imgs)
        props = rng.dirichlet(alpha * np.ones(num_clients))
        counts = rng.multinomial(len(imgs), props)
        start = 0
        for cid, c in enumerate(counts.tolist()):
            if c <= 0:
                continue
            shards[cid].extend(imgs[start : start + c])
            start += c

    # Remove duplicates while keeping determinism
    for cid in range(num_clients):
        seen = set()
        uniq = []
        for p in shards[cid]:
            if p not in seen:
                uniq.append(p)
                seen.add(p)
        shards[cid] = uniq
    return shards


def write_federated_shards(
    base_data_yaml: str,
    train_txt: str,
    val_txt: str,
    out_dir: str,
    shards: List[List[Path]],
) -> None:
    yp = Path(base_data_yaml)
    base_cfg: Dict = yaml.safe_load(open(yp, "r", encoding="utf-8"))
    names = base_cfg.get("names", None)
    nc = base_cfg.get("nc", None)

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Use the global val split for all clients, but materialize it under each client
    # to avoid concurrent cache writes into a shared labels directory (Drive).
    val_images = _list_images(Path(val_txt))

    # Stats
    stats = {}
    for cid, imgs in enumerate(shards):
        client_dir = out_root / f"client_{cid}"
        client_dir.mkdir(parents=True, exist_ok=True)

        # Build a client-local dataset view so Ultralytics writes cache into client_dir/labels/*.
        _write_client_view(client_dir, train_imgs=imgs, val_imgs=val_images)

        client_yaml = {"path": f"client_{cid}", "train": "images/train", "val": "images/val"}
        if nc is not None:
            client_yaml["nc"] = nc
        if names is not None:
            client_yaml["names"] = names

        with open(client_dir / "data.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(client_yaml, f, sort_keys=False)

        # class stats
        cls_counter = Counter()
        for lbl in (client_dir / "labels" / "train").glob("*.txt"):
            for line in lbl.read_text(encoding="utf-8").splitlines():
                toks = line.strip().split()
                if not toks:
                    continue
                try:
                    cls_counter[int(float(toks[0]))] += 1
                except Exception:
                    pass
        stats[f"client_{cid}"] = {"images": len(imgs), "objects_per_class": dict(cls_counter), "val_images": len(val_images)}

    with open(out_root / "partition_stats.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(stats, f, sort_keys=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train/val split + federated partitioning.")
    ap.add_argument("--data_yaml", required=True, help="Base YOLO data.yaml")
    ap.add_argument("--num_clients", type=int, default=10)
    ap.add_argument("--out_dir", default="./federated_data")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--partition", choices=["iid", "dirichlet"], default="dirichlet")
    ap.add_argument("--dirichlet_alpha", type=float, default=0.5)
    ap.add_argument("--train_txt", default="./datasets/coco128/train.txt")
    ap.add_argument("--val_txt", default="./datasets/coco128/val.txt")
    args = ap.parse_args()

    train_txt, val_txt = split_train_val(
        data_yaml=args.data_yaml,
        train_txt=args.train_txt,
        val_txt=args.val_txt,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    yp = Path(args.data_yaml)
    cfg: Dict = yaml.safe_load(open(yp, "r", encoding="utf-8"))
    # Partition only the *train* images
    train_images = _list_images(Path(train_txt))

    if args.partition == "iid":
        shards = partition_iid(train_images, args.num_clients, seed=args.seed)
    else:
        shards = partition_dirichlet_by_dominant_class(train_images, args.num_clients, alpha=args.dirichlet_alpha, seed=args.seed)

    write_federated_shards(
        base_data_yaml=args.data_yaml,
        train_txt=train_txt,
        val_txt=val_txt,
        out_dir=args.out_dir,
        shards=shards,
    )

    print(f"Done. train_txt={train_txt} val_txt={val_txt}")
    print(f"Generated {args.num_clients} client shards in: {Path(args.out_dir).resolve()} (partition={args.partition})")


if __name__ == "__main__":
    main()

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import shutil
import time

import numpy as np
import yaml

from scripts.download_coco import ensure_coco_val2017_for_yaml


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
        root_p = Path(str(root))
        # Typical YOLO: root is relative to the YAML file location.
        p2 = (yaml_path.parent / root_p / p).resolve()
        if p2.exists():
            return p2
        # If YAML lives *inside* the dataset root, users sometimes set `path: datasets/...`
        # (relative to repo CWD). Support that case too.
        p3 = (Path.cwd() / root_p / p).resolve()
        if p3.exists():
            return p3
        return p2
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


def _dataset_root_from_cfg(cfg: Dict, yaml_path: Path) -> Path:
    root = str((cfg.get("path") or "")).strip()
    if not root:
        return yaml_path.parent.resolve()
    root_p = Path(root)
    if root_p.is_absolute():
        return root_p.resolve()
    cand1 = (yaml_path.parent / root_p).resolve()
    if cand1.exists():
        return cand1
    cand2 = (Path.cwd() / root_p).resolve()
    if cand2.exists():
        return cand2
    return cand1


def _derive_default_split_paths(data_yaml: str) -> Tuple[Path, Path]:
    yp = Path(data_yaml)
    cfg: Dict = yaml.safe_load(open(yp, "r", encoding="utf-8")) or {}
    dataset_root = _dataset_root_from_cfg(cfg, yp)
    return dataset_root / "train.txt", dataset_root / "val.txt"


def _derive_default_out_dir(data_yaml: str) -> Path:
    yp = Path(data_yaml)
    cfg: Dict = yaml.safe_load(open(yp, "r", encoding="utf-8")) or {}
    dataset_root = _dataset_root_from_cfg(cfg, yp)
    dataset_name = dataset_root.name or yp.stem
    return (Path.cwd() / "partitions" / dataset_name).resolve()


def _missing_dataset_message(data_yaml: str, attempted_ref: Path, cfg: Dict) -> str:
    yp = Path(data_yaml).resolve()
    dataset_root = _dataset_root_from_cfg(cfg, yp)
    source_train = str((cfg.get("source_train") or "")).strip() or "<unset>"
    train_ref = str((cfg.get("train") or "")).strip() or "<unset>"
    val_ref = str((cfg.get("val") or "")).strip() or "<unset>"
    expected_images = (dataset_root / "images" / "val2017").resolve()
    expected_labels = (dataset_root / "labels" / "val2017").resolve()
    return (
        "Dataset source for partitioning was not found.\n"
        f"- data_yaml: {yp}\n"
        f"- attempted_ref: {attempted_ref.resolve()}\n"
        f"- dataset_root: {dataset_root}\n"
        f"- source_train: {source_train}\n"
        f"- train: {train_ref}\n"
        f"- val: {val_ref}\n"
        "Expected COCO val2017 YOLO layout:\n"
        f"- images: {expected_images}\n"
        f"- labels: {expected_labels}\n"
        "This machine currently does not contain that dataset path, so partitioning cannot start."
    )


def _copy_or_link(src: Path, dst: Path) -> None:
    """Force copy2 on Windows (no symlink)"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    print(f"COPY2 {src.name} -> {dst.name}")
    # Google Drive / networked FS can race on metadata updates; fall back to copyfile if copy2 fails.
    try:
        shutil.copy2(src, dst)
    except FileNotFoundError:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        try:
            shutil.copystat(src, dst)
        except OSError:
            pass


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

    # Prefer `source_train` as the canonical pool (so stale train.txt won't silently shrink the dataset).
    src = str((cfg.get("source_train") or "")).strip()
    train_ref = cfg.get("train", "")
    pool_ref = src or str(train_ref)

    try:
        pool_ref_p = _resolve_ref(cfg, pool_ref, yp)
        all_imgs = _list_images(pool_ref_p)
        # If the pool is a filelist, verify that it resolves on this machine. If not, fall back to source_train.
        if str(pool_ref).lower().endswith(".txt"):
            missing = sum(1 for p in all_imgs if not p.exists())
            if missing > 0 and src:
                pool_ref_p = _resolve_ref(cfg, src, yp)
                all_imgs = _list_images(pool_ref_p)
    except FileNotFoundError:
        # Last resort: try cfg["train"] if source_train is missing/misconfigured.
        pool_ref_p = _resolve_ref(cfg, str(train_ref), yp)
        try:
            all_imgs = _list_images(pool_ref_p)
        except FileNotFoundError as exc:
            raise FileNotFoundError(_missing_dataset_message(data_yaml, pool_ref_p, cfg)) from exc
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
    # IMPORTANT: Ultralytics treats filelist lines as global paths unless they start with "./".
    # Using "./" makes them resolve relative to the filelist location (i.e. dataset root here).
    root = _dataset_root_from_cfg(cfg, yp)

    def _rel(p: Path) -> str:
        try:
            rel = p.resolve().relative_to(root).as_posix()
            return rel if rel.startswith("./") else "./" + rel
        except Exception:
            return str(p.resolve())

    tpath.write_text("\n".join(_rel(p) for p in train_imgs) + "\n", encoding="utf-8")
    vpath.write_text("\n".join(_rel(p) for p in val_imgs) + "\n", encoding="utf-8")
    return str(tpath.resolve()), str(vpath.resolve())


def partition_iid(images: List[Path], num_clients: int, seed: int) -> List[List[Path]]:
    """
    Round-robin IID + guarantee min 1 image per client.
    """
    set_global_seed(seed)
    imgs = list(images)
    if len(imgs) < num_clients:
        # Duplicate data if too few images
        imgs = imgs * ((num_clients // len(imgs)) + 1)
        imgs = imgs[:num_clients * 2]  # Min 2/client
    
    random.shuffle(imgs)
    shards = [[] for _ in range(num_clients)]
    
    # Round-robin
    for i, p in enumerate(imgs):
        shards[i % num_clients].append(p)
    
    # Min 1 guarantee: redistribute from largest
    sizes = [len(s) for s in shards]
    while min(sizes) == 0:
        max_cid = sizes.index(max(sizes))
        if len(shards[max_cid]) > 1:
            # Move 1 image from largest to empty
            empty_cid = sizes.index(0)
            img = shards[max_cid].pop(0)
            shards[empty_cid].append(img)
            sizes[empty_cid] += 1
            sizes[max_cid] -= 1
        else:
            break
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


def ensure_min_images_per_client(shards: List[List[Path]], min_images: int, seed: int) -> List[List[Path]]:
    """Ensure each client has at least `min_images` training images.

    Dirichlet sampling on small datasets can create empty shards. We deterministically
    rebalance by moving images from the largest shards.
    """
    if min_images <= 0:
        return shards
    rng = random.Random(int(seed) + 99991)
    shards = [list(s) for s in shards]
    n = len(shards)
    # Deterministic order for receivers.
    receivers = [i for i in range(n) if len(shards[i]) < min_images]
    for r in receivers:
        need = min_images - len(shards[r])
        for _ in range(need):
            donors = sorted(range(n), key=lambda i: (-len(shards[i]), i))
            donor = next((d for d in donors if len(shards[d]) > min_images), None)
            if donor is None:
                donor = next((d for d in donors if len(shards[d]) > 1), None)
            if donor is None:
                break
            # Pick a deterministic element from donor.
            idx = rng.randrange(len(shards[donor]))
            shards[r].append(shards[donor].pop(idx))
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
    dataset_root = _dataset_root_from_cfg(base_cfg, yp)
    manifest = {
        "base_data_yaml": str(yp.resolve()),
        "dataset_root": str(dataset_root),
        "train_txt": str(Path(train_txt).resolve()),
        "val_txt": str(Path(val_txt).resolve()),
        "num_clients": len(shards),
        "clients": {},
    }

    def _portable_path(p: Path) -> str:
        try:
            return p.resolve().relative_to(dataset_root).as_posix()
        except Exception:
            return str(p.resolve())

    for cid, imgs in enumerate(shards):
        client_dir = out_root / f"client_{cid}"
        client_dir.mkdir(parents=True, exist_ok=True)

        # Build a client-local dataset view so Ultralytics writes cache into client_dir/labels/*.
        _write_client_view(client_dir, train_imgs=imgs, val_imgs=val_images)

        client_yaml = {
            "path": str(client_dir.resolve()),
            "train": str((client_dir / "images" / "train").resolve()),
            "val": str((client_dir / "images" / "val").resolve())
        }
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
        manifest["clients"][f"client_{cid}"] = {
            "train_images": [_portable_path(p) for p in imgs],
            "num_train_images": len(imgs),
            "num_val_images": len(val_images),
            "data_yaml": str((client_dir / "data.yaml").resolve()),
        }

    with open(out_root / "partition_stats.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(stats, f, sort_keys=False)
    with open(out_root / "partition_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train/val split + federated partitioning.")
    ap.add_argument("--data_yaml", required=True, help="Base YOLO data.yaml")
    ap.add_argument("--num_clients", type=int, default=10)
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--partition", choices=["iid", "dirichlet"], default="dirichlet")
    ap.add_argument("--dirichlet_alpha", type=float, default=0.5)
    ap.add_argument("--min_images_per_client", type=int, default=1)
    ap.add_argument("--train_txt", default="")
    ap.add_argument("--val_txt", default="")
    args = ap.parse_args()

    ensure_coco_val2017_for_yaml(args.data_yaml, verbose=True)

    default_train_txt, default_val_txt = _derive_default_split_paths(args.data_yaml)
    train_txt_arg = args.train_txt or str(default_train_txt)
    val_txt_arg = args.val_txt or str(default_val_txt)
    out_dir = args.out_dir or str(_derive_default_out_dir(args.data_yaml))

    train_txt, val_txt = split_train_val(
        data_yaml=args.data_yaml,
        train_txt=train_txt_arg,
        val_txt=val_txt_arg,
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

    shards = ensure_min_images_per_client(shards, min_images=int(args.min_images_per_client), seed=int(args.seed))

    write_federated_shards(
        base_data_yaml=args.data_yaml,
        train_txt=train_txt,
        val_txt=val_txt,
        out_dir=out_dir,
        shards=shards,
    )

    shared_val_images = _list_images(Path(val_txt))
    total_source_images = len(train_images) + len(shared_val_images)
    print(f"Done. train_txt={train_txt} val_txt={val_txt}")
    print(
        f"Source dataset size={total_source_images}, partitioned_train_images={len(train_images)}, "
        f"shared_val_images={len(shared_val_images)}"
    )
    print(f"Generated {args.num_clients} client shards in: {Path(out_dir).resolve()} (partition={args.partition})")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from attack import (
    BackdoorConfig,
    BBoxDistortionConfig,
    LabelFlipConfig,
    ObjectRemovalConfig,
    build_poisoned_dataset,
)


def _abs_path(path_str: str, base_dir: Path) -> Path:
    p = Path(str(path_str)).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (base_dir / p).resolve()


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Build poisoned data for one client shard.")
    ap.add_argument("--client", type=int, required=True, help="Client id, e.g. 3")
    ap.add_argument("--config", default="config.attack.yaml", help="Attack config YAML")
    ap.add_argument(
        "--poison-ratio",
        type=float,
        default=None,
        help="Override attack.backdoor.poison_ratio (fraction in [0,1], e.g. 0.3).",
    )
    ap.add_argument(
        "--out-root",
        default="",
        help="Optional output directory. Defaults to tmp/poison/client_<cid>_<sig>.",
    )
    args = ap.parse_args()

    if args.client < 0:
        raise SystemExit(f"--client must be >= 0, got {args.client}")
    if args.poison_ratio is not None and not (0.0 <= float(args.poison_ratio) <= 1.0):
        raise SystemExit(f"--poison-ratio must be in [0,1], got {args.poison_ratio}")

    cfg_path = _abs_path(args.config, repo_root)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    fed_cfg = cfg.get("federated") or {}
    shard_yaml = _abs_path(str(Path(str(fed_cfg.get("data_dir", "./federated_data_coco10"))) / f"client_{int(args.client)}" / "data.yaml"), repo_root)
    if not shard_yaml.exists():
        raise SystemExit(
            f"Client shard not found: {shard_yaml}\n"
            "Create partitions first, for example with:\n"
            "python data_partition.py --data_yaml ./datasets/coco10/coco10.yaml --num_clients 10 --out_dir ./federated_data_coco10 --seed 1234 --partition iid --dirichlet_alpha 2.0 --val_ratio 0.2"
        )

    attack_cfg = cfg.get("attack") or {}
    lf = attack_cfg.get("label_flip") or {}
    bb = attack_cfg.get("bbox_distortion") or {}
    rm = attack_cfg.get("object_removal") or {}
    bd = attack_cfg.get("backdoor") or {}

    poison_ratio = float(args.poison_ratio) if args.poison_ratio is not None else float(bd.get("poison_ratio", 0.30))

    label_flip = LabelFlipConfig(
        enabled=bool(lf.get("enabled", False)),
        poison_ratio=float(lf.get("poison_ratio", 0.2)),
        src_class_id=int(lf.get("src_class_id", 0)),
        dst_class_id=int(lf.get("dst_class_id", 56)),
        prob=float(lf.get("prob", 0.5)),
        seed=int(lf.get("seed", 42)) + int(args.client),
    )
    bbox = BBoxDistortionConfig(
        enabled=bool(bb.get("enabled", False)),
        poison_ratio=float(bb.get("poison_ratio", 0.2)),
        shift_xy=float(bb.get("shift_xy", bb.get("shift", 0.05))),
        shift_wh=float(bb.get("shift_wh", 0.03)),
        prob=float(bb.get("prob", 0.3)),
        seed=int(bb.get("seed", 43)) + int(args.client),
    )
    removal = ObjectRemovalConfig(
        enabled=bool(rm.get("enabled", False)),
        poison_ratio=float(rm.get("poison_ratio", 0.2)),
        target_class_id=int(rm.get("target_class_id", rm.get("target_class", 0))),
        prob=float(rm.get("prob", 0.3)),
        seed=int(rm.get("seed", 44)) + int(args.client),
    )
    backdoor = BackdoorConfig(
        enabled=bool(bd.get("enabled", False)),
        poison_ratio=poison_ratio,
        oversample_factor=int(bd.get("oversample_factor", 1)),
        trigger_size=int(bd.get("trigger_size", 40)),
        trigger_value=int(bd.get("trigger_value", 255)),
        position=str(bd.get("position", "bottom_right")),
        src_class_id=int(bd.get("src_class_id", bd.get("src_class", 0))),
        target_class_id=int(bd.get("target_class_id", bd.get("target_class", 44))),
        prob=float(bd.get("prob", 1.0)),
        seed=int(bd.get("seed", 45)) + int(args.client),
    )

    if args.out_root:
        out_root = _abs_path(args.out_root, repo_root)
    else:
        sig_obj = {
            "client": int(args.client),
            "label_flip": vars(label_flip),
            "bbox_distortion": vars(bbox),
            "object_removal": vars(removal),
            "backdoor": vars(backdoor),
        }
        sig = hashlib.md5(json.dumps(sig_obj, sort_keys=True).encode("utf-8")).hexdigest()[:10]
        out_root = _abs_path(f"./tmp/poison/client_{int(args.client)}_{sig}", repo_root)

    out_yaml = build_poisoned_dataset(
        shard_data_yaml=str(shard_yaml),
        out_root=str(out_root),
        label_flip=label_flip,
        bbox=bbox,
        removal=removal,
        backdoor=backdoor,
    )

    print(f"Client shard: {shard_yaml}")
    print(f"Poisoned data YAML: {out_yaml}")
    print(f"Output root: {out_root}")
    print(f"Using backdoor poison_ratio={poison_ratio:.4f}")
    meta_path = out_root / "poison_meta.yaml"
    if meta_path.exists():
        print(f"Poison meta: {meta_path}")


if __name__ == "__main__":
    main()

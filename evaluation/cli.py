from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Optional

from .asr import asr_backdoor_object_level, inspect_backdoor_asr_pair
from .map_eval import evaluate_map
from .perception_metrics import evaluate_perception_metrics


@dataclass
class EvalResult:
    name: str
    model: str
    map50: Optional[float]
    map5095: Optional[float]
    asr: Optional[float]
    extra: Dict


def _print_table(rows):
    cols = ["name", "model", "mAP@0.5", "mAP@0.5:0.95", "ASR"]
    data = []
    for r in rows:
        data.append(
            [
                r.name,
                r.model,
                "-" if r.map50 is None else f"{r.map50:.4f}",
                "-" if r.map5095 is None else f"{r.map5095:.4f}",
                "-" if r.asr is None else f"{r.asr:.4f}",
            ]
        )
    widths = [len(c) for c in cols]
    for row in data:
        for i, v in enumerate(row):
            widths[i] = max(widths[i], len(str(v)))
    fmt = "  ".join("{:<" + str(w) + "}" for w in widths)
    print(fmt.format(*cols))
    print(fmt.format(*["-" * w for w in widths]))
    for row in data:
        print(fmt.format(*row))


def _print_perception(rows):
    headers = ["name", "missing", "ghost", "cls_mismatch", "box_dev"]
    data = []
    for row in rows:
        extra = row.extra.get("perception") or {}
        data.append(
            [
                row.name,
                "-" if extra.get("missing_object_rate") is None else f"{extra['missing_object_rate']:.4f}",
                "-" if extra.get("ghost_object_rate") is None else f"{extra['ghost_object_rate']:.4f}",
                "-" if extra.get("class_mismatch_rate") is None else f"{extra['class_mismatch_rate']:.4f}",
                "-" if extra.get("mean_box_deviation") is None else f"{extra['mean_box_deviation']:.4f}",
            ]
        )
    widths = [len(c) for c in headers]
    for row in data:
        for i, v in enumerate(row):
            widths[i] = max(widths[i], len(str(v)))
    fmt = "  ".join("{:<" + str(w) + "}" for w in widths)
    print("\nPerception Metrics")
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for row in data:
        print(fmt.format(*row))


def _print_asr_pair_summary(pair_info: Dict) -> None:
    target = pair_info.get("target") or {}
    src = pair_info.get("src") or {}
    src_mean_ratio = src.get("mean_ratio")
    src_mean_area = src.get("mean_area")
    target_geom = target.get("geometry_score")
    print("\nASR Pair Diagnostics")
    print(
        f"src={src.get('class_id')} ({src.get('name')}) "
        f"train_objs={src.get('train_instances')} val_objs={src.get('val_instances')} "
        f"mean_ratio={src_mean_ratio:.3f} mean_area={src_mean_area:.4f}"
        if src_mean_ratio is not None and src_mean_area is not None
        else f"src={src.get('class_id')} ({src.get('name')}) "
        f"train_objs={src.get('train_instances')} val_objs={src.get('val_instances')}"
    )
    print(
        f"target={target.get('class_id')} ({target.get('name')}) "
        f"train_objs={target.get('train_instances')} train_imgs={target.get('train_images')} "
        f"val_objs={target.get('val_instances')} val_imgs={target.get('val_images')} "
        f"val_cooccur_with_src={target.get('val_images_with_src')} "
        f"geometry_score={target_geom:.3f}" if target_geom is not None else
        f"target={target.get('class_id')} ({target.get('name')}) "
        f"train_objs={target.get('train_instances')} train_imgs={target.get('train_images')} "
        f"val_objs={target.get('val_instances')} val_imgs={target.get('val_images')} "
        f"val_cooccur_with_src={target.get('val_images_with_src')} geometry_score=-"
    )
    for warning in pair_info.get("warnings") or []:
        print(f"- {warning}")
    recs = pair_info.get("recommended_targets") or []
    if recs:
        parts = []
        for item in recs:
            geom = item.get("geometry_score")
            geom_text = "-" if geom is None else f"{geom:.3f}"
            parts.append(
                f"{item['class_id']} ({item['name']}, train_objs={item['train_instances']}, "
                f"cooccur={item['val_images_with_src']}, geom={geom_text})"
            )
        pretty = ", ".join(parts)
        print(f"recommended_targets: {pretty}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--baseline", default="")
    ap.add_argument("--attacked", default="")
    ap.add_argument("--defended", default="")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--asr_src_class_id", type=int, default=-1)
    ap.add_argument("--asr_target_class_id", type=int, default=-1)
    ap.add_argument("--asr_iou", type=float, default=0.5)
    ap.add_argument("--asr_limit_images", type=int, default=0)
    ap.add_argument("--asr_mode", choices=["strict", "relaxed"], default="relaxed")
    ap.add_argument("--asr_trigger", action="store_true", help="Inject trigger into val images when computing ASR")
    ap.add_argument(
        "--asr_pair_report",
        action="store_true",
        help="Print dataset diagnostics for the chosen ASR source/target pair and suggested alternatives.",
    )
    ap.add_argument("--trigger_size", type=int, default=40)
    ap.add_argument("--trigger_value", type=int, default=255)
    ap.add_argument("--trigger_position", default="bottom_right")
    ap.add_argument("--perception", action="store_true", help="Evaluate perception-oriented detection metrics")
    ap.add_argument("--perception_data", default="", help="Optional dataset YAML override for perception metrics")
    ap.add_argument("--perception_conf", type=float, default=0.25)
    ap.add_argument("--perception_max_images", type=int, default=0)
    ap.add_argument("--perception_class_penalty", type=float, default=0.5)
    args = ap.parse_args()

    pair_info = None
    if args.asr_src_class_id >= 0 and args.asr_target_class_id >= 0:
        pair_info = inspect_backdoor_asr_pair(
            data_yaml=args.data,
            src_class_id=int(args.asr_src_class_id),
            target_class_id=int(args.asr_target_class_id),
        )
        if pair_info.get("warnings"):
            _print_asr_pair_summary(
                {
                    **pair_info,
                    "recommended_targets": (pair_info.get("recommended_targets") or [])[:3],
                }
            )
        elif args.asr_pair_report:
            _print_asr_pair_summary(pair_info)

    rows = []
    for name, mp in [("baseline", args.baseline), ("attacked", args.attacked), ("defended", args.defended)]:
        if not mp:
            continue
        mm = evaluate_map(mp, args.data, args.imgsz, args.device)
        asr = None
        if args.asr_src_class_id >= 0 and args.asr_target_class_id >= 0:
            asr = asr_backdoor_object_level(
                model_path=mp,
                data_yaml=args.data,
                src_class_id=args.asr_src_class_id,
                target_class_id=args.asr_target_class_id,
                imgsz=args.imgsz,
                device=args.device,
                conf=args.conf,
                iou_thres=args.asr_iou,
                trigger=bool(args.asr_trigger),
                trigger_size=args.trigger_size,
                trigger_value=args.trigger_value,
                trigger_position=args.trigger_position,
                mode=str(args.asr_mode),
                limit_images=args.asr_limit_images,
            )
        extra = {}
        if args.perception:
            extra["perception"] = evaluate_perception_metrics(
                model_path=mp,
                data_yaml=args.perception_data or args.data,
                imgsz=args.imgsz,
                device=args.device,
                conf=args.perception_conf,
                max_images=args.perception_max_images,
                class_penalty=args.perception_class_penalty,
            )
        rows.append(EvalResult(name=name, model=mp, map50=mm["map50"], map5095=mm["map5095"], asr=asr, extra=extra))

    _print_table(rows)
    if args.perception and rows:
        _print_perception(rows)


if __name__ == "__main__":
    main()

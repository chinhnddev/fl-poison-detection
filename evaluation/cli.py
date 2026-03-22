from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Optional

from .asr import asr_backdoor_object_level
from .map_eval import evaluate_map


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
    ap.add_argument("--asr_mode", choices=["strict", "relaxed"], default="strict")
    ap.add_argument("--asr_trigger", action="store_true", help="Inject trigger into val images when computing ASR")
    ap.add_argument("--trigger_size", type=int, default=16)
    ap.add_argument("--trigger_value", type=int, default=255)
    ap.add_argument("--trigger_position", default="bottom_right")
    args = ap.parse_args()

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
        rows.append(EvalResult(name=name, model=mp, map50=mm["map50"], map5095=mm["map5095"], asr=asr, extra={}))

    _print_table(rows)


if __name__ == "__main__":
    main()

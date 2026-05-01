from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence


@dataclass
class RoundMetricRow:
    round: int
    model_path: str
    map50: Optional[float]
    map5095: Optional[float]


def load_round_tracking_cfg(cfg: Dict) -> Dict:
    eval_cfg = cfg.get("eval") or {}
    rtcfg = dict(eval_cfg.get("round_tracking") or {})
    enabled = bool(rtcfg.get("enabled", False))
    every_n_rounds = max(1, int(rtcfg.get("every_n_rounds", 1) or 1))
    keep_final = bool(rtcfg.get("keep_final", True))
    metric = str(rtcfg.get("selection_metric", "map5095") or "map5095").strip().lower()
    if metric not in {"map50", "map5095"}:
        metric = "map5095"
    patience = max(1, int(rtcfg.get("patience", 5) or 5))
    min_delta = float(rtcfg.get("min_delta", 0.001) or 0.0)
    plot = bool(rtcfg.get("plot", True))
    copy_best = bool(rtcfg.get("copy_best_checkpoint", True))
    imgsz = int(rtcfg.get("imgsz", eval_cfg.get("imgsz", cfg.get("train", {}).get("imgsz", 320))) or 320)
    data_yaml = str(rtcfg.get("data_yaml", eval_cfg.get("data_yaml", "")) or "")
    device = str(rtcfg.get("device", cfg.get("train", {}).get("device", "cpu")) or "cpu")
    return {
        "enabled": enabled,
        "every_n_rounds": every_n_rounds,
        "keep_final": keep_final,
        "selection_metric": metric,
        "patience": patience,
        "min_delta": min_delta,
        "plot": plot,
        "copy_best_checkpoint": copy_best,
        "imgsz": imgsz,
        "data_yaml": data_yaml,
        "device": device,
    }


def should_save_round_snapshot(server_round: int, total_rounds: int, tracking_cfg: Dict) -> bool:
    if int(server_round) <= 0:
        return False
    if not bool(tracking_cfg.get("enabled", False)):
        return False
    every_n = max(1, int(tracking_cfg.get("every_n_rounds", 1) or 1))
    keep_final = bool(tracking_cfg.get("keep_final", True))
    if int(server_round) % every_n == 0:
        return True
    return keep_final and int(server_round) == int(total_rounds)


def discover_round_checkpoints(global_out: str, rounds: int) -> List[tuple[int, Path]]:
    out = Path(global_out)
    found: List[tuple[int, Path]] = []
    for round_idx in range(1, int(rounds) + 1):
        candidate = out.with_name(f"{out.stem}_round_{int(round_idx):04d}{out.suffix}")
        if candidate.exists():
            found.append((int(round_idx), candidate.resolve()))
    return found


def summarize_round_metrics(rows: Sequence[RoundMetricRow], selection_metric: str, patience: int, min_delta: float) -> Dict:
    metric_name = str(selection_metric).strip().lower()
    if metric_name not in {"map50", "map5095"}:
        metric_name = "map5095"

    valid_rows = [row for row in rows if getattr(row, metric_name) is not None]
    if not valid_rows:
        return {
            "selection_metric": metric_name,
            "num_evaluated": 0,
            "best_round": None,
            "best_value": None,
            "best_model_path": "",
            "convergence_round": None,
            "convergence_reason": "no_valid_metrics",
        }

    best_row = max(valid_rows, key=lambda row: float(getattr(row, metric_name)))
    best_value = float(getattr(best_row, metric_name))

    running_best = -math.inf
    last_improvement_round: Optional[int] = None
    stall_count = 0
    convergence_round: Optional[int] = None
    convergence_reason = "not_reached"

    for row in valid_rows:
        value = float(getattr(row, metric_name))
        if value > (running_best + float(min_delta)):
            running_best = value
            last_improvement_round = int(row.round)
            stall_count = 0
        else:
            stall_count += 1
            if convergence_round is None and last_improvement_round is not None and stall_count >= int(patience):
                convergence_round = int(last_improvement_round)
                convergence_reason = f"no_{metric_name}_gain_ge_{float(min_delta):.6f}_for_{int(patience)}_checks"

    return {
        "selection_metric": metric_name,
        "num_evaluated": len(valid_rows),
        "best_round": int(best_row.round),
        "best_value": best_value,
        "best_model_path": str(best_row.model_path),
        "convergence_round": convergence_round,
        "convergence_reason": convergence_reason,
    }


def write_round_metrics_csv(rows: Sequence[RoundMetricRow], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["round", "model_path", "map50", "map5095"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "round": int(row.round),
                    "model_path": str(row.model_path),
                    "map50": "" if row.map50 is None else f"{float(row.map50):.8f}",
                    "map5095": "" if row.map5095 is None else f"{float(row.map5095):.8f}",
                }
            )


def write_round_metrics_json(rows: Sequence[RoundMetricRow], summary: Dict, out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary,
        "rows": [
            {
                "round": int(row.round),
                "model_path": str(row.model_path),
                "map50": row.map50,
                "map5095": row.map5095,
            }
            for row in rows
        ],
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_round_metrics(rows: Sequence[RoundMetricRow], summary: Dict, out_png: Path) -> None:
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rounds = [int(row.round) for row in rows]
    map50 = [row.map50 for row in rows]
    map5095 = [row.map5095 for row in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    if any(v is not None for v in map50):
        ax.plot(rounds, map50, marker="o", linewidth=1.5, label="mAP@0.5")
    if any(v is not None for v in map5095):
        ax.plot(rounds, map5095, marker="s", linewidth=1.5, label="mAP@0.5:0.95")

    best_round = summary.get("best_round")
    best_value = summary.get("best_value")
    metric_name = summary.get("selection_metric", "map5095")
    if best_round is not None and best_value is not None:
        ax.axvline(int(best_round), color="tab:green", linestyle="--", linewidth=1.2, label=f"best {metric_name}")
        ax.scatter([int(best_round)], [float(best_value)], color="tab:green", zorder=5)

    convergence_round = summary.get("convergence_round")
    if convergence_round is not None:
        ax.axvline(int(convergence_round), color="tab:red", linestyle=":", linewidth=1.2, label="plateau")

    ax.set_title("Validation mAP by Federated Round")
    ax.set_xlabel("Round")
    ax.set_ylabel("mAP")
    ax.set_xticks(rounds)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def evaluate_round_checkpoints(
    *,
    global_out: str,
    rounds: int,
    data_yaml: str,
    imgsz: int,
    device: str,
    log_dir: str,
    tracking_cfg: Dict,
) -> Dict:
    from .map_eval import evaluate_map

    checkpoints = discover_round_checkpoints(global_out=global_out, rounds=rounds)
    rows: List[RoundMetricRow] = []
    for round_idx, model_path in checkpoints:
        mm = evaluate_map(str(model_path), data_yaml, imgsz, device)
        rows.append(
            RoundMetricRow(
                round=int(round_idx),
                model_path=str(model_path),
                map50=mm.get("map50"),
                map5095=mm.get("map5095"),
            )
        )

    summary = summarize_round_metrics(
        rows=rows,
        selection_metric=str(tracking_cfg.get("selection_metric", "map5095")),
        patience=int(tracking_cfg.get("patience", 5)),
        min_delta=float(tracking_cfg.get("min_delta", 0.001)),
    )

    out_dir = Path(log_dir).resolve()
    csv_path = out_dir / "round_metrics.csv"
    json_path = out_dir / "round_metrics.json"
    png_path = out_dir / "round_metrics.png"
    write_round_metrics_csv(rows, csv_path)
    write_round_metrics_json(rows, summary, json_path)
    if bool(tracking_cfg.get("plot", True)):
        plot_round_metrics(rows, summary, png_path)

    best_ckpt_out = ""
    if bool(tracking_cfg.get("copy_best_checkpoint", True)) and summary.get("best_model_path"):
        src = Path(str(summary["best_model_path"])).resolve()
        dst = Path(global_out).resolve().with_name(f"{Path(global_out).stem}_best{Path(global_out).suffix}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src != dst:
            shutil.copy2(src, dst)
        best_ckpt_out = str(dst)

    return {
        "rows": rows,
        "summary": summary,
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "png_path": str(png_path),
        "best_checkpoint_path": best_ckpt_out,
    }

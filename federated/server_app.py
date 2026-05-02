from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import yaml
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy

from aggregation import weighted_fedavg
from defense import (
    DefenseConfig,
    DetectionAwareDefenseConfig,
    SPCHMTrustConfig,
    detection_aware_filter,
    robust_filter,
    run_spchm_trust_round,
)
from evaluation.round_tracking import load_round_tracking_cfg, should_save_round_snapshot
from train_yolo import get_parameters, resolve_base_model_for_data, set_parameters_to_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("server")

def _load_defense_cfg(cfg: Dict) -> DefenseConfig:
    d = cfg.get("defense") or {}
    w = d.get("weights") or {}
    return DefenseConfig(
        enabled=bool(d.get("enabled", False)),
        cosine_z_threshold=float(d.get("cosine_z", d.get("cosine_z_threshold", 1.8))),
        norm_z_threshold=float(d.get("norm_z", d.get("norm_z_threshold", 2.5))),
        dist_z_threshold=float(d.get("dist_z", d.get("dist_z_threshold", 2.0))),
        min_score=float(d.get("min_score", 2.5)),
        weight_cosine=float(w.get("cosine", d.get("weight_cosine", 1.5))),
        weight_norm=float(w.get("norm", d.get("weight_norm", 1.0))),
        weight_dist=float(w.get("dist", d.get("weight_dist", 1.0))),
        use_mad=bool(d.get("use_mad", True)),
        min_clients=int(d.get("min_clients", 4)),
    )


def _load_detection_aware_cfg(cfg: Dict) -> Optional[DetectionAwareDefenseConfig]:
    """Return a DetectionAwareDefenseConfig if detection-aware defense is configured.

    The detection-aware defense is activated when ``defense.detection_aware: true``
    is set in the YAML.  All gradient and detection thresholds can be overridden
    under the same ``defense`` block.
    """
    d = cfg.get("defense") or {}
    if not bool(d.get("detection_aware", False)):
        return None
    w = d.get("weights") or {}
    dw = d.get("detection_weights") or {}
    return DetectionAwareDefenseConfig(
        enabled=bool(d.get("enabled", True)),
        cosine_z_threshold=float(d.get("cosine_z", d.get("cosine_z_threshold", 1.8))),
        norm_z_threshold=float(d.get("norm_z", d.get("norm_z_threshold", 2.5))),
        dist_z_threshold=float(d.get("dist_z", d.get("dist_z_threshold", 2.0))),
        use_mad=bool(d.get("use_mad", True)),
        class_freq_z_threshold=float(d.get("class_freq_z", 2.0)),
        bbox_z_threshold=float(d.get("bbox_z", 2.0)),
        detection_rate_z_threshold=float(d.get("detection_rate_z", 2.0)),
        iou_z_threshold=float(d.get("iou_z", 2.0)),
        weight_gradient_cosine=float(w.get("cosine", d.get("weight_gradient_cosine", 1.5))),
        weight_gradient_norm=float(w.get("norm", d.get("weight_gradient_norm", 1.0))),
        weight_gradient_dist=float(w.get("dist", d.get("weight_gradient_dist", 1.0))),
        weight_class_freq=float(dw.get("class_freq", d.get("weight_class_freq", 2.0))),
        weight_bbox=float(dw.get("bbox", d.get("weight_bbox", 1.0))),
        weight_detection_rate=float(dw.get("detection_rate", d.get("weight_detection_rate", 1.0))),
        weight_iou=float(dw.get("iou", d.get("weight_iou", 1.5))),
        min_score=float(d.get("min_score", 2.5)),
        min_clients=int(d.get("min_clients", 4)),
        nc=int(d.get("nc", 80)),
    )


def _load_spchm_cfg(cfg: Dict) -> Optional[SPCHMTrustConfig]:
    d = cfg.get("defense") or {}
    if not bool(d.get("spchm_trust", False)):
        return None

    train_cfg = cfg.get("train") or {}
    runtime_cfg = cfg.get("runtime") or {}
    train_overrides = {
        k: train_cfg.get(k)
        for k in [
            "augment",
            "optimizer",
            "lr0",
            "lrf",
            "mosaic",
            "mixup",
            "copy_paste",
            "fliplr",
            "flipud",
            "hsv_h",
            "hsv_s",
            "hsv_v",
            "degrees",
            "translate",
            "scale",
            "shear",
            "perspective",
        ]
        if k in train_cfg
    }

    seed = int((runtime_cfg.get("seed")) or 1234)
    runtime_train_dir = Path(str(runtime_cfg.get("train_runs_dir", "./runs_fl")))
    return SPCHMTrustConfig(
        enabled=bool(d.get("enabled", True)),
        proxy_data_yaml=str(d.get("proxy_data_yaml", "")),
        root_data_yaml=str(d.get("root_data_yaml", "")),
        proxy_max_images=int(d.get("proxy_max_images", 32)),
        proxy_conf=float(d.get("proxy_conf", 0.25)),
        proxy_imgsz=int(d.get("proxy_imgsz", d.get("root_imgsz", train_cfg.get("imgsz", 320)))),
        proxy_trigger=bool(d.get("proxy_trigger", False)),
        proxy_trigger_size=int(d.get("proxy_trigger_size", ((cfg.get("attack") or {}).get("backdoor") or {}).get("trigger_size", 40))),
        proxy_trigger_value=int(d.get("proxy_trigger_value", ((cfg.get("attack") or {}).get("backdoor") or {}).get("trigger_value", 255))),
        proxy_trigger_position=str(d.get("proxy_trigger_position", ((cfg.get("attack") or {}).get("backdoor") or {}).get("position", "bottom_right"))),
        proxy_trigger_mode=str(d.get("proxy_trigger_mode", "max")),
        tau=float(d.get("tau", 2.0)),
        eps=float(d.get("eps", 1e-8)),
        lambda_box=float(d.get("lambda_box", 1.0)),
        lambda_cls=float(d.get("lambda_cls", 1.0)),
        lambda_miss=float(d.get("lambda_miss", 1.0)),
        lambda_ghost=float(d.get("lambda_ghost", 1.0)),
        hungarian_class_penalty=float(d.get("hungarian_class_penalty", 0.5)),
        root_epochs=int(d.get("root_epochs", 1)),
        root_batch=int(d.get("root_batch", train_cfg.get("batch", 4))),
        root_num_workers=int(d.get("root_num_workers", train_cfg.get("num_workers", 0))),
        root_imgsz=int(d.get("root_imgsz", train_cfg.get("imgsz", 320))),
        root_device=str(d.get("root_device", train_cfg.get("device", "cpu"))),
        trust_floor=float(d.get("trust_floor", 0.0)),
        tmp_dir=str(d.get("tmp_dir", "./artifacts/tmp_spchm")),
        train_runs_dir=str(d.get("train_runs_dir", runtime_train_dir / "spchm_server_root")),
        seed=seed,
        train_overrides=train_overrides,
    )


class DeltaFedAvgStrategy(fl.server.strategy.FedAvg):
    """FedAvg strategy where clients send deltas and server updates global weights."""

    def __init__(self, cfg: Dict, round_stats_out: str = "", **kwargs):
        self.total_rounds = int(kwargs.pop("total_rounds", 0) or 0)
        super().__init__(**kwargs)
        self.cfg = cfg
        self.base_model = resolve_base_model_for_data(
            base_model_path=str(cfg["model"]["initial_weights"]),
            data_yaml=str(cfg["dataset"]["base_data_yaml"]),
            tmp_dir=str((cfg.get("runtime") or {}).get("tmp_dir", "./tmp")),
        )
        self.out_model = str(cfg["model"]["global_out"])
        self._round_global: Optional[List[np.ndarray]] = None
        self._round_stats_out = str(round_stats_out)
        self._dcfg = _load_defense_cfg(cfg)
        self._spchm_cfg: Optional[SPCHMTrustConfig] = _load_spchm_cfg(cfg)
        self._round_tracking_cfg = load_round_tracking_cfg(cfg)
        # Defense priority is handled in aggregate_fit: SPCHM-Trust, then detection-aware,
        # then the legacy gradient-only filter, then plain FedAvg.
        self._da_cfg: Optional[DetectionAwareDefenseConfig] = _load_detection_aware_cfg(cfg)

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        try:
            self._round_global = parameters_to_ndarrays(parameters)
        except Exception:
            self._round_global = None
        pairs = super().configure_fit(server_round, parameters, client_manager)
        for _, fit_ins in pairs:
            fit_ins.config["server_round"] = int(server_round)
        return pairs

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}
        if self._round_global is None:
            # Should not happen after configure_fit, but handle defensively
            self._round_global = get_parameters(self.base_model)

        # Flower returns deltas as Parameters in our client implementation.
        updates_delta = []
        metrics_rows = []
        for cp, fr in results:
            delta = parameters_to_ndarrays(fr.parameters)
            metrics = dict(fr.metrics) if fr.metrics else {}
            updates_delta.append((cp.cid, delta, fr.num_examples))
            metrics_rows.append({"cid": cp.cid, "num_examples": fr.num_examples, **metrics})
            logger.info("round=%s client=%s num_examples=%s metrics=%s", server_round, cp.cid, fr.num_examples, metrics)

        filtered = list(updates_delta)
        dmeta: Dict = {"removed_cids": [], "reason": "defense_disabled"}
        agg_delta = None
        mode = "fedavg"
        if self._spchm_cfg is not None and self._spchm_cfg.enabled:
            mode = "spchm_trust"
            dmeta = run_spchm_trust_round(
                updates=updates_delta,
                global_params=self._round_global,
                cfg=self._spchm_cfg,
                base_model_path=self.base_model,
                server_round=server_round,
            )
            agg_delta = dmeta["aggregated_delta"]
            logger.info(
                "round=%s spchm_trust clients=%s proxy_images=%s fallback_used=%s root_examples=%s",
                server_round,
                len(updates_delta),
                dmeta.get("proxy_num_images", 0),
                dmeta.get("fallback_used", False),
                dmeta.get("root_num_examples", 0),
            )
        elif self._da_cfg is not None and self._da_cfg.enabled:
            mode = "detection_aware"
            # Detection-aware defense: uses both gradient deltas and prediction stats.
            filtered, dmeta = detection_aware_filter(updates_delta, metrics_rows, self._da_cfg)
            logger.info(
                "round=%s detection_aware_defense removed clients=%s has_det_stats=%s",
                server_round,
                dmeta.get("removed_cids", []),
                dmeta.get("has_detection_stats", 0),
            )
        elif self._dcfg.enabled:
            mode = "robust_filter"
            filtered, dmeta = robust_filter(updates_delta, self._dcfg)
            logger.info("round=%s defense removed clients=%s", server_round, dmeta.get("removed_cids", []))

        # Aggregate deltas with weighted average.
        if agg_delta is None:
            agg_delta = weighted_fedavg(filtered)  # treat "weights" as deltas
        new_global = [np.asarray(g) + np.asarray(d) for g, d in zip(self._round_global, agg_delta)]

        # Save global model checkpoint (optional).
        out = Path(self.out_model)
        out.parent.mkdir(parents=True, exist_ok=True)
        set_parameters_to_model(self.base_model, new_global, str(out))
        if should_save_round_snapshot(
            server_round=int(server_round),
            total_rounds=int(self.total_rounds),
            tracking_cfg=self._round_tracking_cfg,
        ):
            snapshot = out.with_name(f"{out.stem}_round_{int(server_round):04d}{out.suffix}")
            set_parameters_to_model(self.base_model, new_global, str(snapshot))
            logger.info("round=%s snapshot saved at %s", server_round, snapshot)

        # Persist round stats (jsonl) if requested.
        if self._round_stats_out:
            p = Path(self._round_stats_out)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a", encoding="utf-8") as f:
                if mode == "spchm_trust":
                    for row in dmeta.get("client_diagnostics", []):
                        rec = {
                            "round": int(server_round),
                            "mode": mode,
                            "cid": row.get("cid", ""),
                            "num_examples": int(row.get("num_examples", 0)),
                            "s_i": float(row.get("s_i", 0.0)),
                            "d_box": float(row.get("d_box", 0.0)),
                            "d_cls": float(row.get("d_cls", 0.0)),
                            "r_miss": float(row.get("r_miss", 0.0)),
                            "r_ghost": float(row.get("r_ghost", 0.0)),
                            "z_i": float(row.get("z_i", 0.0)),
                            "cosine_root": float(row.get("cosine_root", 0.0)),
                            "trust_raw": float(row.get("trust_raw", 0.0)),
                            "trust_weight": float(row.get("trust_weight", 0.0)),
                            "fallback_used": bool(row.get("fallback_used", False)),
                            "s_clean": float(row.get("s_clean", row.get("s_i", 0.0))),
                            "s_trigger": float(row.get("s_trigger", row.get("s_i", 0.0))),
                            "score_median": float(dmeta.get("score_median", 0.0)),
                            "score_mad": float(dmeta.get("score_mad", 0.0)),
                            "proxy_num_images": int(dmeta.get("proxy_num_images", 0)),
                            "proxy_trigger": bool(dmeta.get("proxy_trigger", False)),
                            "root_num_examples": int(dmeta.get("root_num_examples", 0)),
                            "root_checkpoint": str(dmeta.get("root_checkpoint", "")),
                        }
                        f.write(json.dumps(rec) + "\n")
                else:
                    rec = {
                        "round": int(server_round),
                        "mode": mode,
                        "removed_cids": dmeta.get("removed_cids", []),
                        "defense_reason": dmeta.get("reason", ""),
                        "num_results": int(len(results)),
                        "num_kept": int(len(filtered)),
                        "client_metrics": metrics_rows,
                    }
                    f.write(json.dumps(rec) + "\n")

        # Update cached global for next round.
        self._round_global = new_global

        kept_clients = len(updates_delta) if mode == "spchm_trust" else len(filtered)
        return ndarrays_to_parameters(new_global), {"kept_clients": int(kept_clients), "removed_clients": int(len(dmeta.get("removed_cids", [])))}


def run_server(host: str, port: int, rounds: int, cfg_path: str, expected_clients: int = 0, round_stats_out: str = "") -> None:
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    seed = int(((cfg.get("runtime") or {}).get("seed")) or 1234)
    random.seed(seed)
    np.random.seed(seed)

    if round_stats_out:
        stats_path = Path(round_stats_out)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        if stats_path.exists():
            stats_path.unlink()

    base_model = resolve_base_model_for_data(
        base_model_path=str(cfg["model"]["initial_weights"]),
        data_yaml=str(cfg["dataset"]["base_data_yaml"]),
        tmp_dir=str((cfg.get("runtime") or {}).get("tmp_dir", "./tmp")),
    )
    init_params = ndarrays_to_parameters(get_parameters(base_model))

    min_fit = int(cfg["federated"]["min_fit_clients"])
    min_avail = int(cfg["federated"]["min_available_clients"])
    if expected_clients and expected_clients > 0:
        # expected_clients comes from the launcher (run_experiment.py) and represents how many
        # clients we *attempted* to start. Do not force min_* to expected_clients, otherwise
        # a single crashed/slow client can stall the entire run indefinitely (common on Colab).
        # Instead, cap the configured minima so they never exceed the expected count.
        min_fit = min(min_fit, expected_clients)
        min_avail = min(min_avail, expected_clients)

    strategy = DeltaFedAvgStrategy(
        cfg=cfg,
        round_stats_out=round_stats_out,
        total_rounds=int(rounds),
        fraction_fit=1.0,
        min_fit_clients=min_fit,
        min_available_clients=min_avail,
        initial_parameters=init_params,
    )

    fl.server.start_server(
        server_address=f"{host}:{port}",
        config=fl.server.ServerConfig(num_rounds=int(rounds)),
        strategy=strategy,
    )


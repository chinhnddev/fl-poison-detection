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
from defense import DefenseConfig, DetectionAwareDefenseConfig, detection_aware_filter, robust_filter
from train_yolo import get_parameters, set_parameters_to_model

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


class DeltaFedAvgStrategy(fl.server.strategy.FedAvg):
    """FedAvg strategy where clients send deltas and server updates global weights."""

    def __init__(self, cfg: Dict, round_stats_out: str = "", **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.base_model = str(cfg["model"]["initial_weights"])
        self.out_model = str(cfg["model"]["global_out"])
        self._round_global: Optional[List[np.ndarray]] = None
        self._round_stats_out = str(round_stats_out)
        self._dcfg = _load_defense_cfg(cfg)
        # Detection-aware defense takes priority over gradient-only defense when configured.
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
            updates_delta.append((cp.cid, delta, fr.num_examples))
            metrics_rows.append({"cid": cp.cid, "num_examples": fr.num_examples, **(dict(fr.metrics) if fr.metrics else {})})
            logger.info("round=%s client=%s num_examples=%s metrics=%s", server_round, cp.cid, fr.num_examples, dict(fr.metrics))

        filtered = list(updates_delta)
        dmeta: Dict = {"removed_cids": [], "reason": "defense_disabled"}
        if self._da_cfg is not None and self._da_cfg.enabled:
            # Detection-aware defense: uses both gradient deltas and prediction stats.
            filtered, dmeta = detection_aware_filter(updates_delta, metrics_rows, self._da_cfg)
            logger.info(
                "round=%s detection_aware_defense removed clients=%s has_det_stats=%s",
                server_round,
                dmeta.get("removed_cids", []),
                dmeta.get("has_detection_stats", 0),
            )
        elif self._dcfg.enabled:
            filtered, dmeta = robust_filter(updates_delta, self._dcfg)
            logger.info("round=%s defense removed clients=%s", server_round, dmeta.get("removed_cids", []))

        # Aggregate deltas with weighted average.
        agg_delta = weighted_fedavg(filtered)  # treat "weights" as deltas
        new_global = [np.asarray(g) + np.asarray(d) for g, d in zip(self._round_global, agg_delta)]

        # Save global model checkpoint (optional).
        out = Path(self.out_model)
        out.parent.mkdir(parents=True, exist_ok=True)
        set_parameters_to_model(self.base_model, new_global, str(out))

        # Persist round stats (jsonl) if requested.
        if self._round_stats_out:
            p = Path(self._round_stats_out)
            p.parent.mkdir(parents=True, exist_ok=True)
            rec = {
                "round": int(server_round),
                "removed_cids": dmeta.get("removed_cids", []),
                "defense_reason": dmeta.get("reason", ""),
                "num_results": int(len(results)),
                "num_kept": int(len(filtered)),
                "client_metrics": metrics_rows,
            }
            with open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

        # Update cached global for next round.
        self._round_global = new_global

        return ndarrays_to_parameters(new_global), {"kept_clients": int(len(filtered)), "removed_clients": int(len(dmeta.get("removed_cids", [])))}


def run_server(host: str, port: int, rounds: int, cfg_path: str, expected_clients: int = 0, round_stats_out: str = "") -> None:
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    seed = int(((cfg.get("runtime") or {}).get("seed")) or 1234)
    random.seed(seed)
    np.random.seed(seed)

    init_params = ndarrays_to_parameters(get_parameters(cfg["model"]["initial_weights"]))

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


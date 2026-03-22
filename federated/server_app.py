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
from defense import DefenseConfig, clip_layer_norms, robust_filter
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
        layer_aware=bool(d.get("layer_aware", True)),
        head_layer_fraction=float(d.get("head_layer_fraction", 0.2)),
        head_weight_multiplier=float(d.get("head_weight_multiplier", 2.0)),
        clip_norm=bool(d.get("clip_norm", False)),
        clip_norm_multiplier=float(d.get("clip_norm_multiplier", 5.0)),
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
        if self._dcfg.enabled:
            # Per-layer norm clipping: bound each client's per-layer delta to a
            # multiple of the per-layer median norm, preventing layer-targeted
            # perception poisoning from injecting disproportionately large updates
            # into specific sub-networks (e.g. the detection head).
            if self._dcfg.clip_norm:
                filtered = clip_layer_norms(filtered, self._dcfg.clip_norm_multiplier)
                logger.info("round=%s applied per-layer norm clipping (multiplier=%.1f)", server_round, self._dcfg.clip_norm_multiplier)
            filtered, dmeta = robust_filter(filtered, self._dcfg)
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
        min_fit = max(min_fit, expected_clients)
        min_avail = max(min_avail, expected_clients)

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


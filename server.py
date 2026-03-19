import argparse
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import yaml
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy

from aggregation import weighted_fedavg, krum
from defense import DefenseConfig, filter_suspicious
from train_yolo import get_parameters, set_parameters_to_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("server")


class CustomStrategy(fl.server.strategy.FedAvg):
    def __init__(self, cfg: dict, aggregation: str, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.aggregation = aggregation
        self.base_model = cfg["model"]["initial_weights"]
        self.out_model = cfg["model"]["global_out"]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        updates = []
        for cp, fr in results:
            nds = parameters_to_ndarrays(fr.parameters)
            logger.info(
                "round=%s client=%s num_examples=%s metrics=%s",
                server_round, cp.cid, fr.num_examples, dict(fr.metrics)
            )
            updates.append((cp.cid, nds, fr.num_examples))

        dcfg = DefenseConfig(
            enabled=self.cfg["defense"]["enabled"],
            cosine_z=self.cfg["defense"]["cosine_z"],
            norm_z=self.cfg["defense"]["norm_z"],
            min_votes=self.cfg["defense"]["min_votes"],
        )
        filtered, dmeta = filter_suspicious(updates, dcfg)
        logger.info("round=%s defense removed clients=%s", server_round, dmeta["removed"])

        if self.aggregation == "krum":
            agg = krum(filtered, byzantine_count=self.cfg["aggregation"]["krum_byzantine"])
        else:
            agg = weighted_fedavg(filtered)

        # save current global model
        out = Path(self.out_model)
        out.parent.mkdir(parents=True, exist_ok=True)
        set_parameters_to_model(self.base_model, agg, str(out))

        return ndarrays_to_parameters(agg), {"kept_clients": len(filtered), "removed_clients": len(dmeta["removed"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--aggregation", choices=["fedavg", "krum"], default="fedavg")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--expected_clients", type=int, default=0, help="If set, wait for this many clients per round")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    seed = int(((cfg.get("runtime") or {}).get("seed")) or 1234)
    random.seed(seed)
    np.random.seed(seed)

    init_params = ndarrays_to_parameters(get_parameters(cfg["model"]["initial_weights"]))

    min_fit = int(cfg["federated"]["min_fit_clients"])
    min_avail = int(cfg["federated"]["min_available_clients"])
    if args.expected_clients and args.expected_clients > 0:
        min_fit = max(min_fit, args.expected_clients)
        min_avail = max(min_avail, args.expected_clients)

    strategy = CustomStrategy(
        cfg=cfg,
        aggregation=args.aggregation,
        fraction_fit=1.0,
        min_fit_clients=min_fit,
        min_available_clients=min_avail,
        initial_parameters=init_params,
    )

    fl.server.start_server(
        server_address=f"{args.host}:{args.port}",
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()

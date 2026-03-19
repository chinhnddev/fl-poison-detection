import argparse
from pathlib import Path
import yaml
import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from attack import LabelFlipConfig, ModelPoisonConfig, build_poisoned_shard_label_flip, poison_model_update
from train_yolo import get_parameters, set_parameters_to_model, set_global_seed, train_local


class YoloClient(fl.client.NumPyClient):
    def __init__(self, cid: int, cfg: dict, malicious: bool):
        self.cid = cid
        self.cfg = cfg
        self.malicious = malicious
        seed = int(((cfg.get("runtime") or {}).get("seed")) or 1234)
        set_global_seed(seed + int(cid))

        fed_dir = Path(cfg["federated"]["data_dir"])
        shard_yaml = fed_dir / f"client_{cid}" / "data.yaml"
        if not shard_yaml.exists():
            raise FileNotFoundError(
                f"Missing client shard: {shard_yaml}. "
                "Run data partitioning before starting clients."
            )
        self.data_yaml = str(shard_yaml.resolve())

        # Label-flip poisoning (malicious clients only): create poisoned view of shard once.
        a = cfg.get("attack") or {}
        lf = (a.get("label_flip") or {})
        lf_cfg = LabelFlipConfig(
            enabled=bool(lf.get("enabled", False)),
            src_class_id=int(lf.get("src_class_id", 0)),
            dst_class_id=int(lf.get("dst_class_id", 16)),
            prob=float(lf.get("prob", 1.0)),
            seed=int(lf.get("seed", seed)) + int(cid),
        )
        if self.malicious and lf_cfg.enabled:
            out_dir = Path(cfg["runtime"]["tmp_dir"]) / "poison" / f"client_{cid}_label_flip"
            out_yaml = out_dir / "data.yaml"
            if not out_yaml.exists():
                self.data_yaml = build_poisoned_shard_label_flip(self.data_yaml, str(out_dir), lf_cfg)
            else:
                self.data_yaml = str(out_yaml.resolve())

        mp = (a.get("model_poison") or {})
        self.model_poison_cfg = ModelPoisonConfig(
            enabled=bool(mp.get("enabled", False)),
            strength=float(mp.get("strength", 1.0)),
            mode=str(mp.get("mode", "signflip")),
        )
        self.base_model = cfg["model"]["initial_weights"]

    def get_parameters(self, config):
        return get_parameters(self.base_model)

    def fit(self, parameters, config):
        global_params = parameters
        tmp_model = Path(self.cfg["runtime"]["tmp_dir"]) / f"client_{self.cid}_round_model.pt"
        tmp_model.parent.mkdir(parents=True, exist_ok=True)

        # load global parameters into local model
        set_parameters_to_model(self.base_model, parameters, str(tmp_model))

        params, n, metrics = train_local(
            model_path=str(tmp_model),
            data_yaml=self.data_yaml,
            epochs=self.cfg["train"]["local_epochs"],
            imgsz=self.cfg["train"]["imgsz"],
            batch=self.cfg["train"]["batch"],
            device=self.cfg["train"]["device"],
            project=self.cfg["runtime"]["train_runs_dir"],
            name=f"client_{self.cid}",
        )

        # Model poisoning (malicious clients only) in update space.
        if self.malicious and self.model_poison_cfg.enabled:
            params = poison_model_update(global_params, params, self.model_poison_cfg)

        metrics["malicious"] = int(self.malicious)
        return params, n, metrics

    def evaluate(self, parameters, config):
        # Minimal: avoid expensive YOLO validation in smoke/loop mode, but make sure
        # Flower can aggregate evaluation (num_examples must be > 0).
        return 0.0, 1, {"cid": self.cid, "malicious": int(self.malicious)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cid", type=int, required=True)
    ap.add_argument("--server_address", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--malicious", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    client = YoloClient(args.cid, cfg, bool(args.malicious))
    fl.client.start_numpy_client(server_address=args.server_address, client=client)


if __name__ == "__main__":
    main()

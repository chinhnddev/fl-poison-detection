from __future__ import annotations

import logging
import time
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import flwr as fl
import numpy as np
import yaml

from attack import (
    BackdoorConfig,
    BBoxDistortionConfig,
    LabelFlipConfig,
    ModelPoisonConfig,
    ObjectRemovalConfig,
    build_poisoned_dataset,
    poison_delta,
)
from train_yolo import get_parameters, set_global_seed, set_parameters_to_model, train_local

NDArrays = List[np.ndarray]


def _load_attack_cfg(cfg: Dict, cid: int) -> tuple[LabelFlipConfig, BBoxDistortionConfig, ObjectRemovalConfig, BackdoorConfig, ModelPoisonConfig]:
    a = cfg.get("attack") or {}

    lf = a.get("label_flip") or {}
    label_flip = LabelFlipConfig(
        enabled=bool(lf.get("enabled", False)),
        poison_ratio=float(lf.get("poison_ratio", 0.2)),
        src_class_id=int(lf.get("src_class_id", 0)),
        dst_class_id=int(lf.get("dst_class_id", 56)),
        prob=float(lf.get("prob", 0.5)),
        seed=int(lf.get("seed", 42)) + int(cid),
    )

    bb = a.get("bbox_distortion") or {}
    bbox = BBoxDistortionConfig(
        enabled=bool(bb.get("enabled", False)),
        poison_ratio=float(bb.get("poison_ratio", 0.2)),
        shift_xy=float(bb.get("shift_xy", bb.get("shift", 0.05))),
        shift_wh=float(bb.get("shift_wh", 0.03)),
        prob=float(bb.get("prob", 0.3)),
        seed=int(bb.get("seed", 42)) + int(cid),
    )

    rm = a.get("object_removal") or {}
    removal = ObjectRemovalConfig(
        enabled=bool(rm.get("enabled", False)),
        poison_ratio=float(rm.get("poison_ratio", 0.2)),
        target_class_id=int(rm.get("target_class_id", rm.get("target_class", 0))),
        prob=float(rm.get("prob", 0.3)),
        seed=int(rm.get("seed", 42)) + int(cid),
    )

    bd = a.get("backdoor") or {}
    backdoor = BackdoorConfig(
        enabled=bool(bd.get("enabled", False)),
        poison_ratio=float(bd.get("poison_ratio", 0.2)),
        trigger_size=int(bd.get("trigger_size", 16)),
        trigger_value=int(bd.get("trigger_value", 255)),
        position=str(bd.get("position", "bottom_right")),
        src_class_id=int(bd.get("src_class_id", bd.get("src_class", 0))),
        target_class_id=int(bd.get("target_class_id", bd.get("target_class", 56))),
        prob=float(bd.get("prob", 1.0)),
        seed=int(bd.get("seed", 42)) + int(cid),
    )

    mp = a.get("model_poison") or {}
    model_poison = ModelPoisonConfig(
        enabled=bool(mp.get("enabled", False)),
        mode=str(mp.get("mode", "stealth")),
        strength=float(mp.get("strength", 1.0)),
        stealth_beta=float(mp.get("stealth_beta", 0.1)),
        max_scale=float(mp.get("max_scale", 2.0)),
        seed=int(mp.get("seed", 42)) + int(cid),
    )

    return label_flip, bbox, removal, backdoor, model_poison


class YoloDeltaClient(fl.client.NumPyClient):
    """Flower client that returns *delta updates* instead of raw weights."""

    def __init__(self, cid: int, cfg: Dict, malicious: bool):
        self.cid = int(cid)
        self.cfg = cfg
        self.malicious = bool(malicious)

        seed = int(((cfg.get("runtime") or {}).get("seed")) or 1234)
        set_global_seed(seed + self.cid)

        fed_dir = Path(cfg["federated"]["data_dir"])
        shard_yaml = fed_dir / f"client_{self.cid}" / "data.yaml"
        if not shard_yaml.exists():
            raise FileNotFoundError(f"Missing client shard: {shard_yaml}")
        self.data_yaml = str(shard_yaml.resolve())

        self.label_flip, self.bbox, self.removal, self.backdoor, self.model_poison = _load_attack_cfg(cfg, self.cid)

        # Dataset poisoning for malicious clients (build once; reuse).
        data_poison_enabled = any(
            [
                self.label_flip.enabled,
                self.bbox.enabled,
                self.removal.enabled,
                self.backdoor.enabled,
            ]
        )
        if self.malicious and data_poison_enabled:
            # IMPORTANT: poison artifacts must be tied to the exact attack config.
            # Otherwise changing YAML would silently reuse stale poisoned shards.
            sig_obj = {
                "label_flip": vars(self.label_flip),
                "bbox_distortion": vars(self.bbox),
                "object_removal": vars(self.removal),
                "backdoor": vars(self.backdoor),
            }
            sig = hashlib.md5(json.dumps(sig_obj, sort_keys=True).encode("utf-8")).hexdigest()[:10]
            out_dir = Path(cfg["runtime"]["tmp_dir"]) / "poison" / f"client_{self.cid}_{sig}"
            out_yaml = out_dir / "data.yaml"
            if not out_yaml.exists():
                self.data_yaml = build_poisoned_dataset(
                    shard_data_yaml=self.data_yaml,
                    out_root=str(out_dir),
                    label_flip=self.label_flip,
                    bbox=self.bbox,
                    removal=self.removal,
                    backdoor=self.backdoor,
                )
            else:
                self.data_yaml = str(out_yaml.resolve())

            meta_p = out_dir / "poison_meta.yaml"
            if meta_p.exists():
                try:
                    meta = yaml.safe_load(open(meta_p, "r", encoding="utf-8")) or {}
                    logging.getLogger("client").info(
                        "poison_ready cid=%s candidates_backdoor=%s poisoned_images_backdoor=%s backdoor_flipped=%s poisoned_images_label_flip=%s flipped=%s poisoned_images_any=%s",
                        self.cid,
                        meta.get("candidates_backdoor", meta.get("candidates_bd", "?")),
                        meta.get("poisoned_images_backdoor", "?"),
                        meta.get("backdoor_flipped", "?"),
                        meta.get("poisoned_images_label_flip", "?"),
                        meta.get("flipped", "?"),
                        meta.get("poisoned_images_any", "?"),
                    )
                except Exception:
                    pass

        self.base_model = str(cfg["model"]["initial_weights"])
        self.device = str(cfg["train"]["device"])

        logging.getLogger("client").info(
            "cid=%s malicious=%s shard=%s local_epochs=%s imgsz=%s batch=%s device=%s",
            self.cid,
            int(self.malicious),
            self.data_yaml,
            cfg["train"]["local_epochs"],
            cfg["train"]["imgsz"],
            cfg["train"]["batch"],
            self.device,
        )

    def get_parameters(self, config):
        return get_parameters(self.base_model)

    def fit(self, parameters: NDArrays, config):
        logger = logging.getLogger("client")
        server_round = int(config.get("server_round", -1))

        tmp_model = Path(self.cfg["runtime"]["tmp_dir"]) / f"client_{self.cid}_round_{server_round}.pt"
        tmp_model.parent.mkdir(parents=True, exist_ok=True)

        # Set global weights into a local checkpoint.
        set_parameters_to_model(self.base_model, parameters, str(tmp_model))

        t0 = time.time()
        logger.info("FIT_START cid=%s round=%s malicious=%s", self.cid, server_round, int(self.malicious))

        local_params, n, metrics = train_local(
            model_path=str(tmp_model),
            data_yaml=self.data_yaml,
            epochs=int(self.cfg["train"]["local_epochs"]),
            imgsz=int(self.cfg["train"]["imgsz"]),
            batch=int(self.cfg["train"]["batch"]),
            device=self.device,
            project=str(self.cfg["runtime"]["train_runs_dir"]),
            name=f"client_{self.cid}",
            seed=int(((self.cfg.get("runtime") or {}).get("seed") or 1234)) + self.cid + 1000 * max(server_round, 0),
        )

        # delta = local - global
        delta: NDArrays = [np.asarray(lp) - np.asarray(gp) for lp, gp in zip(local_params, parameters)]

        # Model poisoning operates on delta, malicious clients only.
        if self.malicious and self.model_poison.enabled:
            poison_seed = int(self.model_poison.seed) + 1000 * max(server_round, 0)
            delta = poison_delta(delta, self.model_poison, seed=poison_seed)
            logger.info(
                "MODEL_POISON cid=%s round=%s mode=%s strength=%.3f",
                self.cid,
                server_round,
                self.model_poison.mode,
                float(self.model_poison.strength),
            )

        metrics = dict(metrics)
        metrics["malicious"] = int(self.malicious)
        logger.info("FIT_END cid=%s round=%s seconds=%.2f num_examples=%s", self.cid, server_round, time.time() - t0, n)
        return delta, int(n), metrics

    def evaluate(self, parameters, config):
        # Keep Flower evaluation minimal; use evaluation/ scripts for real metrics.
        return 0.0, 1, {"cid": self.cid, "malicious": int(self.malicious)}

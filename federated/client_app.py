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
from train_yolo import collect_detection_stats, get_parameters, set_global_seed, set_parameters_to_model, train_local

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
        oversample_factor=int(bd.get("oversample_factor", 1)),
        trigger_size=int(bd.get("trigger_size", 16)),
        trigger_value=int(bd.get("trigger_value", 255)),
        position=str(bd.get("position", "bottom_right")),
        src_class_id=int(bd.get("src_class_id", bd.get("src_class", 0))),
        target_class_id=int(bd.get("target_class_id", bd.get("target_class", 44))),
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


def _normalize_client_shard_yaml(shard_yaml: Path) -> str:
    """Rewrite client shard YAMLs into a portable, machine-local form."""
    client_dir = shard_yaml.parent
    train_dir = client_dir / "images" / "train"
    val_dir = client_dir / "images" / "val"
    if not train_dir.exists() or not val_dir.exists():
        return str(shard_yaml.resolve())

    cfg = yaml.safe_load(shard_yaml.read_text(encoding="utf-8")) or {}
    portable_cfg = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
    }
    for key, value in cfg.items():
        if key not in {"path", "train", "val"}:
            portable_cfg[key] = value

    if cfg != portable_cfg:
        shard_yaml.write_text(yaml.safe_dump(portable_cfg, sort_keys=False), encoding="utf-8")
        logging.getLogger("client").info("normalized_client_yaml shard=%s", shard_yaml.resolve())

    return str(shard_yaml.resolve())


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
        self.data_yaml = _normalize_client_shard_yaml(shard_yaml)

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
                        "poison_ready cid=%s candidates_backdoor=%s poisoned_images_backdoor=%s poisoned_images_backdoor_replayed=%s backdoor_flipped=%s poisoned_images_label_flip=%s flipped=%s poisoned_images_any=%s",
                        self.cid,
                        meta.get("candidates_backdoor", meta.get("candidates_bd", "?")),
                        meta.get("poisoned_images_backdoor", "?"),
                        meta.get("poisoned_images_backdoor_replayed", 0),
                        meta.get("backdoor_flipped", "?"),
                        meta.get("poisoned_images_label_flip", "?"),
                        meta.get("flipped", "?"),
                        meta.get("poisoned_images_any", "?"),
                    )
                except Exception:
                    pass

        self.base_model = str(cfg["model"]["initial_weights"])
        self.device = str(cfg["train"]["device"])
        self.local_epochs = int(cfg["train"]["local_epochs"])
        self.malicious_local_epochs = int((cfg.get("train") or {}).get("malicious_local_epochs", self.local_epochs))
        self.effective_local_epochs = self.malicious_local_epochs if self.malicious else self.local_epochs
        self.num_workers = int((cfg.get("train") or {}).get("num_workers", 0))

        # Detection-aware defense: collect prediction statistics after local training.
        # Controlled by defense.collect_detection_stats in the YAML config.
        defense_cfg = cfg.get("defense") or {}
        self._spchm_trust = bool(defense_cfg.get("spchm_trust", False))
        self._collect_det_stats = bool(defense_cfg.get("collect_detection_stats", False)) and not self._spchm_trust
        self._det_stats_max_images = int(defense_cfg.get("det_stats_max_images", 50))
        self._det_stats_conf = float(defense_cfg.get("det_stats_conf", 0.25))
        self._det_stats_trigger = bool(defense_cfg.get("det_stats_trigger", False))

        backdoor_cfg = ((cfg.get("attack") or {}).get("backdoor") or {})
        self._det_stats_trigger_size = int(
            defense_cfg.get("det_stats_trigger_size", backdoor_cfg.get("trigger_size", 40))
        )
        self._det_stats_trigger_value = int(
            defense_cfg.get("det_stats_trigger_value", backdoor_cfg.get("trigger_value", 255))
        )
        self._det_stats_trigger_position = str(
            defense_cfg.get("det_stats_trigger_position", backdoor_cfg.get("position", "bottom_right"))
        )

        logging.getLogger("client").info(
            "cid=%s malicious=%s shard=%s local_epochs=%s imgsz=%s batch=%s num_workers=%s device=%s",
            self.cid,
            int(self.malicious),
            self.data_yaml,
            self.effective_local_epochs,
            cfg["train"]["imgsz"],
            cfg["train"]["batch"],
            self.num_workers,
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
            epochs=int(self.effective_local_epochs),
            imgsz=int(self.cfg["train"]["imgsz"]),
            batch=int(self.cfg["train"]["batch"]),
            num_workers=int(self.num_workers),
            device=self.device,
            project=str(self.cfg["runtime"]["train_runs_dir"]),
            name=f"client_{self.cid}",
            seed=int(((self.cfg.get("runtime") or {}).get("seed") or 1234)) + self.cid + 1000 * max(server_round, 0),
            train_overrides={
                # Backdoor triggers are spatial patterns; heavy aug (mosaic/crop/flip) can destroy them.
                # Keep these configurable via YAML; default behavior stays Ultralytics defaults unless set.
                k: (self.cfg.get("train") or {}).get(k)
                for k in [
                    "augment",
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
                if k in (self.cfg.get("train") or {})
            },
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

        # ── Detection-aware defense: collect prediction stats ─────────────────
        if self._collect_det_stats:
            ckpt_path = metrics.pop("_ckpt_path", "")
            if ckpt_path and Path(ckpt_path).exists():
                try:
                    det_tmp_dir = (
                        Path(self.cfg["runtime"]["tmp_dir"])
                        / "det_stats_triggered"
                        / f"cid_{self.cid}"
                        / f"round_{server_round}"
                    )
                    det_json = collect_detection_stats(
                        model_path=ckpt_path,
                        val_yaml=self.data_yaml,
                        imgsz=int(self.cfg["train"]["imgsz"]),
                        device=self.device,
                        max_images=self._det_stats_max_images,
                        conf=self._det_stats_conf,
                        global_model_path=str(tmp_model),
                        trigger=bool(self._det_stats_trigger),
                        trigger_size=int(self._det_stats_trigger_size),
                        trigger_value=int(self._det_stats_trigger_value),
                        trigger_position=str(self._det_stats_trigger_position),
                        trigger_tmp_dir=str(det_tmp_dir),
                    )
                    if det_json:
                        metrics["detection_stats"] = det_json
                        logger.info("DET_STATS cid=%s round=%s len=%s", self.cid, server_round, len(det_json))
                except Exception as exc:
                    logger.warning("DET_STATS_ERROR cid=%s round=%s err=%s", self.cid, server_round, exc)
            else:
                # Remove internal key even if stats not collected
                metrics.pop("_ckpt_path", None)
        else:
            metrics.pop("_ckpt_path", None)

        logger.info("FIT_END cid=%s round=%s seconds=%.2f num_examples=%s", self.cid, server_round, time.time() - t0, n)
        return delta, int(n), metrics

    def evaluate(self, parameters, config):
        # Keep Flower evaluation minimal; use evaluation/ scripts for real metrics.
        return 0.0, 1, {"cid": self.cid, "malicious": int(self.malicious)}

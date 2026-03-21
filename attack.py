"""Backward-compatible wrapper for attack module."""

from attack.config import (
    BackdoorConfig,
    BBoxDistortionConfig,
    LabelFlipConfig,
    ModelPoisonConfig,
    ObjectRemovalConfig,
)
from attack.data_poison import build_poisoned_dataset
from attack.model_poison import poison_delta


def build_poisoned_shard_label_flip(shard_data_yaml: str, out_root: str, cfg: LabelFlipConfig) -> str:
    """Legacy wrapper: label flip only."""
    return build_poisoned_dataset(
        shard_data_yaml=shard_data_yaml,
        out_root=out_root,
        label_flip=cfg,
        bbox=BBoxDistortionConfig(enabled=False),
        removal=ObjectRemovalConfig(enabled=False),
        backdoor=BackdoorConfig(enabled=False),
    )


def poison_model_update(global_params, trained_params, cfg: ModelPoisonConfig):
    """Legacy wrapper: return poisoned *local weights* by poisoning delta then adding back to global."""
    delta = [lp - gp for lp, gp in zip(trained_params, global_params)]
    pd = poison_delta(delta, cfg, seed=cfg.seed)
    return [gp + d for gp, d in zip(global_params, pd)]


__all__ = [
    "BackdoorConfig",
    "BBoxDistortionConfig",
    "LabelFlipConfig",
    "ModelPoisonConfig",
    "ObjectRemovalConfig",
    "build_poisoned_dataset",
    "build_poisoned_shard_label_flip",
    "poison_delta",
    "poison_model_update",
]


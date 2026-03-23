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

__all__ = [
    "BackdoorConfig",
    "BBoxDistortionConfig",
    "LabelFlipConfig",
    "ModelPoisonConfig",
    "ObjectRemovalConfig",
    "build_poisoned_dataset",
    "poison_delta",
]


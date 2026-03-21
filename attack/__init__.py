"""Attacks for FL security experiments (YOLOv8 object detection).

All attacks operate on *client updates* (delta = local - global) or on
client-local datasets (data poisoning/backdoor).
"""

from .config import (
    BackdoorConfig,
    BBoxDistortionConfig,
    LabelFlipConfig,
    ModelPoisonConfig,
    ObjectRemovalConfig,
)
from .data_poison import build_poisoned_dataset
from .model_poison import poison_delta

__all__ = [
    "BackdoorConfig",
    "BBoxDistortionConfig",
    "LabelFlipConfig",
    "ModelPoisonConfig",
    "ObjectRemovalConfig",
    "build_poisoned_dataset",
    "poison_delta",
]


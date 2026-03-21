from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LabelFlipConfig:
    """Flip src->dst labels on a subset of images/objects."""

    enabled: bool = False
    poison_ratio: float = 0.2  # fraction of images to poison
    src_class_id: int = 0
    dst_class_id: int = 56
    prob: float = 0.5  # per-object probability, given the image is poisoned
    seed: int = 42


@dataclass
class BBoxDistortionConfig:
    """Slightly perturb bbox coordinates (stealthy label noise)."""

    enabled: bool = False
    poison_ratio: float = 0.2
    shift_xy: float = 0.05
    shift_wh: float = 0.03
    prob: float = 0.3  # per-object probability, given the image is poisoned
    seed: int = 42


@dataclass
class ObjectRemovalConfig:
    """Remove objects of a target class (label dropping)."""

    enabled: bool = False
    poison_ratio: float = 0.2
    target_class_id: int = 0
    prob: float = 0.3  # per-object probability, given the image is poisoned
    seed: int = 42


@dataclass
class BackdoorConfig:
    """Trigger-based backdoor (patch + conditional label flip on poisoned images)."""

    enabled: bool = False
    poison_ratio: float = 0.2
    trigger_size: int = 16
    trigger_value: int = 255  # white patch
    position: str = "bottom_right"  # bottom_right|bottom_left|top_right|top_left
    src_class_id: int = 0
    target_class_id: int = 56
    prob: float = 1.0  # per-object probability for conditional flip (given image poisoned)
    seed: int = 42


@dataclass
class ModelPoisonConfig:
    """Update-space poisoning (operates on delta arrays).

    mode:
      - signflip:  delta' = -strength * delta
      - scale:     delta' =  strength * delta
      - stealth:   preserves norm and keeps cosine(delta, delta') high by injecting orthogonal noise
    """

    enabled: bool = False
    mode: str = "stealth"
    strength: float = 1.0
    stealth_beta: float = 0.1  # orthogonal noise magnitude, in [0,1)
    max_scale: float = 2.0  # clip delta' to +/- max_scale * ||delta||
    seed: int = 42


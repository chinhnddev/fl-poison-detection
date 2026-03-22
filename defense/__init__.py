"""Defenses for FL security experiments (operate on delta updates)."""

from .detection_aware_filter import DetectionAwareDefenseConfig, detection_aware_filter, parse_detection_stats
from .robust_filter import DefenseConfig, robust_filter

__all__ = [
    "DefenseConfig",
    "DetectionAwareDefenseConfig",
    "detection_aware_filter",
    "parse_detection_stats",
    "robust_filter",
]


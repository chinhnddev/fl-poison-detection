"""Backward-compatible wrapper for defense module."""

from defense.detection_aware_filter import DetectionAwareDefenseConfig, detection_aware_filter, parse_detection_stats
from defense.robust_filter import DefenseConfig, robust_filter

__all__ = [
    "DefenseConfig",
    "DetectionAwareDefenseConfig",
    "detection_aware_filter",
    "parse_detection_stats",
    "robust_filter",
]


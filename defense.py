"""Backward-compatible wrapper for defense module."""

from defense.detection_aware_filter import DetectionAwareDefenseConfig, detection_aware_filter, parse_detection_stats
from defense.robust_filter import DefenseConfig, robust_filter
from defense.spchm_trust import SPCHMTrustConfig, run_spchm_trust_round

__all__ = [
    "DefenseConfig",
    "DetectionAwareDefenseConfig",
    "SPCHMTrustConfig",
    "detection_aware_filter",
    "parse_detection_stats",
    "robust_filter",
    "run_spchm_trust_round",
]


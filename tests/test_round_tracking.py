from __future__ import annotations

import unittest

from evaluation.round_tracking import (
    RoundMetricRow,
    should_save_round_snapshot,
    summarize_round_metrics,
)


class RoundTrackingTests(unittest.TestCase):
    def test_summary_finds_best_and_plateau_round(self) -> None:
        rows = [
            RoundMetricRow(round=1, model_path="r1.pt", map50=0.20, map5095=0.10),
            RoundMetricRow(round=2, model_path="r2.pt", map50=0.24, map5095=0.15),
            RoundMetricRow(round=3, model_path="r3.pt", map50=0.245, map5095=0.151),
            RoundMetricRow(round=4, model_path="r4.pt", map50=0.246, map5095=0.1514),
            RoundMetricRow(round=5, model_path="r5.pt", map50=0.2462, map5095=0.1512),
            RoundMetricRow(round=6, model_path="r6.pt", map50=0.2461, map5095=0.1511),
        ]

        out = summarize_round_metrics(rows, selection_metric="map5095", patience=3, min_delta=0.002)

        self.assertEqual(out["best_round"], 4)
        self.assertAlmostEqual(out["best_value"], 0.1514, places=6)
        self.assertEqual(out["convergence_round"], 2)

    def test_snapshot_policy_honors_frequency_and_final_round(self) -> None:
        cfg = {"enabled": True, "every_n_rounds": 5, "keep_final": True}

        self.assertFalse(should_save_round_snapshot(server_round=1, total_rounds=12, tracking_cfg=cfg))
        self.assertTrue(should_save_round_snapshot(server_round=5, total_rounds=12, tracking_cfg=cfg))
        self.assertTrue(should_save_round_snapshot(server_round=10, total_rounds=12, tracking_cfg=cfg))
        self.assertTrue(should_save_round_snapshot(server_round=12, total_rounds=12, tracking_cfg=cfg))

    def test_disabled_tracking_saves_no_snapshots(self) -> None:
        cfg = {"enabled": False, "every_n_rounds": 1, "keep_final": True}
        self.assertFalse(should_save_round_snapshot(server_round=3, total_rounds=3, tracking_cfg=cfg))


if __name__ == "__main__":
    unittest.main()

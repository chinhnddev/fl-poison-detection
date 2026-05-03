from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml
from PIL import Image

from attack import BackdoorConfig, BBoxDistortionConfig, LabelFlipConfig, ObjectRemovalConfig, build_poisoned_dataset
from defense import DetectionAwareDefenseConfig, DefenseConfig, detection_aware_filter, robust_filter
from defense.spchm_trust import (
    SPCHMTrustConfig,
    combine_consistency_metrics,
    compute_trust_weights,
    mad_normalize_scores,
    run_spchm_trust_round,
    score_prediction_consistency,
)
from federated.client_app import _materialize_runtime_yaml, _poison_yaml_is_valid


class SPCHMTrustUnitTests(unittest.TestCase):
    def test_hungarian_matching_scorer(self) -> None:
        reference = [
            {"cls": 0, "xyxy": [0.0, 0.0, 10.0, 10.0]},
            {"cls": 1, "xyxy": [20.0, 20.0, 30.0, 30.0]},
        ]
        client = [
            {"cls": 0, "xyxy": [1.0, 1.0, 11.0, 11.0]},
            {"cls": 5, "xyxy": [20.0, 20.0, 30.0, 30.0]},
            {"cls": 3, "xyxy": [40.0, 40.0, 50.0, 50.0]},
        ]

        out = score_prediction_consistency(reference, client, class_penalty=0.5)

        self.assertEqual(out["matched_pairs"], 2)
        self.assertAlmostEqual(out["d_cls"], 0.5, places=6)
        self.assertAlmostEqual(out["r_miss"], 0.0, places=6)
        self.assertAlmostEqual(out["r_ghost"], 1.0 / 3.0, places=6)
        self.assertGreater(out["d_box"], 0.0)

    def test_hungarian_matching_iou_gate_rejects_zero_overlap(self) -> None:
        reference = [{"cls": 0, "xyxy": [0.0, 0.0, 10.0, 10.0]}]
        client = [{"cls": 0, "xyxy": [90.0, 90.0, 100.0, 100.0]}]

        out = score_prediction_consistency(
            reference_detections=reference,
            client_detections=client,
            class_penalty=0.5,
            match_iou_threshold=0.5,
        )

        self.assertEqual(out["matched_pairs"], 0)
        self.assertAlmostEqual(out["r_miss"], 1.0, places=6)
        self.assertAlmostEqual(out["r_ghost"], 1.0, places=6)
        self.assertAlmostEqual(out["d_box"], 0.0, places=6)
        self.assertAlmostEqual(out["d_cls"], 0.0, places=6)

    def test_mad_based_normalization(self) -> None:
        out = mad_normalize_scores([0.1, 0.2, 0.15, 1.5], eps=1e-8)

        self.assertAlmostEqual(out["median"], 0.175, places=6)
        self.assertAlmostEqual(out["mad"], 0.05, places=6)
        self.assertEqual(len(out["z_scores"]), 4)
        self.assertEqual(float(out["z_scores"][0]), 0.0)
        self.assertGreater(float(out["z_scores"][-1]), 0.0)

    def test_trust_weight_normalization(self) -> None:
        updates = [
            ("0", [np.array([1.0, 0.0])], 10),
            ("1", [np.array([0.8, 0.0])], 10),
            ("2", [np.array([0.2, 0.0])], 10),
        ]
        out = compute_trust_weights(
            updates=updates,
            delta_root=[np.array([1.0, 0.0])],
            z_scores=[0.0, 1.0, 2.0],
            tau=1.0,
            eps=1e-8,
            trust_floor=0.0,
        )

        self.assertFalse(out["fallback_used"])
        self.assertAlmostEqual(sum(out["trust_weights"]), 1.0, places=6)
        self.assertGreater(out["trust_weights"][0], out["trust_weights"][1])
        self.assertGreater(out["trust_weights"][1], out["trust_weights"][2])

    def test_trigger_metrics_can_strengthen_proxy_score(self) -> None:
        cfg = SPCHMTrustConfig(proxy_trigger=True, proxy_trigger_mode="max")
        combined = combine_consistency_metrics(
            base_metrics={"d_box": 0.05, "d_cls": 0.0, "r_miss": 0.02, "r_ghost": 0.01, "s_i": 0.08},
            extra_metrics={"d_box": 0.10, "d_cls": 0.20, "r_miss": 0.03, "r_ghost": 0.04, "s_i": 0.37},
            cfg=cfg,
        )

        self.assertAlmostEqual(combined["d_box"], 0.10, places=6)
        self.assertAlmostEqual(combined["d_cls"], 0.20, places=6)
        self.assertAlmostEqual(combined["r_miss"], 0.03, places=6)
        self.assertAlmostEqual(combined["r_ghost"], 0.04, places=6)
        self.assertGreater(combined["s_i"], 0.08)

    def test_fallback_when_all_trust_weights_are_zero(self) -> None:
        updates = [
            ("0", [np.array([-1.0, 0.0])], 2),
            ("1", [np.array([0.0, 1.0])], 3),
        ]
        out = compute_trust_weights(
            updates=updates,
            delta_root=[np.array([1.0, 0.0])],
            z_scores=[0.0, 0.0],
            tau=1.0,
            eps=1e-8,
            trust_floor=0.0,
        )

        self.assertTrue(out["fallback_used"])
        self.assertAlmostEqual(out["trust_weights"][0], 0.4, places=6)
        self.assertAlmostEqual(out["trust_weights"][1], 0.6, places=6)

    def test_spchm_default_keeps_all_clients(self) -> None:
        class FakePredictor:
            def __init__(self, base_model_path: str):
                self.loaded_value = 0.0

            def load_parameters(self, params):
                self.loaded_value = float(np.asarray(params[0]).ravel()[0])

            def predict(self, image_paths, imgsz, device, conf):
                if abs(self.loaded_value - 0.0) < 1e-9:
                    return [{"image_id": str(image_paths[0]), "detections": [{"cls": 0, "xyxy": [0.0, 0.0, 10.0, 10.0]}]}]
                if abs(self.loaded_value - 0.1) < 1e-9:
                    return [{"image_id": str(image_paths[0]), "detections": [{"cls": 0, "xyxy": [0.0, 0.0, 9.0, 9.0]}]}]
                return [{"image_id": str(image_paths[0]), "detections": [{"cls": 1, "xyxy": [20.0, 20.0, 30.0, 30.0]}]}]

            def close(self):
                return None

        cfg = SPCHMTrustConfig(
            enabled=True,
            proxy_data_yaml="proxy.yaml",
            root_data_yaml="root.yaml",
            proxy_max_images=1,
            proxy_conf=0.25,
            proxy_imgsz=320,
            tau=1.0,
            eps=1e-8,
            lambda_box=1.0,
            lambda_cls=1.0,
            lambda_miss=1.0,
            lambda_ghost=1.0,
            hungarian_class_penalty=0.5,
            root_epochs=1,
            root_batch=1,
            root_imgsz=320,
            root_device="cpu",
            trust_floor=0.0,
        )
        updates = [
            ("0", [np.array([0.1])], 5),
            ("1", [np.array([0.2])], 7),
        ]

        with patch("defense.spchm_trust.build_root_delta", return_value={"delta_root": [np.array([1.0])], "num_examples": 4, "checkpoint_path": "tmp.pt"}):
            with patch("defense.spchm_trust.load_dataset_images", return_value=[Path("proxy_image.jpg")]):
                with patch("defense.spchm_trust.ReusableYOLOPredictor", FakePredictor):
                    out = run_spchm_trust_round(
                        updates=updates,
                        global_params=[np.array([0.0])],
                        cfg=cfg,
                        base_model_path="yolov8n.pt",
                        server_round=1,
                    )

        self.assertEqual(out["removed_cids"], [])
        self.assertEqual(len(out["client_diagnostics"]), len(updates))
        self.assertAlmostEqual(sum(row["trust_weight"] for row in out["client_diagnostics"]), 1.0, places=6)

    def test_proxy_only_mode_skips_root_training(self) -> None:
        class FakePredictor:
            def __init__(self, base_model_path: str):
                self.loaded_value = 0.0

            def load_parameters(self, params):
                self.loaded_value = float(np.asarray(params[0]).ravel()[0])

            def predict(self, image_paths, imgsz, device, conf):
                return [{"image_id": str(image_paths[0]), "detections": [{"cls": 0, "xyxy": [0.0, 0.0, 10.0, 10.0]}]}]

            def close(self):
                return None

        cfg = SPCHMTrustConfig(
            enabled=True,
            proxy_data_yaml="proxy.yaml",
            root_data_yaml="",
            proxy_max_images=1,
            proxy_conf=0.25,
            proxy_imgsz=320,
            proxy_trigger=False,
            tau=1.0,
            eps=1e-8,
            trust_floor=0.0,
        )
        updates = [
            ("0", [np.array([0.1])], 5),
            ("1", [np.array([0.2])], 7),
        ]

        with patch("defense.spchm_trust.build_root_delta") as build_root_delta_mock:
            with patch("defense.spchm_trust.load_dataset_images", return_value=[Path("proxy_image.jpg")]):
                with patch("defense.spchm_trust.ReusableYOLOPredictor", FakePredictor):
                    out = run_spchm_trust_round(
                        updates=updates,
                        global_params=[np.array([0.0])],
                        cfg=cfg,
                        base_model_path="yolov8n.pt",
                        server_round=1,
                    )

        build_root_delta_mock.assert_not_called()
        self.assertEqual(out["root_num_examples"], 0)
        self.assertEqual(out["root_checkpoint"], "")
        self.assertAlmostEqual(sum(row["trust_weight"] for row in out["client_diagnostics"]), 1.0, places=6)

    def test_legacy_defenses_still_import_and_run(self) -> None:
        updates = [
            ("0", [np.array([1.0, 0.0])], 5),
            ("1", [np.array([1.1, 0.1])], 5),
            ("2", [np.array([0.9, -0.1])], 5),
            ("3", [np.array([8.0, 8.0])], 5),
        ]
        kept_robust, robust_info = robust_filter(updates, DefenseConfig(enabled=True, min_clients=4))
        kept_detection, detection_info = detection_aware_filter(
            updates,
            [{"cid": cid, "num_examples": n} for cid, _, n in updates],
            DetectionAwareDefenseConfig(enabled=True, min_clients=4),
        )

        self.assertGreaterEqual(len(kept_robust), 1)
        self.assertGreaterEqual(len(kept_detection), 1)
        self.assertIn("removed_cids", robust_info)
        self.assertIn("removed_cids", detection_info)

    def test_poisoned_dataset_yaml_uses_absolute_paths_and_resolves_val_path(self) -> None:
        tmp_path = (Path.cwd() / "tmp" / "test_poisoned_dataset_yaml").resolve()
        if tmp_path.exists():
            shutil.rmtree(tmp_path, ignore_errors=True)
        tmp_path.mkdir(parents=True, exist_ok=True)
        try:
            client_dir = tmp_path / "client_2"
            (client_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
            (client_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
            (client_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
            (client_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

            train_img = client_dir / "images" / "train" / "img0.jpg"
            train_img.write_bytes(b"fake-image")
            (client_dir / "labels" / "train" / "img0.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

            shard_yaml = client_dir / "data.yaml"
            shard_yaml.write_text(
                yaml.safe_dump(
                    {
                        "path": "./datasets",
                        "train": str((client_dir / "images" / "train").resolve()),
                        "val": "client_2/images/val",
                        "nc": 1,
                        "names": ["person"],
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            out_yaml = build_poisoned_dataset(
                shard_data_yaml=str(shard_yaml),
                out_root=str(tmp_path / "poison"),
                label_flip=LabelFlipConfig(enabled=False),
                bbox=BBoxDistortionConfig(enabled=False),
                removal=ObjectRemovalConfig(enabled=False),
                backdoor=BackdoorConfig(enabled=False),
            )
            poisoned_cfg = yaml.safe_load(Path(out_yaml).read_text(encoding="utf-8"))
            poison_dir = Path(out_yaml).parent
            self.assertTrue(Path(poisoned_cfg["path"]).is_absolute())
            self.assertEqual(Path(poisoned_cfg["path"]).resolve(), poison_dir.resolve())
            self.assertTrue(Path(poisoned_cfg["train"]).is_absolute())
            self.assertTrue(Path(poisoned_cfg["train"]).exists())
            val_ref = Path(poisoned_cfg["val"])
            resolved_val = val_ref if val_ref.is_absolute() else (poison_dir / val_ref).resolve()
            self.assertEqual(resolved_val.as_posix(), (client_dir / "images" / "val").resolve().as_posix())
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)

    def test_client_runtime_yaml_uses_absolute_paths(self) -> None:
        tmp_path = (Path.cwd() / "tmp" / "test_client_runtime_yaml").resolve()
        if tmp_path.exists():
            shutil.rmtree(tmp_path, ignore_errors=True)
        tmp_path.mkdir(parents=True, exist_ok=True)
        try:
            client_dir = tmp_path / "client_0"
            for path in [
                client_dir / "images" / "train",
                client_dir / "images" / "val",
                client_dir / "labels" / "train",
                client_dir / "labels" / "val",
            ]:
                path.mkdir(parents=True, exist_ok=True)

            shard_yaml = client_dir / "data.yaml"
            shard_yaml.write_text(
                yaml.safe_dump(
                    {
                        "path": r"G:\My Drive\lv\fl-poison-detection\partitions\coco_val2017\baseline\client_0",
                        "train": r"G:\My Drive\lv\fl-poison-detection\partitions\coco_val2017\baseline\client_0\images\train",
                        "val": r"G:\My Drive\lv\fl-poison-detection\partitions\coco_val2017\baseline\client_0\images\val",
                        "nc": 80,
                        "names": ["person"],
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            runtime_yaml = _materialize_runtime_yaml(shard_yaml)
            runtime_cfg = yaml.safe_load(Path(runtime_yaml).read_text(encoding="utf-8"))
            self.assertEqual(Path(runtime_yaml).parent, shard_yaml.parent.resolve())
            self.assertTrue(Path(runtime_cfg["train"]).is_absolute())
            self.assertTrue(Path(runtime_cfg["val"]).is_absolute())
            self.assertEqual(Path(runtime_cfg["train"]).name, "train")
            self.assertEqual(Path(runtime_cfg["val"]).name, "val")
            self.assertEqual(runtime_cfg["nc"], 80)
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)

    def test_poison_runtime_yaml_resolves_train_txt_to_absolute_path(self) -> None:
        tmp_path = (Path.cwd() / "tmp" / "test_poison_runtime_yaml").resolve()
        if tmp_path.exists():
            shutil.rmtree(tmp_path, ignore_errors=True)
        tmp_path.mkdir(parents=True, exist_ok=True)
        try:
            client_dir = tmp_path / "client_0"
            for path in [
                client_dir / "images" / "train",
                client_dir / "images" / "val",
                client_dir / "labels" / "train",
                client_dir / "labels" / "val",
            ]:
                path.mkdir(parents=True, exist_ok=True)

            train_img = client_dir / "images" / "train" / "img0.jpg"
            val_img = client_dir / "images" / "val" / "val0.jpg"
            Image.new("RGB", (32, 32), color=(255, 255, 255)).save(train_img)
            Image.new("RGB", (32, 32), color=(255, 255, 255)).save(val_img)
            (client_dir / "labels" / "train" / "img0.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            (client_dir / "labels" / "val" / "val0.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

            shard_yaml = client_dir / "data.yaml"
            shard_yaml.write_text(
                yaml.safe_dump(
                    {
                        "path": ".",
                        "train": "images/train",
                        "val": "images/val",
                        "nc": 1,
                        "names": ["person"],
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            poison_yaml = build_poisoned_dataset(
                shard_data_yaml=str(shard_yaml),
                out_root=str(tmp_path / "poison"),
                label_flip=LabelFlipConfig(enabled=False),
                bbox=BBoxDistortionConfig(enabled=False),
                removal=ObjectRemovalConfig(enabled=False),
                backdoor=BackdoorConfig(enabled=True, poison_ratio=1.0, trigger_size=8, src_class_id=0, target_class_id=0),
            )

            poisoned_cfg = yaml.safe_load(Path(poison_yaml).read_text(encoding="utf-8"))
            self.assertTrue(Path(poisoned_cfg["train"]).is_absolute())
            self.assertEqual(Path(poisoned_cfg["train"]).name, "train.txt")

            runtime_yaml = _materialize_runtime_yaml(Path(poison_yaml))
            runtime_cfg = yaml.safe_load(Path(runtime_yaml).read_text(encoding="utf-8"))
            self.assertTrue(Path(runtime_cfg["train"]).is_absolute())
            self.assertTrue(Path(runtime_cfg["val"]).is_absolute())
            self.assertEqual(Path(runtime_cfg["train"]).name, "train.txt")
            self.assertEqual(Path(runtime_cfg["train"]).parent, Path(poison_yaml).parent.resolve())
            self.assertTrue(Path(runtime_cfg["train"]).exists())
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)

    def test_poison_yaml_validation_rejects_legacy_train_filelist(self) -> None:
        tmp_path = (Path.cwd() / "tmp" / "test_poison_yaml_validation").resolve()
        if tmp_path.exists():
            shutil.rmtree(tmp_path, ignore_errors=True)
        tmp_path.mkdir(parents=True, exist_ok=True)
        try:
            poison_dir = tmp_path / "poison"
            (poison_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
            (poison_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (16, 16), color=(255, 255, 255)).save(poison_dir / "images" / "train" / "img0.jpg")
            (poison_dir / "labels" / "train" / "img0.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            (poison_dir / "images" / "val").mkdir(parents=True, exist_ok=True)

            (poison_dir / "train.txt").write_text("images/train/img0.jpg\n", encoding="utf-8")
            legacy_yaml = poison_dir / "data.yaml"
            legacy_yaml.write_text(
                yaml.safe_dump(
                    {
                        "path": ".",
                        "train": "train.txt",
                        "val": str((poison_dir / "images" / "val").resolve()),
                        "nc": 1,
                        "names": ["person"],
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            self.assertFalse(_poison_yaml_is_valid(legacy_yaml))

            (poison_dir / "train.txt").write_text(str((poison_dir / "images" / "train" / "img0.jpg").resolve()) + "\n", encoding="utf-8")
            self.assertTrue(_poison_yaml_is_valid(legacy_yaml))
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)

    def test_backdoor_oversample_factor_duplicates_poisoned_samples(self) -> None:
        tmp_path = (Path.cwd() / "tmp" / "test_backdoor_oversample").resolve()
        if tmp_path.exists():
            shutil.rmtree(tmp_path, ignore_errors=True)
        tmp_path.mkdir(parents=True, exist_ok=True)
        try:
            client_dir = tmp_path / "client_0"
            (client_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
            (client_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
            (client_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
            (client_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

            train_img = client_dir / "images" / "train" / "img0.jpg"
            Image.new("RGB", (32, 32), color=(255, 255, 255)).save(train_img)
            (client_dir / "labels" / "train" / "img0.txt").write_text("0 0.5 0.5 0.2 0.6\n", encoding="utf-8")

            shard_yaml = client_dir / "data.yaml"
            shard_yaml.write_text(
                yaml.safe_dump(
                    {
                        "path": str(client_dir.resolve()),
                        "train": str((client_dir / "images" / "train").resolve()),
                        "val": str((client_dir / "images" / "val").resolve()),
                        "nc": 80,
                        "names": [str(i) for i in range(80)],
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            out_yaml = build_poisoned_dataset(
                shard_data_yaml=str(shard_yaml),
                out_root=str(tmp_path / "poison"),
                label_flip=LabelFlipConfig(enabled=False),
                bbox=BBoxDistortionConfig(enabled=False),
                removal=ObjectRemovalConfig(enabled=False),
                backdoor=BackdoorConfig(
                    enabled=True,
                    poison_ratio=1.0,
                    oversample_factor=3,
                    trigger_size=8,
                    src_class_id=0,
                    target_class_id=77,
                    prob=1.0,
                ),
            )
            poisoned_cfg = yaml.safe_load(Path(out_yaml).read_text(encoding="utf-8"))
            train_txt = Path(poisoned_cfg["train"])
            train_lines = [line.strip() for line in train_txt.read_text(encoding="utf-8").splitlines() if line.strip()]
            meta = yaml.safe_load((tmp_path / "poison" / "poison_meta.yaml").read_text(encoding="utf-8"))
            self.assertEqual(len(train_lines), 3)
            self.assertTrue(all(Path(line).is_absolute() for line in train_lines))
            self.assertEqual(meta["poisoned_images_backdoor"], 1)
            self.assertEqual(meta["poisoned_images_backdoor_replayed"], 2)
            for image_path in [Path(line).resolve() for line in train_lines]:
                self.assertTrue(image_path.exists())
                with Image.open(image_path) as im:
                    self.assertEqual(im.size, (32, 32))
                label_path = image_path.parent.parent.parent / "labels" / "train" / f"{image_path.stem}.txt"
                self.assertTrue(label_path.exists())
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

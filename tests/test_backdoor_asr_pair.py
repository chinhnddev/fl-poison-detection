from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from evaluation.asr import inspect_backdoor_asr_pair


class BackdoorPairInspectionTests(unittest.TestCase):
    def test_pair_inspection_warns_on_natural_cooccurrence_and_recommends_better_target(self) -> None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="backdoor_pair_test_")).resolve()
        try:
            images_train = tmp_dir / "images" / "train"
            images_val = tmp_dir / "images" / "val"
            labels_train = tmp_dir / "labels" / "train"
            labels_val = tmp_dir / "labels" / "val"
            for path in [images_train, images_val, labels_train, labels_val]:
                path.mkdir(parents=True, exist_ok=True)

            def add_sample(split: str, name: str, label_lines: list[str]) -> None:
                (tmp_dir / "images" / split / f"{name}.jpg").write_bytes(b"img")
                (tmp_dir / "labels" / split / f"{name}.txt").write_text("\n".join(label_lines) + "\n", encoding="utf-8")

            add_sample("train", "train0", ["0 0.5 0.5 0.2 0.2", "7 0.3 0.3 0.1 0.1"])
            add_sample("train", "train1", ["0 0.5 0.5 0.2 0.2", "7 0.6 0.6 0.1 0.1"])
            add_sample("train", "train2", ["7 0.4 0.4 0.1 0.1"])
            add_sample("train", "train3", ["7 0.4 0.4 0.1 0.1"])
            add_sample("train", "train4", ["5 0.4 0.4 0.1 0.1"])
            add_sample("val", "val0", ["0 0.5 0.5 0.2 0.2", "5 0.6 0.6 0.1 0.1"])
            add_sample("val", "val1", ["0 0.5 0.5 0.2 0.2"])

            train_txt = tmp_dir / "train.txt"
            val_txt = tmp_dir / "val.txt"
            train_txt.write_text(
                "\n".join(str((images_train / f"train{i}.jpg").resolve()) for i in range(5)) + "\n",
                encoding="utf-8",
            )
            val_txt.write_text(
                "\n".join(str((images_val / f"val{i}.jpg").resolve()) for i in range(2)) + "\n",
                encoding="utf-8",
            )

            data_yaml = tmp_dir / "data.yaml"
            data_yaml.write_text(
                yaml.safe_dump(
                    {
                        "path": str(tmp_dir),
                        "train": str(train_txt.resolve()),
                        "val": str(val_txt.resolve()),
                        "nc": 8,
                        "names": ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck"],
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            out = inspect_backdoor_asr_pair(
                data_yaml=str(data_yaml),
                src_class_id=0,
                target_class_id=5,
                min_train_instances=2,
                min_train_images=2,
                top_k_recommendations=3,
            )

            self.assertTrue(out["available"])
            self.assertEqual(out["target"]["val_images_with_src"], 1)
            self.assertTrue(any("co-occurs naturally" in warning for warning in out["warnings"]))
            self.assertTrue(any("sparse in the train split" in warning for warning in out["warnings"]))
            self.assertGreaterEqual(len(out["recommended_targets"]), 1)
            self.assertEqual(out["recommended_targets"][0]["class_id"], 7)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_pair_inspection_warns_on_geometry_mismatch(self) -> None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="backdoor_pair_geom_")).resolve()
        try:
            images_train = tmp_dir / "images" / "train"
            images_val = tmp_dir / "images" / "val"
            labels_train = tmp_dir / "labels" / "train"
            labels_val = tmp_dir / "labels" / "val"
            for path in [images_train, images_val, labels_train, labels_val]:
                path.mkdir(parents=True, exist_ok=True)

            def add_sample(split: str, name: str, label_lines: list[str]) -> None:
                (tmp_dir / "images" / split / f"{name}.jpg").write_bytes(b"img")
                (tmp_dir / "labels" / split / f"{name}.txt").write_text("\n".join(label_lines) + "\n", encoding="utf-8")

            add_sample("train", "train0", ["0 0.5 0.5 0.1 0.4", "5 0.4 0.4 0.45 0.1"])
            add_sample("train", "train1", ["0 0.5 0.5 0.1 0.4", "5 0.4 0.4 0.45 0.1"])
            add_sample("train", "train2", ["0 0.5 0.5 0.1 0.4", "7 0.4 0.4 0.12 0.38"])
            add_sample("train", "train3", ["0 0.5 0.5 0.1 0.4", "7 0.4 0.4 0.12 0.38"])
            add_sample("train", "train4", ["5 0.4 0.4 0.45 0.1"])
            add_sample("train", "train5", ["7 0.4 0.4 0.12 0.38"])
            add_sample("val", "val0", ["0 0.5 0.5 0.1 0.4"])

            train_txt = tmp_dir / "train.txt"
            val_txt = tmp_dir / "val.txt"
            train_txt.write_text(
                "\n".join(str(path.resolve()) for path in sorted(images_train.glob("*.jpg"))) + "\n",
                encoding="utf-8",
            )
            val_txt.write_text(str((images_val / "val0.jpg").resolve()) + "\n", encoding="utf-8")

            data_yaml = tmp_dir / "data.yaml"
            data_yaml.write_text(
                yaml.safe_dump(
                    {
                        "path": str(tmp_dir),
                        "train": str(train_txt.resolve()),
                        "val": str(val_txt.resolve()),
                        "nc": 8,
                        "names": ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck"],
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            out = inspect_backdoor_asr_pair(
                data_yaml=str(data_yaml),
                src_class_id=0,
                target_class_id=5,
                min_train_instances=2,
                min_train_images=2,
                top_k_recommendations=3,
            )

            self.assertTrue(any("geometry far" in warning for warning in out["warnings"]))
            self.assertEqual(out["recommended_targets"][0]["class_id"], 7)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

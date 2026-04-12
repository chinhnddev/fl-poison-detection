from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from data_partition import write_federated_shards
from federated.client_app import _normalize_client_shard_yaml


class DataPortabilityTests(unittest.TestCase):
    def test_partition_writer_emits_portable_client_yaml(self) -> None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="partition_portable_")).resolve()
        try:
            dataset_dir = tmp_dir / "dataset"
            train_img = dataset_dir / "images" / "train" / "img0.jpg"
            val_img = dataset_dir / "images" / "val" / "val0.jpg"
            train_lbl = dataset_dir / "labels" / "train" / "img0.txt"
            val_lbl = dataset_dir / "labels" / "val" / "val0.txt"
            for path in [train_img.parent, val_img.parent, train_lbl.parent, val_lbl.parent]:
                path.mkdir(parents=True, exist_ok=True)

            train_img.write_bytes(b"train")
            val_img.write_bytes(b"val")
            train_lbl.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            val_lbl.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

            base_yaml = tmp_dir / "base.yaml"
            base_yaml.write_text(
                yaml.safe_dump(
                    {
                        "path": str(dataset_dir.resolve()),
                        "train": str(train_img.parent.resolve()),
                        "val": str(val_img.parent.resolve()),
                        "nc": 1,
                        "names": ["person"],
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            train_txt = tmp_dir / "train.txt"
            val_txt = tmp_dir / "val.txt"
            train_txt.write_text(str(train_img.resolve()) + "\n", encoding="utf-8")
            val_txt.write_text(str(val_img.resolve()) + "\n", encoding="utf-8")

            out_dir = tmp_dir / "partitions"
            write_federated_shards(
                base_data_yaml=str(base_yaml),
                train_txt=str(train_txt),
                val_txt=str(val_txt),
                out_dir=str(out_dir),
                shards=[[train_img.resolve()]],
            )

            client_cfg = yaml.safe_load((out_dir / "client_0" / "data.yaml").read_text(encoding="utf-8"))
            self.assertEqual(client_cfg["path"], ".")
            self.assertEqual(client_cfg["train"], "images/train")
            self.assertEqual(client_cfg["val"], "images/val")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_client_normalizer_rewrites_legacy_absolute_paths(self) -> None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="normalize_client_yaml_")).resolve()
        try:
            client_dir = tmp_dir / "client_0"
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

            normalized = _normalize_client_shard_yaml(shard_yaml)
            self.assertEqual(Path(normalized), shard_yaml.resolve())

            client_cfg = yaml.safe_load(shard_yaml.read_text(encoding="utf-8"))
            self.assertEqual(client_cfg["path"], ".")
            self.assertEqual(client_cfg["train"], "images/train")
            self.assertEqual(client_cfg["val"], "images/val")
            self.assertEqual(client_cfg["nc"], 80)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

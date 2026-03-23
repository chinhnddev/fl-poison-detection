# FL Poisoning Detection (YOLOv8 + Flower)

Federated Learning (FL) pipeline for object detection with:

- YOLOv8 (Ultralytics)
- Flower (FL orchestration)
- IID / Dirichlet Non-IID client partitioning
- Honest vs malicious clients (label flip + model poisoning)
- Optional defense filtering before aggregation
- Evaluation: `mAP@0.5`, `mAP@0.5:0.95`, and Attack Success Rate (ASR)

## 1) Install

```bash
pip install -r requirements.txt
```

## 2) Dataset (COCO128) and No-Leak Train/Val Split

This repo uses `datasets/coco128/` and a dataset YAML at `datasets/coco128/coco128.yaml`.

- Train/val are **different**: the YAML points to filelists `train.txt` and `val.txt`.
- The split + per-client shards are created by `data_partition.py`.

If you want to create the split + shards explicitly:

```bash
python data_partition.py --config config.baseline.yaml
```

If `federated.auto_partition: true` (default in configs), `run_experiment.py` will do it automatically.

## 3) Run Experiments (Baseline / Attack / Defended)

Run these in order so you get 3 models for your results table.

### 3.1 Baseline (no attack, no defense)

```bash
python run_experiment.py --config config.baseline.yaml --log_dir ./logs/baseline
```

Output model: `./artifacts/baseline.pt`

### 3.2 Attack (poisoning, no defense)

```bash
python run_experiment.py --config config.attack.yaml --log_dir ./logs/attack
```

Output model: `./artifacts/attack.pt`

### 3.3 Defended (poisoning + defense)

```bash
python run_experiment.py --config config.defended.yaml --log_dir ./logs/defended
```

Output model: `./artifacts/defended.pt`

Notes:

- You can override `--num_clients`, `--malicious_ratio`, `--rounds` from the CLI.
- If port is busy, `run_experiment.py` auto-switches and logs the chosen address.

## 4) Evaluate (mAP + ASR)

Example: `person (0) -> dog (16)` with IoU threshold 0.5.

```bash
python evaluate.py \
  --data ./datasets/coco128/coco128.yaml \
  --baseline ./artifacts/baseline.pt \
  --attacked ./artifacts/attack.pt \
  --defended ./artifacts/defended.pt \
  --device cuda:0 \
  --imgsz 320 \
  --asr_src_class_id 0 \
  --asr_target_class_id 56 \
  --asr_iou 0.5 \
  --asr_trigger \
  --trigger_size 16 \
  --trigger_value 255 \
  --trigger_position bottom_right
```

**ASR trigger arguments:**

| Argument | Default | Description |
|---|---|---|
| `--asr_trigger` | flag | Inject trigger patch into val images before inference. |
| `--trigger_size` | `16` | Side length (in pixels) of the square trigger patch (e.g. `16` → a 16×16 px patch). Must match the `trigger_size` used during training-time backdoor poisoning. |
| `--trigger_value` | `255` | Grayscale fill value for the patch (0–255); `255` = white. |
| `--trigger_position` | `bottom_right` | Corner where the patch is placed: `bottom_right`, `bottom_left`, `top_right`, or `top_left`. |

`evaluate.py` writes a table to stdout and saves YOLO validation outputs under `runs/detect/...`.

## 5) Key Config Knobs

Configs:

- `config.baseline.yaml`
- `config.attack.yaml`
- `config.defended.yaml`

Important fields:

- `runtime.seed`: global reproducibility seed (random/numpy/torch)
- `model.base`: base YOLO checkpoint (e.g. `yolov8n.pt`)
- `data.yaml`: dataset YAML (default `./datasets/coco128/coco128.yaml`)
- `federated.partition`: `iid` or `dirichlet`
- `federated.dirichlet_alpha`: default `0.5`
- `attack.malicious_ratio`: default `0.4` (paper default)
- `attack.backdoor.trigger_size`: side length (in pixels) of the square trigger patch (e.g. `16` → a 16×16 px patch)
- `attack.label_flip.*`: label flip configuration
- `attack.model_poison.*`: delta poisoning (`scale` / `signflip`) + `strength`
- `defense.enabled`: enable/disable defense filtering
- `defense.detection_aware`: enable detection-aware defense (requires `collect_detection_stats: true`)
- `defense.collect_detection_stats`: clients run inference on val set each round and report stats
- `model.global_out`: where the server saves the aggregated global model

## 6) Detection-Aware Defense

The proposed detection-aware defense (`defense/detection_aware_filter.py`) extends the
gradient-based outlier filter with object-detection-specific signals:

| Signal | Detects |
|---|---|
| Class frequency deviation | Label-flip / backdoor attacks |
| Bounding-box distribution deviation | BBox distortion attacks |
| Detection rate deviation | Object removal attacks |
| IoU vs global model | Poisoning that distorts prediction geometry |

### Run with detection-aware defense

```bash
python run_experiment.py --config config.detection_aware.yaml --log_dir ./logs/da_defended
```

Config key fields (under `defense:`):

```yaml
defense:
  enabled: true
  detection_aware: true           # activate detection-aware defense
  collect_detection_stats: true   # clients report prediction stats each round
  det_stats_max_images: 50        # max val images per client per round
  class_freq_z: 2.0               # z-score threshold for class-freq deviation
  bbox_z: 2.0                     # z-score threshold for bbox distribution
  detection_rate_z: 2.0           # z-score threshold for detection rate
  iou_z: 2.0                      # z-score threshold for IoU vs global
  detection_weights:
    class_freq: 2.0               # weight for class-frequency anomaly score
    bbox: 1.0
    detection_rate: 1.0
    iou: 1.5
```

## 7) Code Layout

- `data_partition.py`: train/val split + per-client partitioning (IID/Dirichlet) + stats
- `attack.py`: label flip shard view + model poisoning in delta space
- `client.py`: Flower client; trains only on its shard (no fallback to full dataset)
- `server.py`: aggregation + (optional) defense filtering + model export
- `defense.py`: backward-compatible wrapper for defense modules
- `defense/robust_filter.py`: gradient-based outlier filter (cosine / norm / distance)
- `defense/detection_aware_filter.py`: detection-aware defense (gradient + prediction stats)
- `train_yolo.py`: YOLO train wrapper + seed helpers + parameter (de)serialization + detection stat collection
- `evaluate.py`: mAP + strict ASR implementation

## 8) Google Colab Notes

- Use `--device cuda:0` (not `device=gpu`).
- Logs: pass `--log_dir /content/...` and tail `server.log` / `client_*.log` in another cell.

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
  --asr_trigger
```

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
- `attack.label_flip.*`: label flip configuration
- `attack.model_poison.*`: delta poisoning (`scale` / `signflip`) + `strength`
- `defense.enabled`: enable/disable defense filtering
- `model.global_out`: where the server saves the aggregated global model

## 6) Code Layout

- `data_partition.py`: train/val split + per-client partitioning (IID/Dirichlet) + stats
- `attack.py`: label flip shard view + model poisoning in delta space
- `client.py`: Flower client; trains only on its shard (no fallback to full dataset)
- `server.py`: aggregation + (optional) defense filtering + model export
- `defense.py`: update filtering logic
- `train_yolo.py`: YOLO train wrapper + seed helpers + parameter (de)serialization
- `evaluate.py`: mAP + strict ASR implementation

## 7) Google Colab Notes

- Use `--device cuda:0` (not `device=gpu`).
- Logs: pass `--log_dir /content/...` and tail `server.log` / `client_*.log` in another cell.

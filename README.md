# FL Poisoning Detection (YOLOv8 + Flower)

Federated Learning (FL) pipeline for object detection with:

- YOLOv8 (Ultralytics)
- Flower (FL orchestration)
- IID / Dirichlet non-IID client partitioning
- Honest vs malicious clients (label flip, bbox distortion, object removal, backdoor, model poisoning)
- Legacy gradient-only and detection-aware defenses
- New thesis-mode server-side defense: `SPCHM-Trust`
- Evaluation: `mAP@0.5`, `mAP@0.5:0.95`, ASR, and optional perception metrics

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

## 3) Run Experiments

Run the modes below as separate experiments. Existing configs remain valid and the new thesis mode is additive.

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

### 3.3 Legacy Gradient-Robust Defense

```bash
python run_experiment.py --config config.defended.yaml --log_dir ./logs/defended
```

Output model: `./artifacts/defended.pt`

### 3.4 Legacy Detection-Aware Defense

```bash
python run_experiment.py --config config.detection_aware.yaml --log_dir ./logs/detection_aware
```

Output model: `./artifacts/da_defended.pt`

### 3.5 SPCHM-Trust Thesis Defense

```bash
python run_experiment.py --config config.spchm_trust.yaml --log_dir ./logs/spchm_trust
```

Output model: `./artifacts/spchm_trust.pt`

Notes:

- You can override `--num_clients`, `--malicious_ratio`, `--rounds` from the CLI.
- If port is busy, `run_experiment.py` auto-switches and logs the chosen address.
- `COCO128` + `yolov8n` is kept for smoke tests.
- `COCO2017` + `yolov8s` is the intended research setting; see commented placeholders in `config.spchm_trust.yaml`.

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

## 5) Defense Modes

Legacy modes are preserved:

- `config.baseline.yaml`: no attack, no defense
- `config.attack.yaml`: attack only
- `config.defended.yaml`: legacy gradient-based robust filter with hard removal
- `config.detection_aware.yaml`: legacy client-statistics-based detection-aware filter with hard removal

New thesis mode:

- `config.spchm_trust.yaml`: server-side proxy/root-set scoring with trust-weighted soft aggregation

`SPCHM-Trust` keeps the current client protocol unchanged: clients still send delta updates and the server reconstructs `theta_t + delta_i` internally.

## 6) Key Config Knobs

Configs:

- `config.baseline.yaml`
- `config.attack.yaml`
- `config.defended.yaml`
- `config.detection_aware.yaml`
- `config.spchm_trust.yaml`

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
- `defense.detection_aware`: enable detection-aware defense (requires `collect_detection_stats: true`)
- `defense.collect_detection_stats`: clients run inference on val set each round and report stats
- `defense.spchm_trust`: enable server-side SPCHM-Trust
- `defense.proxy_data_yaml`: clean shared proxy dataset for server-side prediction consistency scoring
- `defense.root_data_yaml`: clean root dataset for trusted root-direction update
- `defense.tau`: anomaly-to-trust decay factor
- `defense.lambda_box|lambda_cls|lambda_miss|lambda_ghost`: composite score weights
- `model.global_out`: where the server saves the aggregated global model

## 7) Legacy Detection-Aware Defense

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

## 8) SPCHM-Trust

The thesis-mode defense lives in `defense/spchm_trust.py` and runs primarily on the server:

1. Reconstruct each client model from the current global model plus its delta update.
2. Run server-side inference on a shared clean proxy set.
3. Compare each reconstructed client model to the current global reference model using Hungarian matching on detections.
4. Compute the composite anomaly score:

```text
s_i = lambda1 * d_box + lambda2 * d_cls + lambda3 * r_miss + lambda4 * r_ghost
```

5. Apply MAD normalization:

```text
z_i = max(0, (s_i - median(S)) / (1.4826 * MAD(S) + eps))
```

6. Train a small server-side root update on clean data and compute trust:

```text
r_i = exp(-tau * z_i) * max(0, cosine(delta_i, delta_root))
```

7. Aggregate client deltas with normalized trust weights instead of hard-removing clients by default.

Round-level SPCHM diagnostics are appended per client to `round_stats.jsonl`.

## 9) Evaluation

Existing evaluation still works:

- `mAP@0.5`
- `mAP@0.5:0.95`
- Backdoor ASR

For backdoor comparisons, it is useful to report both:

- `--asr_mode relaxed` for parity with legacy attack runs
- `--asr_mode strict` to verify that source objects are actually rewritten into the target class

Optional perception-oriented metrics are available from `evaluation/perception_metrics.py` and the CLI:

```bash
python evaluate.py \
  --data ./datasets/coco128/coco128.yaml \
  --defended ./artifacts/spchm_trust.pt \
  --device cuda:0 \
  --imgsz 320 \
  --perception \
  --perception_max_images 50
```

This reports missing-object rate, ghost-object rate, class mismatch rate, and mean matched box deviation.

## 10) Code Layout

- `data_partition.py`: train/val split + per-client partitioning (IID/Dirichlet) + stats
- `attack.py`: label flip shard view + model poisoning in delta space
- `client.py`: Flower client; trains only on its shard (no fallback to full dataset)
- `server.py`: aggregation + (optional) defense filtering + model export
- `defense.py`: backward-compatible wrapper for defense modules
- `defense/robust_filter.py`: gradient-based outlier filter (cosine / norm / distance)
- `defense/detection_aware_filter.py`: detection-aware defense (gradient + prediction stats)
- `defense/spchm_trust.py`: thesis-mode server-side proxy/root-set trust aggregation
- `train_yolo.py`: YOLO train wrapper + seed helpers + parameter (de)serialization + server-side inference helpers
- `evaluation/perception_metrics.py`: optional perception-oriented robustness metrics
- `evaluate.py`: mAP + ASR + optional perception metrics

## 11) Google Colab Notes

- Use `--device cuda:0` (not `device=gpu`).
- Logs: pass `--log_dir /content/...` and tail `server.log` / `client_*.log` in another cell.

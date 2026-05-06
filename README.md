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

## 2) Dataset (COCO val2017) and Partitioning

This repo now uses `COCO val2017` as the main dataset via `datasets/coco_val2017.yaml`.
If the dataset is missing, the pipeline can now download and prepare it automatically.

Expected layout:

```text
datasets/
  coco_val2017.yaml
  coco/
    images/
      val2017/
    labels/
      val2017/
    annotations/
      instances_val2017.json
```

- `python scripts/download_coco.py` downloads `val2017.zip` and `annotations_trainval2017.zip` from the official COCO URLs.
- The download step also converts `instances_val2017.json` into YOLO label files under `datasets/coco/labels/val2017/`.
- `datasets/coco_val2017.yaml` keeps the full COCO 80-class mapping and points to `datasets/coco`.
- `data_partition.py` uses `images/val2017` as the canonical 5,000-image source pool.
- Partition outputs are written under config-specific folders (e.g. `federated_data_baseline`, `federated_data_attack`).
- The script also writes `partition_manifest.json` beside `partition_stats.yaml` for easier inspection.

One-line download command:

```bash
python scripts/download_coco.py
```

Or:

```bash
make download-data
```

### Build a COCO20 Research Subset

To build a standalone 20-class subset from COCO 2017 for FL/object detection experiments:

```bash
python scripts/build_coco20_subset.py \
  --coco-root datasets/coco \
  --out-root datasets/coco20 \
  --train-size 3000 \
  --val-size 1000 \
  --seed 42
```

The script:

- reads `instances_train2017.json` and `instances_val2017.json`
- keeps only 20 selected classes
- removes images that become annotation-empty after filtering
- samples a more balanced subset instead of naive random sampling
- writes filtered COCO annotations, copied/linked images, YOLO labels, `coco20.yaml`, and train/val stats JSON files

Expected output:

```text
datasets/coco20/
  images/train/
  images/val/
  labels/train/
  labels/val/
  annotations/instances_train.json
  annotations/instances_val.json
  coco20.yaml
  stats_train.json
  stats_val.json
```

Run partition explicitly before training/evaluation:

```bash
python data_partition.py \
  --data_yaml ./datasets/coco_val2017.yaml \
  --num_clients 10 \
  --out_dir ./federated_data_baseline \
  --partition iid \
  --val_ratio 0.2
```

Partition is now a separate step. `run_experiment.py` will not partition unless you pass `--partition`.

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
- Default configs now target `COCO val2017` with `yolov8n`.
- Training configs use `batch: 16` and `num_workers: 4` by default; reduce them if your machine runs out of memory.
- `eval.round_tracking` can now evaluate validation mAP at saved round checkpoints, emit `round_metrics.csv/json/png`, and copy the best checkpoint to `*_best.pt`.

## 4) Evaluate (mAP + ASR)

Default ASR pair remains `person (0) -> spoon (44)` with IoU threshold `0.5`.

Recommended order:

```bash
python data_partition.py \
  --data_yaml ./datasets/coco_val2017.yaml \
  --num_clients 10 \
  --out_dir ./federated_data_baseline \
  --partition iid \
  --val_ratio 0.2
```

```bash
python evaluate.py \
  --data ./datasets/coco_val2017.yaml \
  --baseline ./artifacts/baseline.pt \
  --attacked ./artifacts/attack.pt \
  --defended ./artifacts/defended.pt \
  --device cuda:0 \
  --imgsz 320 \
  --asr_src_class_id 0 \
  --asr_target_class_id 44 \
  --asr_iou 0.5 \
  --asr_trigger
```

`evaluate.py` writes a table to stdout and saves YOLO validation outputs under `runs/detect/...`.

## 4.1) Find The Best Federated Round

To study convergence over longer runs such as 50 rounds, enable:

```yaml
eval:
  round_tracking:
    enabled: true
    every_n_rounds: 1
    selection_metric: "map5095"
    patience: 5
    min_delta: 0.001
```

When enabled:

- The server saves round checkpoints as `artifacts/<name>_round_XXXX.pt`
- `run_experiment.py` evaluates each saved checkpoint on the validation set after training
- Results are written to `logs/<run>/round_metrics.csv`, `round_metrics.json`, and `round_metrics.png`
- The best checkpoint is copied to `artifacts/<name>_best.pt`

The summary reports:

- `best_round`: highest validation metric among evaluated checkpoints
- `convergence_round`: the last meaningfully improving round before `patience` consecutive checks without at least `min_delta` gain

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
- `dataset.base_data_yaml`: dataset YAML (default `./datasets/coco_val2017.yaml`)
- `train.batch`: local batch size (default `16`)
- `train.num_workers`: dataloader workers per client (default `4`)
- `federated.partition`: `iid` or `dirichlet`
- `federated.data_dir`: partition output directory (config-specific, e.g. `./federated_data_baseline`).
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

6. Compute trust from the proxy score and the client update direction:

```text
r_i = exp(-tau * z_i)
```

7. Aggregate client deltas with normalized trust weights instead of hard-removing clients by default.

Round-level SPCHM diagnostics are appended per client to `round_stats.jsonl`.

## 9) Evaluation

Existing evaluation still works:

- `mAP@0.5`
- `mAP@0.5:0.95`
- Backdoor ASR

For backdoor comparisons, it is useful to report both:

- `--asr_mode strict` for the main backdoor metric so source objects must be rewritten into the target class
- `--asr_mode relaxed` only when you explicitly want the legacy benchmark that tolerates natural target hits
- `--asr_pair_report` to surface source/target dataset diagnostics and suggested alternatives when a target class is too sparse, naturally co-occurs with the source on val, or has very different box geometry

On COCO val2017, the default backdoor target remains `spoon (44)` for consistency with the existing experiments.

Optional perception-oriented metrics are available from `evaluation/perception_metrics.py` and the CLI:

```bash
python evaluate.py \
  --data ./datasets/coco_val2017.yaml \
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

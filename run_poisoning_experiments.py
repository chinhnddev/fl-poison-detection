import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def _write_yaml(path: Path, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _run(cmd, cwd: Path) -> None:
    print(">", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _assert_exists(path: str, label: str) -> None:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Expected {label} model was not created: {p.resolve()}")


def _evaluate(
    repo: Path,
    data_yaml: str,
    asr_src_class_id: int,
    asr_target_class_id: int,
    baseline: str,
    attacked: str,
    defended: str,
    device: str,
    imgsz: int,
) -> None:
    cmd = [
        sys.executable,
        "evaluate.py",
        "--data",
        data_yaml,
        "--baseline",
        baseline,
        "--attacked",
        attacked,
        "--defended",
        defended,
        "--device",
        device,
        "--imgsz",
        str(imgsz),
    ]
    cmd += ["--asr_src_class_id", str(asr_src_class_id), "--asr_target_class_id", str(asr_target_class_id)]
    _run(cmd, repo)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run baseline/attack/defended poisoning experiments.")
    ap.add_argument("--num_clients", type=int, default=10)
    ap.add_argument("--malicious_ratio", type=float, default=0.4)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--aggregation", choices=["fedavg"], default="fedavg")
    ap.add_argument("--baseline_config", default="config.baseline.yaml")
    ap.add_argument("--attack_config", default="config.attack.yaml")
    ap.add_argument("--defended_config", default="config.defended.yaml")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--log_root", default="./logs_poison")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent
    cfg_attack = yaml.safe_load(open(repo / args.attack_config, "r", encoding="utf-8"))
    data_yaml = str(cfg_attack["dataset"]["base_data_yaml"])
    lf = ((cfg_attack.get("attack") or {}).get("label_flip")) or {}
    src_id = int(lf.get("src_class_id", 0))
    dst_id = int(lf.get("dst_class_id", 56))

    out_dir = repo / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_model = str((out_dir / "baseline.pt").resolve())

    # 1) baseline (no attack): malicious_ratio=0, defense off
    cfg_baseline = yaml.safe_load(open(repo / args.baseline_config, "r", encoding="utf-8"))
    cfg_baseline["train"] = dict(cfg_baseline.get("train") or {})
    cfg_baseline["train"]["device"] = args.device
    cfg_baseline["train"]["imgsz"] = int(args.imgsz)
    cfg_baseline["train"]["batch"] = int(cfg_baseline["train"].get("batch", 2))
    cfg_baseline["defense"] = dict(cfg_baseline.get("defense") or {})
    cfg_baseline["defense"]["enabled"] = False
    cfg_baseline["attack"] = dict(cfg_baseline.get("attack") or {})
    for k in ["label_flip", "bbox_distortion", "object_removal", "model_poison"]:
        cfg_baseline["attack"][k] = dict((cfg_baseline["attack"].get(k) or {}))
        cfg_baseline["attack"][k]["enabled"] = False
    cfg_baseline["model"] = dict(cfg_baseline.get("model") or {})
    cfg_baseline["model"]["global_out"] = baseline_model
    cfg_baseline["eval"] = dict(cfg_baseline.get("eval") or {})
    cfg_baseline["eval"]["run_after_experiment"] = False
    baseline_cfg_path = repo / "tmp" / "cfg_baseline.yaml"
    _write_yaml(baseline_cfg_path, cfg_baseline)
    _run(
        [
            sys.executable,
            "run_experiment.py",
            "--num_clients",
            str(args.num_clients),
            "--malicious_ratio",
            "0.0",
            "--aggregation",
            args.aggregation,
            "--rounds",
            str(args.rounds),
            "--config",
            str(baseline_cfg_path),
            "--log_dir",
            str((Path(args.log_root) / "baseline").resolve()),
        ],
        repo,
    )
    _assert_exists(baseline_model, "baseline")

    # 2) attack (poisoning without defense): malicious_ratio=0.3, defense off
    cfg_attack = yaml.safe_load(open(repo / args.attack_config, "r", encoding="utf-8"))
    cfg_attack["train"] = dict(cfg_attack.get("train") or {})
    cfg_attack["train"]["device"] = args.device
    cfg_attack["train"]["imgsz"] = int(args.imgsz)
    cfg_attack["defense"] = dict(cfg_attack.get("defense") or {})
    cfg_attack["defense"]["enabled"] = False
    cfg_attack["model"] = dict(cfg_attack.get("model") or {})
    cfg_attack["model"]["global_out"] = str((out_dir / "attack.pt").resolve())
    cfg_attack["eval"] = dict(cfg_attack.get("eval") or {})
    cfg_attack["eval"]["run_after_experiment"] = False
    attack_cfg_path = repo / "tmp" / "cfg_attack.yaml"
    _write_yaml(attack_cfg_path, cfg_attack)
    _run(
        [
            sys.executable,
            "run_experiment.py",
            "--num_clients",
            str(args.num_clients),
            "--malicious_ratio",
            str(args.malicious_ratio),
            "--aggregation",
            args.aggregation,
            "--rounds",
            str(args.rounds),
            "--config",
            str(attack_cfg_path),
            "--log_dir",
            str((Path(args.log_root) / "attack").resolve()),
        ],
        repo,
    )
    _assert_exists(str((out_dir / "attack.pt").resolve()), "attack")

    # 3) defended (poisoning + defense): malicious_ratio=0.3, defense on
    cfg_def = yaml.safe_load(open(repo / args.defended_config, "r", encoding="utf-8"))
    cfg_def["train"] = dict(cfg_def.get("train") or {})
    cfg_def["train"]["device"] = args.device
    cfg_def["train"]["imgsz"] = int(args.imgsz)
    cfg_def["defense"] = dict(cfg_def.get("defense") or {})
    cfg_def["defense"]["enabled"] = True
    cfg_def["model"] = dict(cfg_def.get("model") or {})
    cfg_def["model"]["global_out"] = str((out_dir / "defended.pt").resolve())
    cfg_def["eval"] = dict(cfg_def.get("eval") or {})
    cfg_def["eval"]["run_after_experiment"] = False
    defended_cfg_path = repo / "tmp" / "cfg_defended.yaml"
    _write_yaml(defended_cfg_path, cfg_def)
    _run(
        [
            sys.executable,
            "run_experiment.py",
            "--num_clients",
            str(args.num_clients),
            "--malicious_ratio",
            str(args.malicious_ratio),
            "--aggregation",
            args.aggregation,
            "--rounds",
            str(args.rounds),
            "--config",
            str(defended_cfg_path),
            "--log_dir",
            str((Path(args.log_root) / "defended").resolve()),
        ],
        repo,
    )
    _assert_exists(str((out_dir / "defended.pt").resolve()), "defended")

    # Evaluate all three.
    _evaluate(
        repo=repo,
        data_yaml=str((repo / data_yaml).resolve()),
        asr_src_class_id=src_id,
        asr_target_class_id=dst_id,
        baseline=baseline_model,
        attacked=str((out_dir / "attack.pt").resolve()),
        defended=str((out_dir / "defended.pt").resolve()),
        device=args.device,
        imgsz=args.imgsz,
    )
    print(f"Models written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

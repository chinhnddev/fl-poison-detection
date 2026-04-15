import argparse
import math
import random
import subprocess
from subprocess import TimeoutExpired
import sys
import time
import os
from pathlib import Path
import socket
import yaml


def _abs_path(path_str: str, base_dir: Path) -> Path:
    p = Path(str(path_str)).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (base_dir / p).resolve()


def _is_port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
        except OSError:
            return False
    return True


def _pick_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def _wait_for_server(host: str, port: int, timeout_s: float = 60.0) -> bool:
    """Wait until a TCP server is accepting connections on host:port."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            try:
                if s.connect_ex((host, int(port))) == 0:
                    return True
            except OSError:
                pass
        time.sleep(0.5)
    return False


def _tail_text(path: Path, max_lines: int = 120) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return ""
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[-max_lines:])


def main():
    repo_root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_clients", type=int, default=10)
    ap.add_argument("--malicious_ratio", type=float, default=0.4)
    ap.add_argument(
        "--malicious_selection",
        choices=["random", "topk_src_class"],
        default="random",
        help="How to choose malicious clients. topk_src_class uses partition_stats.yaml and attack.(label_flip|backdoor).src_class_id.",
    )
    ap.add_argument(
        "--aggregation",
        choices=["fedavg"],
        default="fedavg",
        help="Research experiments use FedAvg; robustness comes from defenses on deltas.",
    )
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--config", default="config.baseline.yaml")
    ap.add_argument("--log_dir", default="./logs", help="Write server/client logs here")
    ap.add_argument(
        "--partition",
        action="store_true",
        help="Run data_partition.py before starting the experiment.",
    )
    ap.add_argument(
        "--server_ready_timeout_s",
        type=float,
        default=180.0,
        help="Seconds to wait for server to start accepting connections (Colab can be slow)",
    )
    ap.add_argument(
        "--run_timeout_s",
        type=float,
        default=0.0,
        help="Max seconds to wait for the full run to finish (0 = no timeout). Increase for many rounds/CPU runs.",
    )
    args = ap.parse_args()

    config_path = _abs_path(args.config, repo_root)
    log_dir = _abs_path(args.log_dir, repo_root)

    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    seed = int(((cfg.get("runtime") or {}).get("seed")) or 1234)
    random.seed(seed)
    host, port = cfg["server"]["host"], cfg["server"]["port"]
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using dataset={cfg['dataset']['base_data_yaml']} eval_data={cfg['eval']['data_yaml']}")

    if not _is_port_available(host, int(port)):
        new_port = _pick_free_port(host)
        print(f"Port {host}:{port} is in use, switching to {host}:{new_port}")
        port = new_port

    # 1) partition dataset (optional, explicit via --partition)
    data_dir = Path(cfg["federated"]["data_dir"])
    shard_probe = data_dir / "client_0" / "data.yaml"
    needs_partition = args.partition or (not shard_probe.exists())
    if needs_partition:
        if not args.partition:
            print(f"Partition not found at {shard_probe}, creating it now...")
        print(
            "Partitioning dataset -> "
            f"data_yaml={cfg['dataset']['base_data_yaml']} "
            f"out_dir={cfg['federated']['data_dir']} "
            f"num_clients={args.num_clients}"
        )
        cmd = [
            sys.executable, str((repo_root / "data_partition.py").resolve()),
            "--data_yaml", cfg["dataset"]["base_data_yaml"],
            "--num_clients", str(args.num_clients),
            "--out_dir", cfg["federated"]["data_dir"],
            "--seed", str(seed),
            "--partition", str((cfg.get("federated") or {}).get("partition", "dirichlet")),
            "--dirichlet_alpha", str((cfg.get("federated") or {}).get("dirichlet_alpha", 0.5)),
            "--val_ratio", str((cfg.get("dataset") or {}).get("val_ratio", 0.2)),
        ]
        subprocess.run(cmd, check=True)
        print(f"Partition manifest: {Path(cfg['federated']['data_dir']).resolve() / 'partition_manifest.json'}")

    def _any_attack_enabled(c: dict) -> bool:
        a = c.get("attack") or {}
        for key in ["label_flip", "bbox_distortion", "object_removal", "backdoor", "model_poison"]:
            if bool((a.get(key) or {}).get("enabled", False)):
                return True
        return False

    # 4) malicious assignment
    # Baseline runs should not sample "malicious" clients at all (keeps logs/results unambiguous).
    if not _any_attack_enabled(cfg):
        k = 0
    else:
        k = math.floor(args.num_clients * args.malicious_ratio)

    def _choose_malicious_topk_src_class() -> set[int]:
        stats_path = Path(cfg["federated"]["data_dir"]) / "partition_stats.yaml"
        if not stats_path.exists():
            return set(random.sample(range(args.num_clients), k)) if k > 0 else set()
        stats = yaml.safe_load(open(stats_path, "r", encoding="utf-8")) or {}
        # Determine src class from config (prefer label_flip, fall back to backdoor).
        attack_cfg = cfg.get("attack") or {}
        src = int((((attack_cfg.get("label_flip") or {}).get("src_class_id")) or ((attack_cfg.get("backdoor") or {}).get("src_class_id")) or 0))
        scored = []
        for cid in range(args.num_clients):
            key = f"client_{cid}"
            entry = stats.get(key, {}) or {}
            opc = entry.get("objects_per_class", {}) or {}
            # YAML may load keys as int or str depending on how it was written.
            count = int(opc.get(src, opc.get(str(src), 0)) or 0)
            scored.append((count, cid))
        scored.sort(reverse=True)
        # If all counts are zero, fall back to random selection.
        if not scored or scored[0][0] == 0:
            return set(random.sample(range(args.num_clients), k)) if k > 0 else set()
        return set(cid for _, cid in scored[:k])

    # Config may override CLI selection.
    cfg_sel = str(((cfg.get("attack") or {}).get("malicious_selection")) or "").strip().lower()
    sel = args.malicious_selection
    if cfg_sel in {"random", "topk_src_class", "topk_src_class_id", "topk_src"}:
        sel = "topk_src_class" if cfg_sel.startswith("topk") else "random"

    if k > 0:
        if sel == "topk_src_class":
            malicious = _choose_malicious_topk_src_class()
        else:
            malicious = set(random.sample(range(args.num_clients), k))
    else:
        malicious = set()

    # Persist run metadata for paper-style bookkeeping.
    meta = {
        "seed": seed,
        "num_clients": int(args.num_clients),
        "malicious_ratio": float(args.malicious_ratio),
        "malicious_selection": str(sel),
        "malicious_cids": sorted(list(malicious)),
        "rounds": int(args.rounds),
        "aggregation": str(args.aggregation),
        "config": str(config_path),
    }
    # Defensive: ensure log_dir exists even if user launches from an unexpected CWD
    # or a previous run cleaned it up.
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "run_meta.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False)

    # 2) launch server
    server_cmd = [
        sys.executable, str((repo_root / "server.py").resolve()),
        "--host", host, "--port", str(port),
        "--rounds", str(args.rounds),
        "--aggregation", args.aggregation,
        "--config", str(config_path),
        "--expected_clients", str(args.num_clients),
        "--round_stats_out", str((log_dir / "round_stats.jsonl").resolve()),
    ]
    server = None
    clients = []
    bad = []
    server_log = None
    client_logs = []
    server_log_path = log_dir / "server.log"
    try:
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"  # make subprocess logs appear in near-realtime

        print(f"Starting server -> {host}:{port} (logs: {server_log_path.resolve()})")
        server_log = open(server_log_path, "w", encoding="utf-8", buffering=1)
        server = subprocess.Popen(server_cmd, stdout=server_log, stderr=subprocess.STDOUT, text=True, env=env, cwd=str(repo_root))
        # Server startup can take time (imports, weights load). Wait for the port to accept connections.
        # If the process exits early, fail fast and print the log tail for debugging.
        deadline = time.time() + float(args.server_ready_timeout_s)
        ready = False
        while time.time() < deadline:
            if server.poll() is not None:
                server_log.flush()
                tail = _tail_text(server_log_path)
                msg = f"Server failed to start on {host}:{port} (exit={server.returncode})."
                if tail:
                    msg += "\n--- server.log (tail) ---\n" + tail
                raise SystemExit(msg)
            if _wait_for_server(host, int(port), timeout_s=2.0):
                ready = True
                break
            time.sleep(0.5)
        if not ready:
            server_log.flush()
            tail = _tail_text(server_log_path)
            msg = f"Server did not become ready on {host}:{port} within timeout ({args.server_ready_timeout_s}s)."
            if tail:
                msg += "\n--- server.log (tail) ---\n" + tail
            raise SystemExit(msg)

        # 3) launch clients
        print(f"Server is ready. Launching {args.num_clients} clients (logs: {log_dir.resolve()})")
        for cid in range(args.num_clients):
            clog = open(log_dir / f"client_{cid}.log", "w", encoding="utf-8", buffering=1)
            client_logs.append(clog)
            c = subprocess.Popen([
                sys.executable, str((repo_root / "client.py").resolve()),
                "--cid", str(cid),
                "--server_address", f"{host}:{port}",
                "--config", str(config_path),
                "--malicious", "1" if cid in malicious else "0",
            ], stdout=clog, stderr=subprocess.STDOUT, text=True, env=env, cwd=str(repo_root))
            clients.append(c)

# 5) wait for experiment to finish
        try:
            rt = float(args.run_timeout_s)
            if rt > 0:
                server.wait(timeout=rt)
            else:
                server.wait()
        except subprocess.TimeoutExpired:
            print("Server timeout -> check logs")
            try:
                server_log.flush()
            except Exception:
                pass
            tail = _tail_text(server_log_path)
            if tail:
                print("--- server.log (tail) ---")
                print(tail)
        bad = []
        for i, c in enumerate(clients):
            try:
                c.wait(timeout=30.0)
            except subprocess.TimeoutExpired:
                print(f"Client {i} timeout -> force kill")
                c.kill()
                c.wait(timeout=10.0)
                bad.append(i)
        bad = [i for i, p in enumerate(clients) if p.returncode not in (0, None)]
    finally:
        # 6) FORCE kill ALL processes
        for c in clients:
            if c.poll() is None:
                c.kill()
                try:
                    c.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    c.kill()
        # Kill server last
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
        for f in client_logs:
            try:
                f.flush()
                f.close()
            except Exception:
                pass
        if server_log is not None:
            try:
                server_log.flush()
                server_log.close()
            except Exception:
                pass

    if bad:
        raise SystemExit(f"One or more clients failed (cids={bad}). See logs in: {log_dir.resolve()}")

    print("Experiment done.")
    print(f"Aggregation: {args.aggregation}")
    print(f"Malicious clients: {sorted(list(malicious))}")
    print(f"Global model saved at: {Path(cfg['model']['global_out']).resolve()}")

    # 7) evaluate (if enabled)
    if cfg["eval"]["run_after_experiment"]:
        eval_cmd = [
            sys.executable, str((repo_root / "evaluate.py").resolve()),
            "--data", cfg["eval"]["data_yaml"],
            "--baseline", cfg["eval"]["baseline_model"],
            "--attacked", cfg["eval"]["attacked_model"],
            "--defended", cfg["eval"]["defended_model"],
            "--device", cfg["train"]["device"],
            "--asr_src_class_id", str(cfg["eval"].get("asr_src_class_id", 0)),
            "--asr_target_class_id", str(cfg["eval"].get("asr_target_class_id", cfg["eval"].get("poisoned_class_id", 16))),
            "--asr_iou", str(cfg["eval"].get("asr_iou", 0.5)),
            "--asr_mode", str(cfg["eval"].get("asr_mode", "strict")),
        ]
        if bool(cfg["eval"].get("asr_trigger", False)):
            eval_cmd.append("--asr_trigger")
            eval_cmd += ["--trigger_size", str(int(cfg["eval"].get("trigger_size", 16)))]
            eval_cmd += ["--trigger_value", str(int(cfg["eval"].get("trigger_value", 255)))]
            eval_cmd += ["--trigger_position", str(cfg["eval"].get("trigger_position", "bottom_right"))]
        subprocess.run(eval_cmd, check=False, cwd=str(repo_root))


if __name__ == "__main__":
    main()

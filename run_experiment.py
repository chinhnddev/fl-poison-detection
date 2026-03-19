import argparse
import math
import random
import subprocess
import sys
import time
from pathlib import Path
import socket
import yaml


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_clients", type=int, default=10)
    ap.add_argument("--malicious_ratio", type=float, default=0.4)
    ap.add_argument("--aggregation", choices=["fedavg", "krum"], default="fedavg")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--log_dir", default="./logs", help="Write server/client logs here")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    seed = int(((cfg.get("runtime") or {}).get("seed")) or 1234)
    random.seed(seed)
    host, port = cfg["server"]["host"], cfg["server"]["port"]
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    if not _is_port_available(host, int(port)):
        new_port = _pick_free_port(host)
        print(f"Port {host}:{port} is in use, switching to {host}:{new_port}")
        port = new_port

    # 1) partition dataset (required for true FL)
    if (cfg.get("federated") or {}).get("auto_partition", True):
        cmd = [
            sys.executable, "data_partition.py",
            "--data_yaml", cfg["dataset"]["base_data_yaml"],
            "--num_clients", str(args.num_clients),
            "--out_dir", cfg["federated"]["data_dir"],
            "--seed", str(seed),
            "--partition", str((cfg.get("federated") or {}).get("partition", "dirichlet")),
            "--dirichlet_alpha", str((cfg.get("federated") or {}).get("dirichlet_alpha", 0.5)),
            "--val_ratio", str((cfg.get("dataset") or {}).get("val_ratio", 0.2)),
        ]
        subprocess.run(cmd, check=True)

    # 4) malicious assignment
    k = math.floor(args.num_clients * args.malicious_ratio)
    malicious = set(random.sample(range(args.num_clients), k)) if k > 0 else set()

    # 2) launch server
    server_cmd = [
        sys.executable, "server.py",
        "--host", host, "--port", str(port),
        "--rounds", str(args.rounds),
        "--aggregation", args.aggregation,
        "--config", args.config,
        "--expected_clients", str(args.num_clients),
    ]
    server = None
    clients = []
    bad = []
    server_log = None
    client_logs = []
    try:
        server_log = open(log_dir / "server.log", "w", encoding="utf-8")
        server = subprocess.Popen(server_cmd, stdout=server_log, stderr=subprocess.STDOUT, text=True)
        # Server startup can take time (imports, weights load). Wait for the port to accept connections.
        if not _wait_for_server(host, int(port), timeout_s=90.0):
            server_log.flush()
            raise SystemExit(f"Server did not become ready on {host}:{port} within timeout")
        if server.poll() is not None:
            server_log.flush()
            raise SystemExit(f"Server failed to start on {host}:{port} (exit={server.returncode})")

        # 3) launch clients
        for cid in range(args.num_clients):
            clog = open(log_dir / f"client_{cid}.log", "w", encoding="utf-8")
            client_logs.append(clog)
            c = subprocess.Popen([
                sys.executable, "client.py",
                "--cid", str(cid),
                "--server_address", f"{host}:{port}",
                "--config", args.config,
                "--malicious", "1" if cid in malicious else "0",
            ], stdout=clog, stderr=subprocess.STDOUT, text=True)
            clients.append(c)

        # 5) wait rounds finish
        for c in clients:
            c.wait()
        bad = [i for i, p in enumerate(clients) if p.returncode not in (0, None)]
    finally:
        # 6) stop server (even if clients failed)
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
            sys.executable, "evaluate.py",
            "--data", cfg["eval"]["data_yaml"],
            "--baseline", cfg["eval"]["baseline_model"],
            "--attacked", cfg["eval"]["attacked_model"],
            "--defended", cfg["eval"]["defended_model"],
            "--device", cfg["train"]["device"],
            "--asr_src_class_id", str(cfg["eval"].get("asr_src_class_id", 0)),
            "--asr_target_class_id", str(cfg["eval"].get("asr_target_class_id", cfg["eval"].get("poisoned_class_id", 16))),
            "--asr_iou", str(cfg["eval"].get("asr_iou", 0.5)),
        ]
        subprocess.run(eval_cmd, check=False)


if __name__ == "__main__":
    main()

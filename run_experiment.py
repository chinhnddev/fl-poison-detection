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


def _load_partition_manifest(path: Path) -> dict:
    try:
        return yaml.safe_load(open(path, "r", encoding="utf-8")) or {}
    except Exception:
        return {}


def _echo_client_progress(
    log_dir: Path,
    client_cids: set[int],
    offsets: dict[int, int],
    announced: set[tuple[int, str]],
) -> None:
    """Mirror important per-client progress lines to the main console."""
    for cid in sorted(client_cids):
        log_path = log_dir / f"client_{cid}.log"
        if not log_path.exists():
            continue
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(offsets.get(cid, 0))
                chunk = f.read()
                offsets[cid] = f.tell()
        except Exception:
            continue
        if not chunk:
            continue
        for raw_line in chunk.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            event = None
            if "poison_ready" in line:
                event = "poison_ready"
            elif "FIT_START" in line:
                event = "fit_start"
            elif "FIT_END" in line:
                event = "fit_end"
            if event is None:
                continue
            key = (cid, line)
            if key in announced:
                continue
            print(f"[client {cid}] {line}", flush=True)
            announced.add(key)


def main():
    repo_root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_clients", type=int, default=10)
    ap.add_argument("--malicious_ratio", type=float, default=0.4)
    ap.add_argument(
        "--malicious_selection",
        choices=["random", "topk_src_class", "fixed"],
        default="random",
        help=(
            "How to choose malicious clients. topk_src_class uses partition_stats.yaml and "
            "attack.(label_flip|backdoor).src_class_id; fixed uses attack.malicious_clients."
        ),
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
    manifest_path = data_dir / "partition_manifest.json"
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
    else:
        manifest = _load_partition_manifest(manifest_path)
        print(f"Reusing partition at: {data_dir.resolve()}")
        if manifest:
            manifest_clients = int(manifest.get("num_clients", 0) or 0)
            manifest_partition = str(manifest.get("partition", "") or "")
            manifest_seed = manifest.get("seed", None)
            manifest_val_ratio = manifest.get("val_ratio", None)
            print(
                "Partition manifest -> "
                f"num_clients={manifest_clients} partition={manifest_partition or '?'} "
                f"seed={manifest_seed if manifest_seed is not None else '?'} "
                f"val_ratio={manifest_val_ratio if manifest_val_ratio is not None else '?'}"
            )
            if manifest_clients and manifest_clients < int(args.num_clients):
                raise SystemExit(
                    f"Shared partition only has {manifest_clients} clients, but this run requested {args.num_clients}. "
                    f"Rebuild {data_dir.resolve()} with at least {args.num_clients} clients."
                )
            cfg_partition = str((cfg.get("federated") or {}).get("partition", "") or "")
            cfg_val_ratio = (cfg.get("dataset") or {}).get("val_ratio", None)
            cfg_seed = ((cfg.get("runtime") or {}).get("seed")) or None
            mismatch = []
            if manifest_partition and cfg_partition and manifest_partition != cfg_partition:
                mismatch.append(f"partition manifest={manifest_partition} config={cfg_partition}")
            if manifest_val_ratio is not None and cfg_val_ratio is not None and float(manifest_val_ratio) != float(cfg_val_ratio):
                mismatch.append(f"val_ratio manifest={manifest_val_ratio} config={cfg_val_ratio}")
            if manifest_seed is not None and cfg_seed is not None and int(manifest_seed) != int(cfg_seed):
                mismatch.append(f"seed manifest={manifest_seed} config={cfg_seed}")
            if mismatch:
                print("WARNING: shared partition settings differ from current config; existing shard data will be reused.")
                for msg in mismatch:
                    print(f"  - {msg}")

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

    def _choose_malicious_fixed() -> set[int]:
        attack_cfg = cfg.get("attack") or {}
        raw = attack_cfg.get("malicious_clients", attack_cfg.get("malicious_cids", []))
        if isinstance(raw, str):
            items = [x.strip() for x in raw.replace(";", ",").split(",") if x.strip()]
        elif isinstance(raw, (list, tuple, set)):
            items = list(raw)
        else:
            raise SystemExit("attack.malicious_clients must be a list or comma-separated string when malicious_selection=fixed")
        try:
            fixed = {int(x) for x in items}
        except Exception as exc:
            raise SystemExit(f"Invalid attack.malicious_clients value: {raw!r}") from exc
        if not fixed:
            raise SystemExit("malicious_selection=fixed requires attack.malicious_clients to contain at least one client id")
        invalid = sorted(cid for cid in fixed if cid < 0 or cid >= args.num_clients)
        if invalid:
            raise SystemExit(f"Invalid malicious client id(s) for num_clients={args.num_clients}: {invalid}")
        if len(fixed) != k:
            print(
                f"Using fixed malicious clients {sorted(fixed)} "
                f"(count={len(fixed)}; malicious_ratio would select {k})",
                flush=True,
            )
        return fixed

    # Config may override CLI selection.
    cfg_sel = str(((cfg.get("attack") or {}).get("malicious_selection")) or "").strip().lower()
    sel = args.malicious_selection
    if cfg_sel in {"random", "topk_src_class", "topk_src_class_id", "topk_src", "fixed"}:
        if cfg_sel.startswith("topk"):
            sel = "topk_src_class"
        else:
            sel = cfg_sel

    if k > 0:
        if sel == "topk_src_class":
            malicious = _choose_malicious_topk_src_class()
        elif sel == "fixed":
            malicious = _choose_malicious_fixed()
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
    round_stats_path = log_dir / "round_stats.jsonl"
    if round_stats_path.exists():
        round_stats_path.unlink()

    # 2) launch server
    server_cmd = [
        sys.executable, str((repo_root / "server.py").resolve()),
        "--host", host, "--port", str(port),
        "--rounds", str(args.rounds),
        "--aggregation", args.aggregation,
        "--config", str(config_path),
        "--expected_clients", str(args.num_clients),
        "--round_stats_out", str(round_stats_path.resolve()),
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
        honest = sorted(set(range(args.num_clients)) - malicious)
        print(
            f"Progress mirror: tracking all clients 0..{args.num_clients - 1} "
            f"(malicious={sorted(malicious)}, honest={honest})",
            flush=True,
        )
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
        tracked_clients = set(range(args.num_clients))
        client_offsets = {cid: 0 for cid in tracked_clients}
        client_announced: set[tuple[int, str]] = set()
        wait_started = time.time()
        timed_out = False
        try:
            rt = float(args.run_timeout_s)
            while True:
                _echo_client_progress(log_dir, tracked_clients, client_offsets, client_announced)
                if server.poll() is not None:
                    break
                if rt > 0 and (time.time() - wait_started) > rt:
                    timed_out = True
                    break
                time.sleep(2.0)
            if timed_out:
                print("Server timeout -> check logs")
                try:
                    server_log.flush()
                except Exception:
                    pass
                tail = _tail_text(server_log_path)
                if tail:
                    print("--- server.log (tail) ---")
                    print(tail)
            else:
                _echo_client_progress(log_dir, tracked_clients, client_offsets, client_announced)
                server.wait(timeout=0.1)
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

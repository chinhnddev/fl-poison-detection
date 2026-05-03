import argparse

from federated.server_app import run_server


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--start_round", type=int, default=1)
    ap.add_argument("--resume_weights", default="", help="Resume global model from this .pt checkpoint")
    ap.add_argument("--aggregation", choices=["fedavg"], default="fedavg")  # research experiments use FedAvg
    ap.add_argument("--config", default="config.baseline.yaml")
    ap.add_argument("--expected_clients", type=int, default=0)
    ap.add_argument("--round_stats_out", default="", help="Optional JSONL output for per-round stats")
    ap.add_argument("--append_round_stats", action="store_true", help="Append to round_stats_out instead of overwriting")
    args = ap.parse_args()

    run_server(
        host=str(args.host),
        port=int(args.port),
        rounds=int(args.rounds),
        start_round=int(args.start_round),
        cfg_path=str(args.config),
        resume_weights=str(args.resume_weights),
        expected_clients=int(args.expected_clients),
        round_stats_out=str(args.round_stats_out),
        append_round_stats=bool(args.append_round_stats),
    )


if __name__ == "__main__":
    main()

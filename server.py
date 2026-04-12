import argparse

from federated.server_app import run_server


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--aggregation", choices=["fedavg"], default="fedavg")  # research experiments use FedAvg
    ap.add_argument("--config", default="config.baseline.yaml")
    ap.add_argument("--expected_clients", type=int, default=0)
    ap.add_argument("--round_stats_out", default="", help="Optional JSONL output for per-round stats")
    args = ap.parse_args()

    run_server(
        host=str(args.host),
        port=int(args.port),
        rounds=int(args.rounds),
        cfg_path=str(args.config),
        expected_clients=int(args.expected_clients),
        round_stats_out=str(args.round_stats_out),
    )


if __name__ == "__main__":
    main()

import argparse
import logging

import flwr as fl
import yaml

from federated.client_app import YoloDeltaClient


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cid", type=int, required=True)
    ap.add_argument("--server_address", required=True)
    ap.add_argument("--config", default="config.baseline.yaml")
    ap.add_argument("--malicious", type=int, default=0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True)

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    client = YoloDeltaClient(args.cid, cfg, bool(args.malicious))
    fl.client.start_numpy_client(server_address=args.server_address, client=client)


if __name__ == "__main__":
    main()

"""Run the local masked-update coordinator.

This is a protocol demonstrator, not a production federated server. All
expected clients must submit before a masked sum can be recovered.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fedforge.distributed import CoordinatorServer
from fedforge.security import random_secret


def main() -> int:
    parser = argparse.ArgumentParser(description="FedForge local secure-aggregation coordinator")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--clients", required=True, help="Comma-separated expected client ids")
    parser.add_argument("--secret", default=os.environ.get("FEDFORGE_SHARED_SECRET"))
    args = parser.parse_args()
    secret = args.secret or random_secret()
    clients = tuple(item.strip() for item in args.clients.split(",") if item.strip())
    if not clients:
        parser.error("--clients must contain at least one client id")
    print("Shared secret for local clients:", secret)
    print("Waiting for", len(clients), "masked updates on", f"http://{args.host}:{args.port}/update")
    server = CoordinatorServer((args.host, args.port), secret, clients)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

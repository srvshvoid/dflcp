"""Send one signed, pairwise-masked update to a local coordinator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fedforge.distributed import UpdateEnvelope, post_update


def main() -> int:
    parser = argparse.ArgumentParser(description="FedForge local masked-update client")
    parser.add_argument("--url", default="http://127.0.0.1:8765/update")
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--peers", required=True, help="Comma-separated client ids")
    parser.add_argument("--secret", required=True)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--num-examples", type=int, default=1)
    parser.add_argument("--parameters", required=True, help="JSON array of model parameters")
    args = parser.parse_args()
    parameters = json.loads(args.parameters)
    if not isinstance(parameters, list):
        parser.error("--parameters must be a JSON array")
    envelope = UpdateEnvelope.create(
        args.client_id,
        args.round,
        parameters,
        args.num_examples,
        args.secret,
        peers=tuple(item.strip() for item in args.peers.split(",") if item.strip()),
        secure_aggregation=True,
    )
    print(json.dumps(post_update(args.url, envelope)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

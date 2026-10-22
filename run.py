"""Launch the FedForge Live Server and Interactive Dashboard."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure 'src' is always in python path
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
os.environ["PYTHONPATH"] = str(SRC_DIR)

import uvicorn
from fedforge.api.app import create_fastapi_app


def main() -> None:
    parser = argparse.ArgumentParser(description="FedForge Live Server and Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable live auto-reload")
    args = parser.parse_args()

    app = create_fastapi_app(runs_root="runs")

    print("\n" + "=" * 65)
    print("   FedForge: Federated Cross-Platform Deepfake Detection")
    print("=" * 65)
    print(f" -> Web Dashboard:        http://{args.host}:{args.port}/")
    print(f" -> Interactive API Docs: http://{args.host}:{args.port}/docs")
    print(f" -> Federation Engine:    http://{args.host}:{args.port}/federation/state")
    print("=" * 65 + "\n")

    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()

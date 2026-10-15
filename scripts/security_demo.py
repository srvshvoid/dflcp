"""Run the labelled, synthetic security-lab replay sequence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fedforge.simulator import run_smoke_experiment


def run_security_demo(output: Path | str = "runs/security-demo") -> Path:
    """Run clean, attack, and SecAgg-off median replay stages.

    The resulting manifest records actual artifacts but deliberately does not
    assert that the toy attack succeeds or that the defense recovers recall.
    Those are research hypotheses requiring real data and pre-registered runs.
    """

    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    stages = (
        ("clean", Path("configs/security/clean.json")),
        ("attack", Path("configs/security/attack.json")),
        ("defense", Path("configs/security/defense.json")),
    )
    records: list[dict[str, object]] = []
    for name, config in stages:
        run_dir = root / name
        result = run_smoke_experiment(config, run_dir)
        records.append(
            {
                "stage": name,
                "run_dir": run_dir.as_posix(),
                "config": config.as_posix(),
                "rounds": len(result.metrics),
                "final_metrics": result.metrics[-1],
                "evidence_status": "synthetic-unverified",
            }
        )
    manifest = {
        "mode": "synthetic-replay",
        "privacy_robustness_boundary": "defense stage is SecAgg-off because it inspects plaintext updates",
        "success_claim": "not evaluated by this script",
        "stages": records,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/security-demo")
    args = parser.parse_args()
    print(run_security_demo(args.output))
    print("Synthetic security replay only; no attack-success or recovery claim was made.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

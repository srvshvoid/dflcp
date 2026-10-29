"""Evaluate the live detector on a cross-dataset real/fake holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fedforge.vision.detector import MediaDetectionPipeline


def evaluate(
    dataset: Path, runs_root: Path, checkpoint: Path | None = None
) -> dict[str, object]:
    pipeline = MediaDetectionPipeline(
        runs_root=runs_root,
        checkpoint_path=checkpoint,
    )
    rows = []
    for label_name, label in (("real", 0), ("fake", 1)):
        for image in sorted((dataset / label_name).glob("*")):
            if image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            result = pipeline.analyze_image_bytes(image.read_bytes(), image.name)
            rows.append(
                {
                    "file": str(image),
                    "label": label,
                    "score": result["overall_score"],
                    "verdict": result["verdict"],
                }
            )
    if not rows:
        raise ValueError(f"No supported images found below {dataset}")
    confusion = {"real_as_real": 0, "real_as_fake": 0, "fake_as_real": 0, "fake_as_fake": 0}
    for row in rows:
        predicted_fake = row["verdict"] == "LIKELY_SYNTHETIC"
        if row["label"] == 0 and not predicted_fake:
            confusion["real_as_real"] += 1
        elif row["label"] == 0:
            confusion["real_as_fake"] += 1
        elif predicted_fake:
            confusion["fake_as_fake"] += 1
        else:
            confusion["fake_as_real"] += 1
    decided = sum(
        row["verdict"] != "INCONCLUSIVE" for row in rows
    )
    accuracy = sum(
        (row["label"] == 1) == (row["verdict"] == "LIKELY_SYNTHETIC")
        for row in rows
        if row["verdict"] != "INCONCLUSIVE"
    ) / max(1, decided)
    return {
        "checkpoint": pipeline.checkpoint_path,
        "thresholds": {
            "authentic": min(0.35, pipeline.threshold),
            "synthetic": 0.95,
        },
        "authentic_threshold": 0.35,
        "synthetic_threshold": 0.95,
        "sample_count": len(rows),
        "inconclusive_count": sum(row["verdict"] == "INCONCLUSIVE" for row in rows),
        "coverage": round(decided / len(rows), 4),
        "decided_accuracy": round(accuracy, 4),
        "confusion": confusion,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/external/real_fake_images"))
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.dataset, args.runs_root, args.checkpoint)
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

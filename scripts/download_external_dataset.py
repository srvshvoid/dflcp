"""Download a reproducible, licensed real/fake image holdout from Hugging Face.

The default keeps the download small enough for a project workstation while
still testing cross-dataset generalisation.  The source repository is
Apache-2.0 licensed and stores labels as ``real`` and ``fake`` folders.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen


DATASET = "Madushan996/real_fake_images"
API_URL = f"https://huggingface.co/api/datasets/{DATASET}"
RESOLVE_URL = f"https://huggingface.co/datasets/{DATASET}/resolve/main"


def download(output: Path, per_class: int) -> dict[str, int]:
    request = Request(API_URL, headers={"User-Agent": "FedForge/1.0"})
    with urlopen(request, timeout=60) as response:
        metadata = json.load(response)
    files = [item["rfilename"] for item in metadata["siblings"]]
    counts: dict[str, int] = {}
    source_prefix = {"real": "real/", "fake": "fake_fx/"}
    for label in ("real", "fake"):
        selected = [name for name in files if name.startswith(source_prefix[label])][:per_class]
        target = output / label
        target.mkdir(parents=True, exist_ok=True)
        for index, filename in enumerate(selected, 1):
            destination = target / Path(filename).name
            if not destination.is_file():
                request = Request(
                    f"{RESOLVE_URL}/{filename}",
                    headers={"User-Agent": "FedForge/1.0"},
                )
                with urlopen(request, timeout=120) as response:
                    destination.write_bytes(response.read())
            if index % 50 == 0:
                print(f"{label}: {index}/{len(selected)}")
        counts[label] = len(selected)
    (output / "SOURCE.json").write_text(
        json.dumps(
            {
                "dataset": DATASET,
                "license": "Apache-2.0",
                "labels": {"real": 0, "fake": 1},
                "counts": counts,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/external/real_fake_images"))
    parser.add_argument("--per-class", type=int, default=200)
    args = parser.parse_args()
    if args.per_class <= 0:
        raise SystemExit("--per-class must be positive")
    print(download(args.output, args.per_class))


if __name__ == "__main__":
    main()

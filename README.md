# FedForge

FedForge is a reproducible shell for a federated-learning research prototype.
This repository currently packages an offline, CPU-only smoke path and a
static replay viewer. It is deliberately honest about the evidence boundary:
the smoke path is synthetic plumbing, not a research benchmark.

## Current status

The following are **pending** and must not be inferred from a successful local
smoke command:

- licensed, documented **real datasets** and their verification manifests;
- a confirmed **GPU** environment and the intended CNN/ViT model stack;
- measured **Flower** federation and Flower SecAgg+ integration;
- the production **Next.js** dashboard and live API path;
- **PostgreSQL** persistence and **Redis**/queue deployment.
- image/video multipart inference, face crops, calibration, and explanation maps.

No benchmark number, generalization claim, privacy guarantee, attack result, or
deployment capacity is reported by this package. See
[`docs/RESULTS.md`](docs/RESULTS.md) for the reporting rule.

## Requirements and reproducibility

- Linux or another POSIX environment with `make`;
- Python **3.11.2** (the version is pinned in [`.python-version`](.python-version));
- Docker is optional for the static viewer image.

The CPU smoke path uses only the Python standard library. The intentionally
empty [`requirements.lock`](requirements.lock) is still checked in as the
review point for future dependencies; optional research/deployment stacks are
not silently installed. Use the same Python patch release as the pin rather
than relying on a different interpreter that happens to be called `python3`.

## Setup

From the repository root, run:

```sh
python3 --version                 # must print Python 3.11.2
make setup                         # creates .venv and verifies the empty lockfile, offline
```

`make setup` uses `--no-index`, `--no-deps`, and `--no-build-isolation`; it
does not download packages. The Makefile runs the source through
`PYTHONPATH=src`, so no editable-install build backend or wheel is needed for
the CPU smoke slice. The source implementation and tests are expected at
`src/fedforge/` and `tests/`. If either directory is missing,
`make test`/`make smoke` stop with an explicit message instead of reporting a
pass.

## Test, smoke, and demo commands

Run the standard-library test suite:

```sh
make test
```

Run the deterministic CPU smoke command and write its replay artifacts to
`runs/smoke/`:

```sh
make smoke
```

The command must write `metrics.jsonl` and `provenance.json`. The values are
synthetic replay records only. A different output directory or seed can be
selected without editing files:

```sh
make smoke RUN_DIR=runs/another-replay
```

The resolved synthetic configuration is [`configs/smoke.json`](configs/smoke.json);
keep any experiment-specific configuration separate and review it before use.

Prepare the labelled replay used by the demo script:

```sh
make demo
make demo-security                # labelled synthetic clean/attack/defense replay
make report                       # optional, writes docs/RESULTS.generated.md
make reproduce RQ=1              # validates a preregistered contract; real run pending
```

`make demo` does not open a browser or claim a live service. It creates
`runs/demo/metrics.jsonl` through the same smoke contract and prints the next
command. Serve the dependency-free dashboard in a second terminal:

```sh
make run PORT=8000
```

Open <http://127.0.0.1:8000>, choose the generated `metrics.jsonl`, and keep
the `SYNTHETIC REPLAY — NOT A BENCHMARK` banner visible during any recording.
The full talk track and failure cases are in
[`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md). `make demo-security` runs three
synthetic stages (clean, semantic label-flip, and SecAgg-off coordinate
median), records their actual artifacts, and explicitly makes no claim that
the toy attack succeeds or that the defense recovers recall.

## Packaging and checks

These commands run real local checks and do not install a linter or fabricate
metrics:

```sh
make lint       # Python bytecode compilation check
make check      # lint plus packaging/docs/dashboard/file checks
```

`make check` does not replace `make test`; run both for a source checkout.
The `pyproject.toml` has no runtime dependencies and discovers packages below
`src/`. `make report` reads only actual run artifacts and writes a clearly
labelled replay index to the ignored file `docs/RESULTS.generated.md`; it never
turns synthetic values into a benchmark table.


## Optional read-only API boundary

The standard-library `fedforge.api.RunRepository` reads the same provenance and
JSONL artifacts without a database. The optional `create_fastapi_app` adapter
exposes `/health`, `/runs`, `/runs/{id}`, `/runs/{id}/rounds`, and the explicit
`/privacy/payload-manifest` contract when a separately reviewed FastAPI stack
is installed. `/detect` returns a documented `501` for image/video uploads in
this slice; `/detect/smoke` is numeric-feature checkpoint plumbing only. It must
not be described as the requested media inference service.

Pre-registered research contracts live under [`configs/`](configs/), including
RQ1-RQ5 hypotheses, null hypotheses, and success criteria. They contain no
measured result.

## Container build and local deployment

The Dockerfile is pinned to `python:3.11.2-slim`, installs the local package
without network access, and serves only the static replay viewer by default:

```sh
docker build --tag fedforge:cpu-smoke .
docker run --rm --publish 8000:8000 fedforge:cpu-smoke
```

Then visit <http://127.0.0.1:8000>. This is a local dashboard container, not a
claim that the pending Next.js/PostgreSQL/Redis production deployment exists.
Do not upload real sensitive data to this static viewer.

## Repository map

- [`Makefile`](Makefile) — offline setup, test, smoke, demo, lint, check, and
  static-server commands.
- [`configs/smoke.json`](configs/smoke.json) — the explicitly synthetic smoke
  configuration consumed by `make smoke`.
- [`dashboard/index.html`](dashboard/index.html) — dependency-free local JSONL
  replay viewer with explicit synthetic/unverified labeling.
- [`docs/ASSUMPTIONS.md`](docs/ASSUMPTIONS.md) — original scope and gate
  assumptions; preserved as the project decision record.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — verified slice and pending
  system boundaries.
- [`docs/DATASET_VERIFICATION.md`](docs/DATASET_VERIFICATION.md) — evidence
  checklist and unfilled dataset manifest placeholder.
- [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md) — assets, actors, boundaries,
  and limitations.
- [`docs/ETHICS.md`](docs/ETHICS.md) — responsible research and misuse
  constraints.
- [`docs/LITERATURE_VERIFICATION.md`](docs/LITERATURE_VERIFICATION.md) —
  citation-verification protocol; no unverified references are presented as
  support.
- [`docs/RESULTS.md`](docs/RESULTS.md) — what may and may not be reported.
- [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md) — exact demo sequence and
  honest narration.

## Contributing evidence

Add a dependency only after pinning it in `requirements.lock` and proving that
the CPU smoke path remains runnable offline. Add a dataset only with a filled
verification record, license evidence, disjoint split checks, and an approved
retention policy. Replace a `pending` result with a measured result only when
the run directory contains the configuration, seed, provenance, raw metrics,
and independently checkable artifact references.

# Gate status and evidence

This status page records the boundary of the current implementation. A
**partial** gate is not a claim that the full master prompt has passed.
Synthetic smoke artifacts are ignored by git and are not research results.

## Evidence captured in this checkout

| Check | Command | Observed result |
| --- | --- | --- |
| Runtime/setup | `make setup` | Python 3.11.2 accepted; empty runtime lock installed with `--no-index` |
| Unit/integration slice | `make test` | 11 standard-library tests passed |
| Source/packaging checks | `make lint && make check` | Python compilation, TOML parse, required-file and pin checks passed |
| Deterministic smoke | `make smoke RUN_DIR=runs/final-check` | provenance, JSONL metrics, and checkpoint written; synthetic replay label printed |
| Reproducibility | direct `run_smoke_experiment` comparison | two same-seed `metrics.jsonl` files were byte-for-byte identical (502 bytes) |
| Dashboard HTTP | local `http.server` probe | HTTP 200 and the synthetic-replay banner were present |
| Security replay | `make demo-security DEMO_RUN_DIR=runs/final` | clean/attack/defense stages and manifest written; no attack-success claim made |
| Report plumbing | `make report` | generated ignored `docs/RESULTS.generated.md` from actual local run artifacts |
| Container build | `docker build ...` | not run: Docker is not installed in the current environment |
| Ruff/Black/Mypy | `python -m ruff`, `python -m black`, `python -m mypy` | not run: tools are not installed; `make lint` is compile-only |

## Gate matrix

| Gate | Status | What is actually present | Required before marking complete |
| --- | --- | --- | --- |
| A — Data | Partial | Deterministic numeric fixture, video/source split checks, synthetic manifests in memory | Licensed dataset records, face extraction, perceptual-hash and DF40 overlap audit |
| B — Centralized model | Partial | Deterministic logistic frame baseline, video mean pooling, numeric checkpoint boundary | CNN/ViT, real media preprocessing, calibration and measured image/video inference |
| C — Federated core | Partial | CPU FedAvg/FedProx/personalized-head simulator, payload audit, robust primitives | Pinned Flower simulation/deployment and live metrics/WebSocket/database path |
| D — Privacy | Partial | Explicit SecAgg/robust compatibility validator and payload schema | Flower SecAgg+ measurement, DP accountant, clipping/quantization utility curves |
| E — Security | Partial | Targeted semantic label flip, trimmed mean/median/Krum/norm clipping, labelled replay | Real-data attack matrix, trigger/adaptive attacks, live defense switching and measured recovery |
| F — Dashboard | Partial | Dependency-free replay viewer and optional read-only API adapter | Next.js dashboard, live WebSocket, detection console, Playwright e2e |
| G — Experiments | Partial | RQ1–RQ5 preregistration YAML and transparent report generator | At least three real-data seeds, video bootstrap CIs, baselines and ablations |
| H — Packaging | Partial | Pinned Python, lockfile, Make targets, Dockerfile, compose dev profile, CI workflow | Verify Docker/CI and add full service profiles without overstating readiness |

The next smallest material step is to verify authorized dataset access and the
installed Flower/PyTorch versions before changing the smoke model into a real
media pipeline.

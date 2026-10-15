# FedForge assumptions and implementation plan

## Scope decision

The requested system spans a research platform, a production-style service, a
dashboard, several protected federated-learning protocols, and experiments
that require licensed datasets and GPU time. The current workspace contains no
source code, datasets, credentials, or evidence of a GPU. It is therefore not
honest to claim that Gates A-H or the centerpiece demo are complete in one
offline build step.

This first implementation is a reproducible, CPU-capable vertical slice. It
uses deterministic synthetic face-like feature records for smoke tests and
implements the research contracts that can be verified without private
datasets: video-level partitioning/leakage checks, a small typed detector,
FedAvg/FedProx/personalized-head simulation, plaintext robust aggregators,
SecAgg compatibility validation, a targeted label-flip attack, metrics,
calibration, run provenance, a FastAPI-compatible API boundary, and a replay
dashboard. Real dataset preprocessing, Flower SecAgg+ wiring, Opacus
accounting, GPU inference, PostgreSQL/Redis deployment, and measured DF40
generalization remain explicitly gated follow-up work.

## Decisions and constraints

1. **Runtime:** Python 3.11 is available in the container. `.python-version`
   pins `3.11.2`; dependencies are kept small and pinned in `requirements.lock`.
   The smoke path uses the Python standard library so it can run offline.
2. **Model:** the smoke model is a deterministic logistic frame classifier
   with mean-pooling to a video score. It is an interface-compatible stand-in,
   not the requested CNN/ViT forensic model. A real `timm`/PyTorch model must
   be added only after dataset access and GPU budget are confirmed.
3. **Federation:** the simulator is intentionally dependency-free and makes
   the privacy/robustness boundary explicit. Individual plaintext updates are
   available only when SecAgg is disabled. No claim is made that this is
   Flower SecAgg+.
4. **Data:** no dataset facts, counts, licenses, or benchmark metrics are
   asserted here. The synthetic smoke fixture is not a research result and is
   never used to populate `docs/RESULTS.md` as a benchmark.
5. **Experiments:** every generated run records its resolved config, seed,
   git hash, environment lock hash, JSONL metrics, and a checkpoint. The
   report generator refuses to invent results and marks missing real-data
   experiments as pending.
6. **Dashboard/API:** the initial dashboard is a static, dependency-free
   replay viewer and the API is an optional FastAPI adapter. They consume run
   JSONL rather than claiming a live database/WebSocket deployment.
7. **Security:** the smoke attack is a targeted semantic label flip. Trimmed
   mean, median, Krum, and norm clipping are implemented for toy updates, with
   honest-client rejection recorded. Trigger backdoors, adaptive attackers,
   and production threat mitigations are not claimed.

## Gate plan

- **A:** synthetic deterministic manifests and leakage checker now; licensed
  dataset manifests and DF40 disjointness are pending.
- **B:** deterministic CPU baseline now; CNN/ViT, face extraction, calibration
  on real validation videos, and image/video inference are pending.
- **C:** simulator and payload audit now; Flower simulation/deployment and
  database/WebSocket streaming are pending.
- **D:** compatibility/configuration checks now; measured Flower SecAgg+ and
  client/sample-level DP runs are pending.
- **E:** toy targeted attack/defense metrics now; live under-five-minute demo
  and attack evidence on real data are pending.
- **F:** replay viewer/API contract now; full Next.js dashboard and Playwright
  demo are pending.
- **G:** experiment config/report plumbing now; RQ1-RQ5 with three or more
  real-data seeds and bootstrap confidence intervals are pending.
- **H:** offline `make setup`, `make test`, `make smoke`, and `make demo` are
  the initial packaging target. Docker and CI are included as guarded
  follow-up targets and must not be reported as passing until run.

## Acceptance criteria for this slice

- A clean Python 3.11 checkout can run `make test` without downloading data.
- `make smoke` creates a run directory containing provenance and JSONL metrics,
  and re-running it with the same seed reproduces the same metrics.
- The leakage checker rejects video/source overlap and the payload audit
  rejects local-head or raw-media fields.
- A SecAgg-on configuration rejects plaintext robust aggregators and a
  SecAgg-off configuration accepts them.
- README and dashboard wording distinguishes synthetic replay evidence from
  real research results.

# FedForge architecture

## Final-project implementation boundary

The repository now contains a working FastAPI media service, a trained
ResNet-18 image classifier, benchmark evaluation artifacts, federated
training/aggregation code, a dashboard, and regression tests. The upload
policy is deliberately three-way: clearly authentic, clearly synthetic, or
`INCONCLUSIVE` for borderline/out-of-distribution scores. This prevents a
binary model from presenting uncertain real-world uploads as facts.

The measured model evidence is still CIFAKE-based. Real-world camera and
face-swap reliability requires an additional licensed dataset and must not be
claimed from the CIFAKE metrics alone.

## Status boundary

FedForge is currently packaged around a dependency-free CPU smoke slice. The
architecture below separates what can be verified locally from integrations
that require data access, infrastructure, or a measured experiment. A box in
the pending section is not represented as running merely because its name
appears in a configuration file.

## Verified local path

```text
Python 3.11.2
    |
    v
deterministic CPU smoke command (src/fedforge/)
    |
    +--> provenance.json
    +--> metrics.jsonl
    |
    v
dashboard/index.html  -- local file picker -->  transparent replay table
```

The shell supplies the reproducibility boundary:

- `.python-version` fixes the interpreter patch version;
- `requirements.lock` records that the smoke path has no third-party
  dependency;
- `make setup` refuses an interpreter other than Python 3.11.2 and performs
  offline installation;
- `make smoke` requires both a provenance artifact and JSONL metrics before it
  reports that a replay was written;
- the dashboard reads a user-selected local file and renders its fields. It
  does not call a remote API or calculate a benchmark.

The source implementation owns the deterministic fixture, validation logic,
and metric schema. The packaging contract is intentionally narrow: `make
smoke` invokes the standard-library Python driver in the Makefile against
those source APIs, and the output files are `provenance.json` and
`metrics.jsonl`. This keeps the shell usable while the optional service/API
entry point remains pending.

## Local multi-process prototype

The repository now includes a bounded local-process protocol in
[`src/fedforge/distributed.py`](../src/fedforge/distributed.py), with
[`scripts/run_coordinator.py`](../scripts/run_coordinator.py) and
[`scripts/send_masked_update.py`](../scripts/send_masked_update.py). Clients
send only signed model envelopes; raw images, labels, and file paths remain
client-local. Pairwise additive masks cancel only when every expected client
submits, so the coordinator can recover an aggregate without receiving an
individual unmasked vector. The protocol is intentionally an educational
building block: it does not yet provide dropout recovery, key rotation,
weighted secure aggregation, replay protection across rounds, or a formal
cryptographic review. Put the HTTP endpoint behind TLS for transport
confidentiality; HMAC signatures provide integrity and authentication, not
encryption.

The PyTorch runner remains available as an in-process baseline. The process
protocol is separate so that its boundaries can be tested without pretending
that a sequential simulation is distributed deployment.

## Privacy controls

`run_fl.py` accepts `--dp-clip-norm` and `--dp-noise-multiplier`. These apply
client-update clipping and Gaussian noise before transmission and record the
parameters in run configuration. They are a client-level DP building block,
not a completed privacy guarantee: epsilon/delta accounting, sampling
amplification, composition, and a production privacy budget are still
required.

Set `FEDFORGE_API_TOKEN` to require Bearer authentication on mutating API
routes and sensitive federation metadata. Authentication is disabled when the
variable is absent to preserve the offline smoke and existing local tests;
shared or exposed deployments must set it.

## Planned research path

The following interfaces are architectural placeholders, not enabled services:

1. **Dataset boundary:** a licensed manifest and preprocessing job must produce
   verified video/source-level splits before any real-data run.
2. **Model boundary:** a selected CPU/GPU model and calibration protocol must
   be pinned separately from the synthetic smoke model.
3. **Federation boundary:** Flower clients, server orchestration, dropout
   recovery, key management, and robust aggregation need compatibility tests
   and payload audits. The local masked-update protocol is not Flower SecAgg+
   evidence.
4. **Persistence boundary:** PostgreSQL and Redis would be deployment
   dependencies for durable runs and streaming state. Neither is started by
   this repository.
5. **UI boundary:** a future Next.js application may consume a versioned API;
   the current HTML viewer is a static replay fallback and not that API.
6. **Evidence boundary:** results must retain config, seed, code revision,
   environment lock hash, raw metrics, and artifact references. A chart alone
   is not sufficient evidence.

## Trust and data flow

The local viewer treats replay JSONL as untrusted input. It uses DOM
`textContent` rather than interpreting loaded fields as HTML or JavaScript. It
does not upload the file. This is a presentation safeguard, not a guarantee
that an arbitrary future service will protect sensitive data.

No raw media is expected in a smoke replay. Real media, face crops, labels,
client updates, and model checkpoints require an explicit data governance
decision before they cross a local or hosted boundary. See
[`DATASET_VERIFICATION.md`](DATASET_VERIFICATION.md),
[`THREAT_MODEL.md`](THREAT_MODEL.md), and [`ETHICS.md`](ETHICS.md).

## API and experiment contract boundary

`src/fedforge/api/` provides a standard-library `RunRepository` for provenance
and JSONL metrics plus an optional FastAPI adapter. The adapter exposes only
read-only replay routes and an explicit numeric smoke endpoint; its media
`/detect` route returns `501` until file sniffing, sandboxed decode, face
extraction, calibration, and retention controls are implemented.

`configs/experiments/RQ1.yaml` through `RQ5.yaml` are preregistration records
for hypotheses, null hypotheses, and success criteria. They do not contain
results. `scripts/security_demo.py` records a synthetic three-stage replay and
keeps the SecAgg/plaintext-robustness incompatibility visible.

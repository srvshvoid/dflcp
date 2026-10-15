# FedForge architecture

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

## Planned research path

The following interfaces are architectural placeholders, not enabled services:

1. **Dataset boundary:** a licensed manifest and preprocessing job must produce
   verified video/source-level splits before any real-data run.
2. **Model boundary:** a selected CPU/GPU model and calibration protocol must
   be pinned separately from the synthetic smoke model.
3. **Federation boundary:** Flower clients, server orchestration, SecAgg+, and
   robust aggregation need compatibility tests and payload audits. A local
   plaintext toy update is not Flower SecAgg+ evidence.
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

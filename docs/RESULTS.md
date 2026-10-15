# Results and evidence policy

## Current result status

There are no real-dataset results in this repository. The permitted local
smoke artifact is a deterministic, synthetic replay used to exercise file
formats, provenance plumbing, and the dashboard boundary. It is not a
benchmark and must not be used to estimate model quality, generalization,
privacy, fairness, attack resistance, or production performance.

The following evidence areas remain pending:

| Area | Permitted status today | Missing evidence |
| --- | --- | --- |
| Real dataset evaluation | Pending | Verified license, manifest, disjoint splits, measured runs |
| GPU model comparison | Pending | Pinned model/environment and resource record |
| Flower federation / SecAgg+ | Pending | Integration, protocol, dropout, and payload evidence |
| Robustness / poisoning | Pending | Stated threat model and reproducible attack matrix |
| Privacy accounting | Pending | Mechanism parameters, accountant, and composition record |
| Next.js/API deployment | Pending | Versioned API, service tests, and deployment evidence |
| PostgreSQL/Redis operations | Pending | Schema, migrations, queue behavior, and recovery checks |

## What a valid result record requires

Before a real claim is written, retain:

- the exact dataset verification record and artifact hashes;
- configuration, code revision, Python/dependency lock information, and seed;
- raw per-run metrics and the aggregation/reporting script;
- split/leakage audit and exclusions;
- uncertainty or confidence procedure appropriate to the question;
- threat model, privacy parameters, and known limitations;
- enough independent reruns to support the stated conclusion.

The report must link to these artifacts. A screenshot from the static viewer,
an unlabeled table, or a single synthetic run is not sufficient.

## Reproducible smoke evidence

When the source implementation is present, the following commands are the
only local smoke evidence this package prepares:

```sh
make setup
make test
make smoke RUN_DIR=runs/smoke
make check
```

Inspect `runs/smoke/provenance.json` and `runs/smoke/metrics.jsonl`, then load
the latter in `dashboard/index.html`. Report the result as “synthetic CPU
smoke replay generated” and include the exact command. Do not copy its fields
into a real-results table.

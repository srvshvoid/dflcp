# FedForge threat model

## Purpose and scope

This is a planning threat model for the CPU smoke slice and the pending
federated system. It is not a security certification. The current static
viewer is a local file renderer; it is not a multi-tenant service and does not
claim to protect production secrets.

## Assets

- real video, face crops, labels, and source identities;
- client model updates and any client-local state;
- aggregation configuration, seeds, provenance, and checkpoints;
- experiment results and claims derived from them;
- credentials, dataset licenses, and access/deletion records.

## Trust boundaries

1. **Input boundary:** a dataset or replay file enters the local process. Files
   may be malformed, mislabeled, duplicated, or more sensitive than expected.
2. **Client boundary:** future federated clients are separate data holders and
   cannot be assumed honest merely because they join a run.
3. **Transport/aggregation boundary:** future Flower and SecAgg+ services
   would carry updates and control messages. The current smoke path is not
   that service and does not provide cryptographic secure aggregation.
4. **Storage boundary:** future PostgreSQL, Redis, object storage, and logs may
   retain metadata or sensitive artifacts. They are not provisioned here.
5. **Presentation boundary:** the static viewer renders local JSONL. It does
   not upload the file, but the surrounding browser and host remain in scope.

## Adversaries and failure modes

- an accidental source/video overlap that inflates evaluation;
- a malicious or compromised client sending poisoned, malformed, or extreme
  updates;
- an operator exposing raw updates, labels, media, or identifiers in logs;
- a user treating synthetic replay values as real benchmark evidence;
- a dependency or deployment compromise in a future service;
- an unauthorized reader of local run directories or copied dashboard files.

## Controls in this slice

- deterministic CPU fixtures and a pinned interpreter make reruns inspectable;
- the smoke shell requires provenance and JSONL artifacts rather than printing
  a fabricated success message;
- the dashboard labels synthetic replay data and displays loaded fields without
  ranking or calculating them;
- dataset verification, ethics, and results documents keep pending evidence
  visible;
- package setup uses an empty lockfile and offline pip flags, so optional
  network services are not silently pulled into the smoke path.

The local process prototype adds signed envelopes, optional pairwise masking,
optional client clipping/noise, and configurable API bearer authentication.
These controls do not provide a production differential-privacy budget,
dropout-tolerant secure aggregation, key rotation, authorization roles,
encryption unless TLS is configured, or deletion guarantees.

## Required controls before real deployment

- authenticated and authorized client/server communication;
- a reviewed Flower SecAgg+ design with key-management and dropout handling;
- privacy accounting for any DP mechanism, with parameters and composition;
- update validation and poisoning/robustness evaluation under stated threat
  assumptions;
- least-privilege storage, encryption, retention/deletion tests, and redacted
  logs;
- dependency/container scanning and reproducible release provenance;
- incident response and a process for correcting an overstated result.

See [`DATASET_VERIFICATION.md`](DATASET_VERIFICATION.md) and
[`ETHICS.md`](ETHICS.md) before enabling a real-data path.

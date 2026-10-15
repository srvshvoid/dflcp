# FedForge demo script

This script demonstrates reproducible packaging and transparent replay
handling. It does not demonstrate a real dataset, GPU model, Flower network,
secure aggregation, or production deployment.

## Preparation

Use a clean checkout with Python 3.11.2 and `make`:

```sh
python3 --version
make setup
make test
make check
```

If the source package or tests are absent, stop and say that the checkout is
packaging-only. Do not replace the missing implementation with made-up output.

## Walkthrough

1. **Show the boundary.** Open this document and the top of
   `dashboard/index.html`. Point to the `SYNTHETIC REPLAY — NOT A BENCHMARK`
   banner and the pending integrations in the README.
2. **Run the smoke path.** In a terminal run:

   ```sh
   make smoke RUN_DIR=runs/demo
   ```

   Explain that the command is expected to produce `provenance.json` and
   `metrics.jsonl`, not a research conclusion. Show the provenance fields
   before opening the metrics file.
3. **Serve the viewer.** In another terminal run:

   ```sh
   make run PORT=8000
   ```

   Visit <http://127.0.0.1:8000>, choose `runs/demo/metrics.jsonl`, and show
   the source name, record count, raw JSON, and transparent table. The viewer
   does not compute a score or contact a service.
4. **State the limitation.** Say: “These are synthetic CPU replay records.
   They do not establish real-data accuracy, privacy, security, or deployment
   performance. Real datasets, GPU runs, Flower, Next.js, PostgreSQL, and
   Redis are pending.”
5. **Close with the evidence path.** Point to
   [`docs/RESULTS.md`](RESULTS.md),
   [`docs/DATASET_VERIFICATION.md`](DATASET_VERIFICATION.md), and
   [`docs/THREAT_MODEL.md`](THREAT_MODEL.md). Explain that a real result needs
   those records plus raw, reproducible artifacts.

## Optional security replay

To exercise the attack/defense configuration boundary without real media, run:

```sh
make demo-security
```

This writes clean, targeted-label-flip, and SecAgg-off coordinate-median
stages below `runs/demo-security/` plus a manifest. It is a **synthetic replay**:
the script records actual metrics but does not assert attack success, target
recall collapse, or defense recovery. The defense stage is SecAgg-off because
plaintext robust aggregators cannot inspect individual updates hidden by secure
aggregation.

## Failure states to show honestly

- If `make setup` rejects the interpreter, install/use exactly Python 3.11.2;
  do not loosen the pin for the recording.
- If `make smoke` reports missing `src/fedforge/`, the implementation is not
  present in the checkout. Stop rather than fabricating a replay.
- If required output files are missing, the smoke command did not meet its
  artifact contract. Show the error and investigate the implementation.
- If the dashboard has no file selected, it displays an empty state. That is
  not a zero metric and should not be described as one.
- If a replay file is malformed, keep the error visible and do not edit the
  data to make the demo pass.

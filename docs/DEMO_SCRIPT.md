# FedForge final-project runbook

This runbook starts the implemented media API and federated-learning
dashboard. It is suitable for a project presentation, while the CIFAKE
dataset boundary and the `INCONCLUSIVE` safety state must remain visible.

## Preparation

From the repository root:

```sh
```powershell
$env:PYTHONPATH="S:\CLG\3SEM1\DFL\CP\src"
python -m pytest -q
python run.py --port 8000
```
```

Open <http://127.0.0.1:8000>.

## Walkthrough

   1. Upload an authentic image and show `AUTHENTIC` or `INCONCLUSIVE`.
   2. Upload a benchmark synthetic image and show `LIKELY_SYNTHETIC`.
   3. Open the federated tab and run several rounds.
   4. Run the security experiment and compare clean, attack, and defense stages.
   5. Explain that clients send model updates, not raw uploaded images.

   Close with the evidence path. Point to
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

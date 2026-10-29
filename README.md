# FedForge: Federated Deepfake Detection under Byzantine Adversaries

FedForge is a PyTorch and FastAPI research codebase and course project exploring **horizontal federated learning for synthetic (AI-generated) image detection** on the [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) benchmark dataset.

The project evaluates federated convergence under non-IID data distributions, empirical vulnerability to Byzantine poisoning attacks (semantic label-flipping and gradient update-scaling), and mitigation effectiveness across robust aggregation algorithms (Trimmed Mean, Coordinate Median, Multi-Krum, and Gradient Norm Clipping).

All reported numbers are strictly empirical: trained with PyTorch, evaluated on a held-out test split of 300 CIFAKE images, and logged to `runs/*/metrics.json`.

---

## Key Features

- **Empirical Image Detection Pipeline**: Pretrained ResNet-18 feature extractor paired with a trainable linear binary classification head (513 parameters: $512 \times 1$ weights + 1 bias).
- **Federated Learning Algorithms**: Standard `FedAvg` and proximal-regularized `FedProx` ($\mu = 0.01$).
- **Client Heterogeneity Modeling**: Uniform IID stratified partitioning and Non-IID Dirichlet distribution partitioning ($\alpha \in \{0.5, 0.1\}$).
- **Byzantine Adversarial Threat Engine**:
  - Semantic Label-Flipping ($20\%$ and $40\%$ Byzantine client corruption).
  - Update-Scaling / Norm Explosion ($10\times$ gradient amplification).
- **Byzantine-Robust Aggregation Defenses**:
  - Coordinate-wise Trimmed Mean ($\beta = 0.20$).
  - Coordinate Median.
  - Multi-Krum ($f = 1$).
  - Client Gradient Norm Clipping ($M = 1.0$).
- **Communication & Privacy Boundary**:
  - Exactly 513 parameters ($2,052$ bytes) exchanged per client update ($200.4$ KB total for 10 rounds).
  - Strict zero-raw-pixel privacy boundary (raw media and crops never leave client storage).
  - Plaintext parameter transmission (cryptographic SecAgg and DP-SGD are explicitly not implemented).
- **Interactive Dashboard & REST API**:
  - FastAPI service providing `/detect` (with calibrated validation thresholds and 503 fallback if uncheckpointed), `/runs`, `/runs/{id}`, `/runs/{id}/rounds`, and `/ws/federation`.
  - Vanilla JS dashboard (`dashboard/index.html`) featuring active run selection, empirical convergence curve rendering, and a real 2x2 confusion matrix.

---

## Empirical Benchmark Summary

Full experiment logs, round telemetry, and methodology are documented in [`docs/RESULTS.md`](docs/RESULTS.md).

| Experiment Setup | Aggregator / Defense | Byzantine Ratio | Test Accuracy | Test ROC-AUC | Test F1 | Communication Wire |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Centralized Baseline** | SGD (Pointers Pooled) | 0% | **73.00%** | **0.8188** | 0.7666 | 0 B (Local) |
| **Federated (IID)** | FedAvg (Plaintext) | 0% | **63.33%** | **0.7381** | 0.6726 | 200.4 KB |
| **Federated (Non-IID α=0.5)** | FedAvg (Plaintext) | 0% | **59.33%** | **0.6351** | 0.6755 | 200.4 KB |
| **Federated (Non-IID α=0.1)** | FedAvg (Plaintext) | 0% | **65.33%** | **0.7188** | 0.7011 | 200.4 KB |
| **Attack: Label Flip (1 Malicious)** | FedAvg (No Defense) | 20% | 65.33% | 0.7777 | 0.6923 | 200.4 KB |
| **Defense: Label Flip (1 Malicious)**| Trimmed Mean ($\beta=0.2$) | 20% | **68.33%** | **0.7893** | 0.7214 | 200.4 KB |
| **Defense: Label Flip (1 Malicious)**| Coordinate Median | 20% | **68.33%** | 0.7164 | 0.7022 | 200.4 KB |
| **Defense: Label Flip (1 Malicious)**| Multi-Krum ($f=1$) | 20% | 62.67% | 0.7362 | 0.6872 | 200.4 KB |
| **Attack: Label Flip (2 Malicious)** | FedAvg (No Defense) | 40% | 52.33% | 0.5860 | 0.6635 | 200.4 KB |
| **Defense: Label Flip (2 Malicious)**| Coordinate Median | 40% | **60.33%** | **0.6481** | 0.6448 | 200.4 KB |
| **Attack: Scaling (10x Norm)** | FedAvg (No Defense) | 20% | 60.00% | 0.6658 | 0.6939 | 200.4 KB |
| **Defense: Scaling (10x Norm)** | Norm Clipping ($M=1.0$) | 20% | **76.00%** | **0.8645** | 0.7677 | 200.4 KB |

---

## Installation & Setup

### Requirements
- Python 3.10+ (tested on Python 3.11 and 3.13)
- PyTorch 2.0+ and TorchVision 0.15+

### Editable Install
```bash
# Clone the repository
git clone https://github.com/srvshvoid/dflcp.git
cd dflcp

# Install package and dependencies in editable mode
pip install -e .
pip install pytest httpx
```

### Local process federation and privacy controls

The default PyTorch experiment runner is intentionally an in-process
simulation. For a local multi-process protocol demonstration, start a
coordinator and submit signed, pairwise-masked updates:

```bash
python scripts/run_coordinator.py --clients client-a,client-b --secret local-secret
python scripts/send_masked_update.py --client-id client-a --peers client-a,client-b --secret local-secret --parameters "[0.1, 0.2]"
python scripts/send_masked_update.py --client-id client-b --peers client-a,client-b --secret local-secret --parameters "[0.3, 0.4]"
```

This prototype requires every expected client before masks can cancel. It is
not a replacement for a reviewed production SecAgg+ implementation and
should be deployed behind TLS. HMAC authenticates envelopes and protects
integrity; it does not encrypt them.

Client-level clipping and Gaussian noise can be enabled for experiments:

```bash
python scripts/run_fl.py --dp-clip-norm 1.0 --dp-noise-multiplier 0.5
```

The run records these parameters but does not calculate an epsilon/delta
privacy budget. Set `FEDFORGE_API_TOKEN` before exposing the API; mutating
routes and sensitive federation metadata then require a Bearer token.

---

## Step-by-Step Reproduction Guide

### 1. Download & Prepare the CIFAKE Dataset
The pipeline automatically partitions data into balanced train (1,400 images), validation (300 images), and test (300 images) sets.
```bash
python -c "from fedforge.ml.data import download_and_prepare_cifake; download_and_prepare_cifake()"
```

### 2. Train the Centralized Baseline
Trains the ResNet-18 linear classification head on pooled training data:
```bash
python -m fedforge.ml.train --epochs 10 --batch-size 32 --lr 0.01 --output runs/centralized
```

### 3. Run Federated Learning Experiments
Train 5 clients across 10 communication rounds under various client data distributions:
```bash
# Standard IID Partitioning
python scripts/run_fl.py --output runs/fl_iid --clients 5 --rounds 10 --partition-strategy iid

# Moderate Non-IID Dirichlet (alpha = 0.5)
python scripts/run_fl.py --output runs/fl_dirichlet_05 --clients 5 --rounds 10 --partition-strategy dirichlet --dirichlet-alpha 0.5

# Extreme Non-IID Dirichlet (alpha = 0.1)
python scripts/run_fl.py --output runs/fl_dirichlet_01 --clients 5 --rounds 10 --partition-strategy dirichlet --dirichlet-alpha 0.1
```

### 4. Run Byzantine Attack & Defense Experiments
```bash
# 1 Malicious Client (Label-flipping) vs FedAvg
python scripts/run_fl.py --output runs/attack_1mal_none --clients 5 --rounds 10 --malicious-clients client_0 --attack label_flip --defense none

# 1 Malicious Client vs Trimmed Mean
python scripts/run_fl.py --output runs/defense_1mal_trimmed_mean --clients 5 --rounds 10 --malicious-clients client_0 --attack label_flip --defense trimmed_mean

# 1 Malicious Client vs Coordinate Median
python scripts/run_fl.py --output runs/defense_1mal_median --clients 5 --rounds 10 --malicious-clients client_0 --attack label_flip --defense coordinate_median

# 1 Malicious Client vs Krum
python scripts/run_fl.py --output runs/defense_1mal_krum --clients 5 --rounds 10 --malicious-clients client_0 --attack label_flip --defense krum

# 2 Malicious Clients (40% Byzantine) vs Coordinate Median
python scripts/run_fl.py --output runs/defense_2mal_median --clients 5 --rounds 10 --malicious-clients client_0,client_1 --attack label_flip --defense coordinate_median

# Update Scaling Attack (10x Norm) vs Norm Clipping Defense
python scripts/run_fl.py --output runs/defense_scaling_norm_clip --clients 5 --rounds 10 --malicious-clients client_0 --attack update_scaling --scaling-factor 10.0 --defense norm_clip --clip-norm 1.0
```

### 5. Compile Automated Report
Regenerate [`docs/RESULTS.md`](docs/RESULTS.md) from the completed runs:
```bash
python scripts/make_report.py --runs runs --output docs/RESULTS.md
```

### 6. Launch the Interactive Dashboard & API
```bash
python run.py
```
Open <http://localhost:8000> in your browser to view the live dashboard, inspect empirical convergence curves, review held-out confusion matrices, or test live image inference via `/detect`.

---

## Running the Verification Test Suite

The test suite consists of 47 unit and integration tests covering:
- Dataset downloading, preprocessing, and Dirichlet partitioning.
- PyTorch model forward passes and probability calibration.
- Federated aggregation algorithms (`FedAvg`, `FedProx`, `trimmed_mean`, `coordinate_median`, `krum`, `norm_clip`).
- Adversarial attacks (`label_flip`, `update_scaling`).
- REST and WebSocket API endpoints, static dashboard serving, and 503 fallback error handling.

Execute the suite:
```bash
pytest -v
```

---

## Project Structure

```
FedForge/
├── configs/                  # Configuration files
├── dashboard/
│   └── index.html            # Dashboard UI (vanilla HTML/CSS/JS, no CDN dependencies)
├── data/
│   ├── manifest.json         # Dataset split manifest & SHA-256 hashes
│   ├── train/                # 1,400 balanced CIFAKE images
│   ├── val/                  # 300 balanced CIFAKE images
│   └── test/                 # 300 held-out CIFAKE test images
├── docs/
│   ├── RESULTS.md            # Empirical benchmark tables and analysis
│   └── ARCHITECTURE.md       # Architecture documentation
├── runs/                     # Experiment artifacts (checkpoints, metrics.json, rounds.jsonl)
├── scripts/
│   ├── run_fl.py             # CLI runner for federated experiments
│   └── make_report.py        # Automated report compiler
├── src/
│   └── fedforge/
│       ├── api/              # FastAPI application & RunRepository
│       ├── ml/               # PyTorch training, models, attacks, and robust defenses
│       └── vision/           # Media detection & forensic visual overlay pipeline
├── tests/                    # 47 unit and integration tests
├── pyproject.toml            # Project package configuration & dependencies
└── README.md
```

---

## Limitations & Scope Boundary

1. **Synthetic Image Domain**: Experiments are conducted on the CIFAKE dataset ($32 \times 32$ images upsampled to $224 \times 224$ for ResNet-18). Visual artifacts reflect generative AI models from CIFAR-10 rather than high-resolution deepfake video frames.
2. **Head-Only Fine-Tuning**: To enable deterministic CPU execution and lightweight communication (2,052 bytes per round), the ResNet-18 convolutional backbone is kept frozen; only the linear classification head is trained.
3. **Simulated In-Process Federation**: Client communication is simulated sequentially in-process on CPU; asynchronous network lag, packet drops, and client churn are not modeled.
4. **Privacy Protocol**: Model updates are transmitted as plaintext parameter arrays. Cryptographic Secure Aggregation (SecAgg) and Differential Privacy (DP-SGD) are not implemented.

### Cross-Dataset Validation

The repository includes reproducible tooling for a licensed external
real/fake image holdout. Download a controlled subset and evaluate the same
live upload path used by the API:

```powershell
$env:PYTHONPATH="S:\CLG\3SEM1\DFL\CP\src"
python scripts/download_external_dataset.py --per-class 200
python scripts/evaluate_external.py --output runs/external-validation.json
```

The report includes coverage, abstention (`INCONCLUSIVE`) count, and accuracy
on decided samples. The source metadata is written to
`data/external/real_fake_images/SOURCE.json`.

To train a clean mixed-domain checkpoint after downloading the holdout:

```powershell
python scripts/train_cross_dataset.py --epochs 5 --output runs/cross_dataset
```

This keeps the federated experiment checkpoints intact and creates a separate
deployment candidate whose test split contains both CIFAKE and external
images.

The same external data can be included in federated training:

```powershell
python scripts/run_fl.py --data data/cifake `
  --extra-data data/external/real_fake_images `
  --output runs/fl_mixed_iid --clients 5 --rounds 10
```

---

## License

This project is released under the [MIT License](LICENSE). CIFAKE dataset images are released under CC0: Public Domain.

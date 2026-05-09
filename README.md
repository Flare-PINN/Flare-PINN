# Flare-PINN: Weak-Form Physics-Informed Learning for Operational Solar Flare Forecasting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A hybrid deep learning framework combining physics-informed neural networks (PINNs) with magnetohydrodynamic (MHD) constraints for operational solar flare forecasting.

## Results

### Test Set Performance (Distance-to-Corner Thresholding)

All test results use **frozen Distance-to-Corner (D2C) thresholds** computed on validation set to prevent data leakage.

**Key Findings:**
- Physics improves TSS by +0.052 at 24h over the no-physics ablation (paper-lock comparison; bootstrap p=0.010)
- Weak-form outperforms strong-form across all horizons (+0.050 at 6h, +0.089 at 12h, +0.038 at 24h; multi-seed Welch's t-test)
- D2C thresholding optimizes ROC space (minimizes distance to perfect classifier)

**24-hour Forecast Horizon (Multi-Seed Test Performance):**

| Model | TSS | POD | FAR | Brier | Threshold |
|-------|-----|-----|-----|-------|-----------|
| **Flare-PINN (Weak-Form, 3-seed mean ± SD)** | **0.790 ± 0.007** | 0.945 | 0.906 | 0.041 | per-seed (D2C) |
| Strong-Form PINN (3-seed mean ± SD) | 0.752 ± 0.011 | 0.890 | 0.897 | 0.039 | per-seed (D2C) |
| DeFN (5-seed mean ± SD) | 0.730 ± 0.039 | 0.851 | 0.890 | 0.019 | per-seed (D2C) |
| Benchmark (No Physics) | 0.746 | 0.879 | 0.904 | 0.040 | 0.246 (D2C) |

*POD/FAR/Brier shown for paper-lock checkpoint; TSS shows multi-seed mean ± SD across 3 training seeds (Flare-PINN, Strong-Form) or 5 seeds (DeFN).*

**Multi-Horizon Performance (Flare-PINN, Paper-Lock + 3-seed robustness):**

| Horizon | TSS (paper-lock) | TSS (3-seed mean ± SD) | POD | FAR | Threshold (D2C) |
|---------|------------------|------------------------|-------|-------|-----------------|
| 6h      | 0.826 | 0.817 ± 0.011 | 0.914 | 0.940 | 0.260 |
| 12h     | 0.833 | 0.811 ± 0.021 | 0.950 | 0.921 | 0.236 |
| 24h     | 0.798 | 0.790 ± 0.007 | 0.945 | 0.906 | 0.237 |

### Dataset Statistics (80/5/15 Chronological Split)

| Split | Period | Windows | M+ Flares | Positive Rate |
|-------|--------|---------|-----------|---------------|
| **Train** | Jan 2011 - Aug 2015 | 28,405 | 1,081 | 3.81% |
| **Validation** | Aug 2015 - Dec 2015 | 1,905 | 98 | 5.14% |
| **Test** | Dec 2015 - Dec 2017 | 5,716 | 91 | 1.59% |

### DeFN baseline (comparison)

Faithful PyTorch reimplementation of **Deep Flare Net** (Nishizuka et al.; original TF: [defn18](https://github.com/komeisugiura/defn18)), trained on a reconstructed **79-feature** table with the **same frozen validation D2C thresholding** as Flare-PINN.

- **Code:** `src/baselines/defn/defn/` (model, train loop, predictions writer), `tools/defn/` (`build_defn_features.py`, `audit_defn_state.py`)
- **Pre-trained predictions (this repo):** `outputs/baselines/defn/seed{10,24,42,100,123}_test.npz` (per-seed test probabilities, labels, frozen D2C thresholds)
- **DeFN feature table (deposited):** `data/defn/defn_features.parquet` (38,102 windows × 79 features)
- **Train (5 seeds, ~15 min on CPU):**  
  ```bash
  python -m src.baselines.defn.defn.train_with_predictions \
      --seeds 24,10,100,42,123 --max-iters 16000 \
      --output-dir outputs/baselines/defn
  ```
- **Per-seed TSS at 6h / 12h / 24h:**

  | Seed | 6h | 12h | 24h |
  |------|-----|-----|-----|
  | 10   | 0.693 | 0.761 | 0.679 |
  | 24   | 0.761 | 0.757 | 0.774 |
  | 42   | 0.680 | 0.769 | 0.753 |
  | 100  | 0.570 | 0.733 | 0.741 |
  | 123  | 0.760 | 0.789 | 0.702 |
  | **Mean ± SD** | **0.693 ± 0.078** | **0.762 ± 0.020** | **0.730 ± 0.039** |

## Overview

This repository implements a novel hybrid architecture that integrates:
1. **Convolutional-GRU encoder** for spatiotemporal magnetogram processing
2. **Coordinate-based implicit neural field** via Feature-wise Linear Modulation (FiLM)
3. **Weak-form MHD constraints** (2.5D resistive induction equation) for physics-informed regularization

The model forecasts probability of >=M-class solar flares at 6h, 12h, and 24h horizons using SDO/HMI SHARP vector magnetogram data (2011-2017).

**Key Features:**
- Weak-form physics loss enforcing 2.5D resistive induction equation
- Boundary-vanishing Fourier sine test functions (rigorous variational formulation)
- Physics-informed regularization with implicit neural field representation
- Class-Balanced Focal Loss for extreme class imbalance (3.81% positive rate in training)
- Fully deterministic training and evaluation pipeline
- Chronological train/validation/test split (80/5/15) preventing temporal leakage

## Installation

### Requirements

- Python 3.11+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU) or Apple MPS (for Apple Silicon)

### Quick Start

```bash
# Clone the anonymous mirror (code repository for double-blind review)
git clone https://github.com/Flare-PINN/Flare-PINN.git
cd Flare-PINN

# Create conda environment
conda env create -f environment.yml
conda activate flare-pinn

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

### JSOC Account Setup (for data download)

Register for a free JSOC account at http://jsoc.stanford.edu/ajax/register_email.html, then:

```bash
export JSOC_EMAIL="your_email@institution.edu"
```

## Quick Inference (Pre-trained Models)

Paper-lock checkpoints are provided in `outputs/checkpoints/`. The selection metadata for the headline checkpoint is in `outputs/finalists/flare_pinn_paper_lock.json`.

1. **CNN-GRU Benchmark** (No Physics): `benchmark_classifier/checkpoint_step_0040000.pt`
   - Test TSS (24h): 0.746
   - Training steps: 40,000

2. **Flare-PINN** (Weak-Form, Headline): `weak_form/final/checkpoint_step_0044000.pt`
   - Test TSS (24h): 0.798 (6h: 0.826, 12h: 0.833)
   - Selected step: 44,000 (max mean validation TSS)
   - Per-seed sensitivity runs: `weak_form/seed1/`, `weak_form/seed42/`

3. **Strong-Form PINN** (Baseline): `strong_form/final/checkpoint_step_0044000.pt`
   - Test TSS (24h): 0.765
   - Per-seed sensitivity runs: `strong_form/seed1/`, `strong_form/seed42/`

### Running Inference

Evaluate the paper-lock Flare-PINN checkpoint on the held-out test set:

```bash
cd /path/to/flare-pinn   # repo root
conda activate flare-pinn
export PYTHONPATH="$PWD:$PYTHONPATH"

# Run inference on test set (auto-loads frozen D2C thresholds from validation NPZ)
python tools/validation/validate_checkpoint.py \
  --config src/configs/flare_pinn_final.yaml \
  --checkpoint outputs/checkpoints/weak_form/final/checkpoint_step_0044000.pt \
  --data data/windows_test_15.parquet

# Generate confusion matrices
python tools/validation/compute_confusion_matrix.py \
  --results outputs/checkpoints/weak_form/final/validation_results/checkpoint_step_0044000_test.npz \
  --output-dir final_results/confusion_matrices/
```

**Expected test TSS:** 0.798 (24h), 0.833 (12h), 0.826 (6h)

## Data Pipeline

### Overview

The data pipeline downloads and processes SDO/HMI SHARP magnetograms (2011-2017):

```
1. Fetch Flares (HEK) -> 2. HARP-NOAA Mapping -> 3. Download SHARP -> 4. Consolidate Frames -> 5. Create Windows
```

**Important:** Run consolidation (Step 4) **before** creating windows (Step 5).

### Step-by-Step Instructions

#### 1. Fetch GOES Flare Events

Downloads M+ and X-class flares from NOAA's Heliophysics Event Knowledgebase (HEK).

```bash
python data_scripts/fetch_flares.py \
  --start 2011-01-01 \
  --end 2018-01-01 \
  --min-class M1.0
```

**Output:** 
- `data/flares_hek.parquet` (M+ and X-class flares with peak times and NOAA AR numbers)
- Saves monthly chunks to `data/hek_chunks/` during download, then combines them

**What it does:**
- Queries SunPy HEK for flare events in specified date range
- Filters by GOES class (>=M1.0 by default)
- Extracts: start time, peak time, end time, GOES class, NOAA AR number
- Resume-safe: skips already-downloaded months

---

#### 2. Bootstrap HARP-NOAA Mapping

Creates temporal mapping between HARP numbers (data identifiers) and NOAA AR numbers (flare labels).

```bash
python data_scripts/bootstrap_harp_mapping.py
```

**Output:** 
- `data/harp_noaa_mapping.parquet` (HARP <-> NOAA AR mapping with temporal overlap)

**What it does:**
- Downloads JSOC's canonical `all_harps_with_noaa_ars.txt` mapping
- Unions with any existing local mapping for maximum coverage
- Format: `[harpnum, noaa_ar]` pairs

**Note:** This mapping is essential because SHARP data uses HARP numbers, but flare events are labeled with NOAA AR numbers.

---

#### 3. Download SHARP Magnetograms

Downloads HMI SHARP CEA 720s vector magnetogram data from JSOC.

```bash
export JSOC_EMAIL="your_email@institution.edu"  # Required for JSOC access

python data_scripts/download_sharp_cea.py \
  --start 2011-01-01 \
  --end 2018-01-01 \
  --out ~/flare_data/sharp_cea \
  --segments Br Bt Bp bitmap \
  --step 1h \
  --chunk day \
  --workers 4 \
  --verbose
```

**Output:** 
- `~/flare_data/sharp_cea/YYYY/MM/DD/*.npz` (Br, Bt, Bp components at 1h cadence)
- Total size: ~40-50GB (after NPZ conversion)

**What it does:**
- Creates JSOC export requests (day-by-day or month-by-month)
- Downloads FITS files from JSOC staging URLs
- Converts FITS -> NPZ (float32 data + uint8 mask)
- **Resume-safe:** Skips existing NPZ files, only downloads missing data
- Parallel downloads with configurable worker pool

**Time:** ~4-8 hours for 2011-2017 (depends on JSOC queue and bandwidth)

**Requirements:** Free JSOC account (register at http://jsoc.stanford.edu/ajax/register_email.html)

---

#### 4. Consolidate Frames (Run BEFORE Step 5)

Bundles individual NPZ files into per-HARP consolidated files for 50x faster I/O.

```bash
python data_scripts/consolidate_frames.py \
  --frames-meta ~/flare_data/sharp_cea/frames_meta.parquet \
  --npz-root ~/flare_data/sharp_cea \
  --output-dir ~/flare_data/consolidated \
  --target-size 128 \
  --workers 8
```

**Output:** 
- `~/flare_data/consolidated/H<harpnum>.npz` (per-HARP bundles, e.g., `H377.npz`, `H1234.npz`)
- `~/flare_data/consolidated/frames_meta.parquet` (metadata index for all frames)

**What it does:**
- Groups individual NPZ files by HARP number
- Resizes magnetograms to 128x128 (or specified size)
- Stores as float16 (50% size reduction)
- Pre-computes normalization statistics (median, IQR, NaN fractions)
- Creates searchable metadata index with timestamps, CMD, quality metrics

**Why:** Training dataset loads entire HARP timeseries at once -> consolidation reduces 50,000+ file opens to ~1,000 file opens.

**Time:** ~30-60 minutes for 2011-2017 data

---

#### 5. Create Rolling Windows

Generates classification windows with flare labels at multiple horizons.

```bash
python data_scripts/create_windows.py \
  --frames ~/flare_data/consolidated/frames_meta.parquet \
  --events data/flares_hek.parquet \
  --input-hours 48 \
  --stride-hours 6 \
  --horizons 6 12 24 \
  --cmd-threshold 70 \
  --min-frames 6 \
  --partial-ok-frac 0.7 \
  --splits 80 5 15 \
  --output-train data/windows_train_80.parquet \
  --output-val data/windows_val_5.parquet \
  --output-test data/windows_test_15.parquet
```

**Output:** 
- `data/windows_train_80.parquet` (~28,405 windows)
- `data/windows_val_5.parquet` (~1,905 windows)
- `data/windows_test_15.parquet` (~5,716 windows)

**What it does:**
- Creates 48-hour rolling windows with 6-hour stride per HARP
- Labels windows with flare occurrence (>=M1.0) at 6h/12h/24h horizons
- Filters by limb proximity (CMD < 70 deg) to avoid geometric distortion
- Chronologically splits into train (80%) / validation (5%) / test (15%)
- Matches flares to HARPs using HARP-NOAA mapping from Step 2
- Handles partial windows near limb crossings (>=70% coverage required)

**Window Format:**
Each row contains:
- `harpnum`: HARP identifier
- `t0`: Window start time
- `frame_paths`: List of NPZ paths in window
- `y_geq_M_6h`, `y_geq_M_12h`, `y_geq_M_24h`: Binary labels (1 = flare, 0 = quiet)
- `is_masked_*h`: Whether horizon is censored (e.g., near end of HARP lifetime)
- `cmd_deg`, `obs_coverage`: Quality metrics

**Time:** ~5-10 minutes

**Rationale:** 80/5/15 chronological split prevents temporal leakage. Validation set is kept small to maximize training data while providing sufficient positives for threshold selection.

---

## Training

### Two-Stage Training Protocol

#### Stage 1: Data-Driven Baseline (0 -> 40k steps)

Train CNN-GRU encoder without physics constraints:

```bash
./run_benchmark.sh
```

**Expected test TSS:** 0.746 (24h horizon)

**Hyperparameters:**
- Learning rate: 2e-4 -> 1e-6 (cosine decay, 1000-step warmup)
- Batch size: 16 (2 per GPU x 8 gradient accumulation)
- Encoder dropout: 0.15, Classifier dropout: 0.35
- Class-Balanced Focal Loss (beta=0.9997, gamma=1.5)
- Seed: 24

**Checkpoint:** `outputs/checkpoints/benchmark_classifier/checkpoint_step_0040000.pt`

---

#### Stage 2: Physics Fine-Tuning (40k -> 46k steps)

Resume from 40k baseline and enable weak-form MHD constraints:

```bash
python src/train.py \
  --config src/configs/flare_pinn_final.yaml \
  --resume outputs/checkpoints/benchmark_classifier/checkpoint_step_0040000.pt
```

**Physics loss configuration (paper-lock):**
- `physics_grad_scale`: 0.08
- `lambda_phys_schedule`: piecewise linear, 0.0 (until normalized t=0.8) -> 0.20 (at t=1.0)
- Learning rate: cosine decay, fresh schedule
- Seed: 1234
- Selection: max mean validation TSS over candidate steps in [44k, 46k]; selected step = 44,000

**Selected checkpoint:** `outputs/checkpoints/weak_form/final/checkpoint_step_0044000.pt`
**Selection metadata:** `outputs/finalists/flare_pinn_paper_lock.json`

## Evaluation

### Reproduce Paper Results

```bash
# Navigate to project root
cd /path/to/flare-pinn
conda activate flare-pinn
export PYTHONPATH="$PWD:$PYTHONPATH"

# 1. Test Benchmark (No Physics) - 40k checkpoint
python tools/validation/validate_checkpoint.py \
  --config src/configs/benchmark_classifier.yaml \
  --checkpoint outputs/checkpoints/benchmark_classifier/checkpoint_step_0040000.pt \
  --data data/windows_test_15.parquet

# 2. Test Flare-PINN (Weak-Form) - paper-lock 44k checkpoint
python tools/validation/validate_checkpoint.py \
  --config src/configs/flare_pinn_final.yaml \
  --checkpoint outputs/checkpoints/weak_form/final/checkpoint_step_0044000.pt \
  --data data/windows_test_15.parquet

# 3. Test Strong-Form PINN - 44k checkpoint
python tools/validation/validate_checkpoint.py \
  --config src/baselines/strong_form/config_matched.yaml \
  --checkpoint outputs/checkpoints/strong_form/final/checkpoint_step_0044000.pt \
  --data data/windows_test_15.parquet

# 4. Test DeFN (5-seed ensemble) - reads test predictions from outputs/baselines/defn/
python -m src.baselines.defn.defn.train_with_predictions \
  --seeds 24,10,100,42,123 --max-iters 16000 \
  --output-dir outputs/baselines/defn

# 5. Statistical comparisons (Table 4: bootstrap + Welch's t-test)
python tools/validation/compute_block_bootstrap.py
```

**Note:** `validate_checkpoint.py` automatically:
1. Runs silent validation to compute D2C thresholds (not shown in output)
2. Freezes those thresholds for test set evaluation
3. Reports only test set metrics (no data leakage!)

---

### Generate Confusion Matrices

```bash
# For each model, generate confusion matrix from test results
python tools/validation/compute_confusion_matrix.py \
  --results outputs/checkpoints/weak_form/final/validation_results/checkpoint_step_0044000_test.npz \
  --output-dir final_results/confusion_matrices/
```

---

## Repository Structure

```
flare-pinn/
├── data/                                # Data files (deposited with code release)
│   ├── flares_hek.parquet               # GOES flare catalog (from HEK)
│   ├── harp_noaa_mapping.parquet        # HARP <-> NOAA AR mapping
│   ├── scalar_features.parquet          # Per-frame scalar features
│   ├── windows_train_val_8005.parquet   # Train+val windows
│   ├── windows_test_15.parquet          # Test windows (chronological holdout)
│   └── defn/
│       └── defn_features.parquet        # 79-feature DeFN input table
├── data_scripts/                        # Data pipeline
│   ├── fetch_flares.py                  # 1. Download GOES catalog from HEK
│   ├── bootstrap_harp_mapping.py        # 2. Create HARP-NOAA mapping from JSOC
│   ├── download_sharp_cea.py            # 3. Download SHARP magnetograms from JSOC
│   ├── consolidate_frames.py            # 4. Bundle frames per HARP
│   ├── create_windows.py                # 5. Generate rolling windows
│   └── spatial_temporal_matching.py     # Spatial fallback HARP-NOAA matching
├── src/
│   ├── configs/                         # Training configurations
│   │   ├── benchmark_classifier.yaml    # Stage 1 (no physics, 0->40k)
│   │   ├── flare_pinn_final.yaml        # Stage 2 paper-lock config
│   │   └── data_{train,validation,test}.yaml
│   ├── data/                            # Dataset implementations
│   ├── models/pinn/                     # Flare-PINN architecture (hybrid_model, encoder, physics)
│   ├── models/eval/                     # Metrics, bootstrap, calibration utilities
│   ├── baselines/
│   │   ├── classical/                   # Logistic Regression, XGBoost, SVM training
│   │   ├── strong_form/                 # Strong-form physics baseline (config + physics)
│   │   └── defn/defn/                   # DeFN reimplementation (config, model, train, predictions)
│   ├── utils/                           # Training utilities
│   └── train.py                         # Main training script
├── tools/                               # Evaluation and analysis
│   ├── defn/                            # DeFN feature build (`build_defn_features.py`), state audit
│   ├── validation/
│   │   ├── validate_checkpoint.py       # Test metrics with frozen D2C thresholds
│   │   ├── compute_block_bootstrap.py   # Table 4 (bootstrap + Welch's t-test)
│   │   ├── compute_confusion_matrix.py  # Confusion matrices
│   │   └── calibrate_platt.py           # Post-hoc Platt calibration
│   ├── visualization/                   # Figure regeneration scripts
│   ├── analysis/                        # Latent diagnostics, post-hoc errors, Fig. 6
│   └── training/                        # Training driver scripts
├── outputs/
│   ├── checkpoints/
│   │   ├── benchmark_classifier/        # No-physics CNN-GRU (40k)
│   │   ├── weak_form/{final,seed1,seed42}/   # Flare-PINN paper-lock + 2 seeds
│   │   └── strong_form/{final,seed1,seed42}/ # Strong-Form paper-lock + 2 seeds
│   ├── baselines/defn/                  # Per-seed DeFN test predictions (5 seeds)
│   ├── classical_baselines/             # LR/XGBoost/SVM test predictions
│   └── finalists/flare_pinn_paper_lock.json   # Paper-lock checkpoint metadata
├── final_results/                       # Published figures and metrics
│   ├── figures/                         # Main-text and supplementary figures
│   ├── tables/                          # Table CSVs
│   ├── metrics/                         # TSS, calibration, ROC/PR, bootstrap CSVs
│   ├── methodology/                     # Diagnostic outputs (latent, posthoc errors)
│   └── paper/                           # Final paper bundle (figures + tables + metrics)
├── environment.yml
└── README.md
```

## Computational Requirements

- **Training:** ~8 hours on Apple M2 Max (64GB RAM) or NVIDIA A100 (40GB)
- **Inference:** ~69ms per active region (MPS), ~15 inferences/sec
- **Data storage:** ~50GB (raw SHARP) + ~35GB (consolidated)
- **Memory:** 16GB+ RAM recommended for training

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Acknowledgments

- **HMI/SHARP data** provided by NASA/SDO and the HMI science team
- **GOES flare catalog** from NOAA/SWPC via Heliophysics Event Knowledgebase (HEK)
- **JSOC infrastructure** at Stanford University for data access

## Contact

*Author and affiliation withheld for double-blind review.*

## Troubleshooting

### Common Issues

**Q: "PermissionError" during JSOC download**  
A: Ensure `JSOC_EMAIL` is set and registered at http://jsoc.stanford.edu

**Q: "CUDA out of memory" during training**  
A: Reduce batch size or enable gradient checkpointing in config

**Q: "Test metrics differ slightly from paper"**  
A: Ensure you're using EMA weights (`--use-ema`) and correct data split. Note: Early training logs may show non-deterministic spikes due to sampling - this was fixed with deterministic coordinate sampling and stable sorting.

**Q: "Slow training on Apple Silicon"**  
A: Verify PyTorch MPS backend is enabled: `torch.backends.mps.is_available()`

For additional support, open an issue on GitHub.

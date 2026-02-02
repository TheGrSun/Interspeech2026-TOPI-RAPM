# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**R-APM (Retrieval-Augmented Pragmatic Mapper)** - A system for cross-lingual prosody transfer from English to Spanish for the Interspeech 2026 TOPI Challenge. The task is to predict Spanish HuBERT prosodic features (101-dim) from English HuBERT features (1024-dim).

**Key Finding**: Pure retrieval achieves strong performance (0.8722 cosine similarity), but Retrieval + Fusion achieves best results (0.8742). The retrieval error is uncorrelated with input features (correlation ~0.08).

## Environment Setup

### Dependencies

```bash
# Core dependencies
pip install torch torchvision torchaudio
pip install numpy scikit-learn tqdm pyyaml
```

Required packages:
- `torch` - PyTorch for model training
- `numpy` - Array operations
- `scikit-learn` - PCA, feature selection
- `tqdm` - Progress bars
- `pyyaml` - Configuration loading

## Common Commands

### Training

```bash
# Main R-APM v2 training (uses config.yaml)
cd src
python train_rapm_v2.py

# Final submission training with optimal hyperparameters
python train_final_submission.py
```

### Feature Extraction

```bash
# Extract HuBERT features from audio (for test set)
cd submit
python extract_hubert_features.py --model_path <path> --input_dir <dir> --output_dir <dir> --device cuda
```

### Generate Submission

```bash
# Generate submission using final model
cd submit
python generate_final_submission.py

# Generate baseline retrieval submission
python generate_submission.py
```

## Architecture Overview

### Data Pipeline

```
Raw HuBERT Features (1024-dim)
    ↓
Feature Selection (variance-based)
    ↓
Selected Features (101-dim) ← Competition submission format
```

**Critical**: Use variance-based feature selection. PCA performs poorly (~0.45 cosine) because it optimizes reconstruction rather than preserving cosine similarity structure.

### R-APM v2 Architecture

```
EN_1024 → Enhanced Retrieval → ES_retrieved_1024
              (Top-K=70)            ↓
                              Feature Selection
                                    ↓
                              ES_retrieved_101
                                    ↓
    ┌───────────────────────────────────────────┐
    │    Enhanced Correction Network (Optional) │
    │  • Multi-head Self-Attention (8 heads)    │
    │  • Multi-scale MLP [1, 2, 4]              │
    │  • Gating Mechanism                       │
    │  • Residual Connection                    │
    └───────────────────────────────────────────┘
                    ↓
        Output = ES_retrieved + Delta
```

**Key Insight**: Best results come from Retrieval + Fusion with Top-K=70, temperature=0.04.

### Key Modules

- `src/models/enhanced_retrieval.py` - Hybrid similarity (cosine + Wasserstein), Top-K retrieval
- `src/models/relative_correction.py` - Residual correction network (minimal impact)
- `src/data/feature_selector.py` - Variance-based dimension selection (101 dims)

## Configuration

- **Config File**: `config.yaml` (YAML-based configuration)

**Critical Parameters**:
```yaml
model:
  retrieval:
    top_k: 70             # Optimal value
    temperature: 0.04     # Sharp attention
    similarity_type: "cosine"  # Best performing

training:
  loss_type: "cosine"    # Directly optimize competition metric
  learning_rate: 0.0008
  epochs: 500
```

## Data Locations

- Training data: `dral-features/features/` (EN_*.npy, ES_*.npy, 1024-dim each)
- Test data: `test-features/`
- Checkpoints: `checkpoints/`
  - `ensemble_v1/` - Best ensemble model (0.8863 performance)
  - `best_final_submission.pth` - Final submission checkpoint
- Submissions: `submit/predictions_final/`

## Submission Format

Competition requires:
- Input: EN_*.npy (1024-dim HuBERT features)
- Output: ES_*.npy (101-dim selected features)
- Format: Each prediction as separate (101,) numpy array, dtype float64
- Package: Zip all .npy files together

## Important Findings

1. **Retrieval + Fusion is best**: Achieves 0.8742 vs baseline 0.8732
2. **Correction is ineffective**: Error-EN correlation ~0.15 (essentially random)
3. **Variance selection >> PCA**: PCA destroys cosine similarity structure
4. **Optimal Top-K**: 70 (with temperature 0.04)
5. **Cosine similarity is best**: Hybrid offers no benefit over pure cosine

## References

- DRAL Dataset: https://www.cs.utep.edu/nigel/dral/
- Competition: https://www.codabench.org/competitions/12225/
- System Description: `system_description.md`

## Project Structure

```
E:\interspeech2026\
├── config.yaml                 # V2 configuration
├── CLAUDE.md                   # This file
├── system_description.md       # System description paper
│
├── src/
│   ├── __init__.py
│   ├── train_rapm_v2.py       # Main training script
│   ├── train_final_submission.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── enhanced_retrieval.py
│   │   └── relative_correction.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── feature_selector.py
│   └── losses/
│       └── __init__.py
│
├── submit/
│   ├── generate_final_submission.py
│   ├── generate_submission.py
│   ├── extract_hubert_features.py
│   ├── README.md
│   └── predictions_final/
│
├── checkpoints/               # Model checkpoints
├── official_mdekorte/         # Official baseline code
├── Models/                    # Official baseline models
├── dral-features/             # Training data
└── test-features/             # Test data
```

# Interspeech 2026 TOPI S2ST Challenge: R-APM

**Retrieval-Augmented Pragmatic Mapper for Cross-Lingual Prosody Transfer**

[简体中文](README.zh-CN.md) | [日本語](README.ja.md)

---

## Overview

R-APM is a retrieval-based system for cross-lingual prosody transfer from English to Spanish. It predicts Spanish HuBERT prosodic features (101-dim) from English HuBERT features (1024-dim) using a hybrid retrieval + fusion architecture.

> **📄 System Description**: [R-APM: Retrieval-Augmented Pragmatic Mapper for Cross-Lingual Prosody Transfer.PDF](docs/R-APM_System_Description.pdf) (Coming Soon)

## Key Results

### Internal Split (Official Train/Test Split)

| Model | Score | vs MLP Baseline |
|-------|-------|-----------------|
| **1024-Fusion** (Submission) | **0.8742** | **+0.10%** |
| 1024-Pure | 0.8722 | -0.11% |
| 103-Fusion | 0.8654 | -0.90% |
| 103-Pure | 0.8642 | -1.03% |
| **MLP Baseline** | 0.8732 | - |

### Official Test Set

| Model | Score | vs MLP Baseline |
|-------|-------|-----------------|
| **1024-Fusion** (Submission) | **0.8288** | **-2.86%** |
| **MLP Baseline** | **0.8574** | - |

> **Note**: Internal split uses the official train/test filelists from `official_baseline/data/filelists/`. The MLP baseline outperforms our system on the official test set, indicating challenges in generalization to unseen speakers.

## Architecture

### Pure Retrieval Mode

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PURE RETRIEVAL MODE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Input: EN_1024 (English HuBERT Features, 1024-dim)                        │
│       │                                                                     │
│       ▼                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐          │
│   │         Enhanced Retrieval Module                           │          │
│   │  • Similarity: Cosine                                       │          │
│   │  • Top-K Retrieval: K=70                                    │          │
│   │  • Temperature: 0.04 (Sharp Attention)                      │          │
│   └─────────────────────────────────────────────────────────────┘          │
│       │                                                                     │
│       ▼                                                                     │
│   ES_retrieved_1024 (Retrieved Spanish Features, 1024-dim)                  │
│       │                                                                     │
│       ▼                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐          │
│   │         Feature Selection Module                            │          │
│   │  • Method: Predefined Official Indices (101 dims)           │          │
│   │  • Source: Competition baseline feature selection          │          │
│   └─────────────────────────────────────────────────────────────┘          │
│       │                                                                     │
│       ▼                                                                     │
│   Output: ES_101 (Spanish Prosodic Features, 101-dim)                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Fusion Mode ⭐ **SUBMISSION MODEL**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FUSION MODE (Submission)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Input: EN_1024 (English HuBERT Features, 1024-dim)                        │
│       │                                                                     │
│       ├────────────────────────────────────────┐                            │
│       ▼                                        ▼                            │
│   ┌─────────────────────┐         ┌─────────────────────────┐             │
│   │ Enhanced Retrieval  │         │   Direct Projection      │             │
│   │ • Top-K=70          │         │   EN_1024 → 101         │             │
│   │ • Temp=0.04         │         └─────────────────────────┘             │
│   └─────────────────────┘                    │                             │
│       │                                      │                             │
│       ▼                                      │                             │
│   ES_retrieved_1024                          │                             │
│       │                                      │                             │
│       ▼                                      │                             │
│   Feature Selection (101-dim)                 │                             │
│       │                                      │                             │
│       ▼                                      ▼                             │
│   ┌─────────────────────────────────────────────────────┐                  │
│   │              FUSION NETWORK                          │                  │
│   │  ┌─────────────────────────────────────────────┐    │                  │
│   │  │  • Multi-head Self-Attention (8 heads)      │    │                  │
│   │  │  • Multi-scale MLP: [256, 128, 64]          │    │                  │
│   │  │  • Gating Mechanism (Attention-based)       │    │                  │
│   │  │  • Layer Normalization + Dropout(0.0)       │    │                  │
│   │  └─────────────────────────────────────────────┘    │                  │
│   │                                                     │                  │
│   │  Output = ES_retrieved + Delta(EN_input)           │                  │
│   └─────────────────────────────────────────────────────┘                  │
│       │                                                                     │
│       ▼                                                                     │
│   Output: ES_101_fused (Spanish Prosodic Features, 101-dim)                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
git clone --recurse-submodules https://github.com/TheGrSun/Interspeech2026-TOPI-RAPM.git
cd Interspeech2026-TOPI-RAPM
pip install -r requirements.txt
```

## Usage

```bash
# Training
python src/train.py --config config/default.yaml

# Evaluation
python src/evaluate.py --checkpoint checkpoints/best_model.pth

# Generate submission
cd submit
python generate_submission.py
```

## Dataset

Download the DRAL dataset from: https://www.cs.utep.edu/nigel/dral/

## Citation

```bibtex
@inproceedings{rapm2026,
  title={Retrieval-Augmented Pragmatic Mapper for Cross-Lingual Prosody Transfer},
  author={Xiaoyang Luo and others},
  booktitle={Interspeech 2026},
  year={2026}
}
```

## License

MIT License

## Acknowledgments

- Interspeech 2026 TOPI S2ST Challenge organizers
- DRAL Dataset creators

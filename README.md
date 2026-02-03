# Interspeech 2026 TOPI S2ST Challenge: R-APM

**Retrieval-Augmented Pragmatic Mapper for Cross-Lingual Prosody Transfer**

[简体中文](README.zh-CN.md) | [日本語](README.ja.md)

---

## Authors

**Xiaoyang Luo**, **Siyuan Jiang**, **Shuya Yang**, **Dengfeng Ke**, **Yanlu Xie**, **Jinsong Zhang**

Speech Acquisition and Intelligent Technology Laboratory (SAIT LAB)
Beijing Language and Culture University, Beijing, China

---

## Overview

R-APM is a retrieval-based system for cross-lingual prosody transfer from English to Spanish. It predicts Spanish HuBERT prosodic features (101-dim) from English HuBERT features (1024-dim) using a hybrid retrieval + fusion architecture.

> **📄 Paper**: [InterspeechPaperRAPM.tex.pdf](InterspeechPaperRAPM.tex.pdf) - Interspeech 2026 TOPI Challenge System Description

## Key Results

| System | Ret. Dim | Internal (Seen) Cosine | Gain | Official (Unseen) Cosine | Gain |
|--------|----------|------------------------|------|--------------------------|------|
| **Baseline MLP** | - | 0.8732 | - | **0.8574** | - |
| **Config A: High-Res** | | | | | |
| ─ Pure Ret | 1024 | 0.8722 | - | 0.8286 | - |
| ─ Ret + Fusion | 1024 | **0.8742** | +0.0020 | 0.8290 | +0.0004 |
| **Config B: Subspace** | | | | | |
| ─ Pure Ret | 103 | 0.8730 | - | 0.8318 | - |
| ─ Ret + Fusion | 103 | 0.8741 | +0.0011 | **0.8331** | +0.0013 |

> **Note**: Internal split uses the official train/test filelists. Config B (103-dim subspace) achieves best performance on official test set with unseen speakers.

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

### Fusion Mode ⭐ **SUBMISSION MODEL (Config B)**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FUSION MODE (Config B)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Input: EN_1024 (English HuBERT Features, 1024-dim)                        │
│       │                                                                     │
│       ▼                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐          │
│   │         Enhanced Retrieval Module                           │          │
│   │  • Query Projection: 1024 → 103 (english_winners)          │          │
│   │  • Top-K Retrieval: K=70                                    │          │
│   │  • Temperature: 0.04 (Sharp Attention)                      │          │
│   │  • Similarity: Cosine                                       │          │
│   └─────────────────────────────────────────────────────────────┘          │
│       │                                                                     │
│       ▼                                                                     │
│   ES_retrieved_1024 (Retrieved Spanish Features, 1024-dim)                  │
│       │                                                                     │
│       ▼                                                                     │
│   Feature Selection (101-dim via spanish_winners)                            │
│       │                                                                     │
│       ▼                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐          │
│   │              FUSION NETWORK (MLP)                           │          │
│   │  • Input: Concat[EN_1024, ES_retrieved_101] = 1125-dim      │          │
│   │  • Architecture: [1125 → 256 → 128 → 101]                   │          │
│   │  • Activation: GELU + LayerNorm                             │          │
│   │  • Output: ES_pred = ES_retrieved + Delta                   │          │
│   └─────────────────────────────────────────────────────────────┘          │
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
@inproceedings{luo2026rapm,
  title={{R-APM: Retrieval-Augmented Pragmatic Mapper for Cross-Lingual Prosody Transfer}},
  author={Luo, Xiaoyang and Jiang, Siyuan and Yang, Shuya and Ke, Dengfeng and Xie, Yanlu and Zhang, Jinsong},
  booktitle={Interspeech 2026},
  year={2026},
  note={TOPI Challenge System Description}
}
```

## License

MIT License

## Acknowledgments

- Interspeech 2026 TOPI S2ST Challenge organizers
- DRAL Dataset creators

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

### System Architecture

**Figure 1: R-APM System Architecture**

![R-APM Architecture](docs/images/figure1_architecture.png)

### Fusion Network Design

**Figure 2: Fusion Network with Residual Connection**

![Fusion Network](docs/images/figure2_fusion.png)

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

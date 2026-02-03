# R-APM: Retrieval-Augmented Pragmatic Mapper

PyTorch implementation for cross-lingual prosody transfer using retrieval-augmented learning. This is the official implementation for Interspeech 2026 TOPI Challenge.

## Paper

**R-APM: Retrieval-Augmented Pragmatic Mapper for Cross-Lingual Prosody Transfer**
Xiaoyang Luo, Siyuan Jiang, Shuya Yang, Dengfeng Ke, Yanlu Xie, Jinsong Zhang
Interspeech 2026

> See: [InterspeechPaperRAPM.tex.pdf](../InterspeechPaperRAPM.tex.pdf)

## Project Structure

```
src/
├── models/
│   ├── retrieval.py     # Retrieval Module (Top-K kNN)
│   └── fusion.py        # Fusion Network (MLP)
├── data/
│   └── dataset.py       # Dataset loader
├── train_rapm_v2.py     # Main training script
└── README.md            # This file
```

## System Architecture

### Config A: High-Res (1024-dim)
```
EN_1024 -> Retrieval(1024-dim) -> ES_retrieved_1024 -> Selection -> ES_101
```

### Config B: Pragmatic-Subspace (103-dim) **[Submission Model]**
```
EN_1024 -> Projection(103-dim) -> Retrieval -> ES_retrieved_1024 -> Selection -> ES_101
                                                               |
                                    Fusion Network: [1125 -> 256 -> 128 -> 101]
                                                               |
                                              ES_pred = ES_retrieved + Delta
```

### Key Dimensions

| Stage | Dimension | Description |
|-------|-----------|-------------|
| Input (EN HuBERT) | 1024 | English features |
| Query Projection (Config B) | 103 | Using `english_winners` indices |
| Retrieved Value | 1024 | Full Spanish features |
| Output (ES) | 101 | Using `spanish_winners` indices |
| Fusion Input | 1125 | Concat[EN_1024, ES_retrieved_101] |

### Core Parameters

- **Retrieval**: Top-K=70, Temperature=0.04, Cosine Similarity
- **Fusion Network**: MLP [1125 -> 256 -> 128 -> 101], GELU activation, LayerNorm
- **Training**: 100 epochs, AdamW (lr=1e-3), Cosine Embedding Loss

## Key Results

| Config | Internal (Seen) | Official (Unseen) |
|--------|-----------------|-------------------|
| **Config A (1024-dim)** | | |
| Pure Retrieval | 0.8722 | 0.8286 |
| Ret + Fusion | 0.8742 | 0.8290 |
| **Config B (103-dim)** | | |
| Pure Retrieval | 0.8730 | 0.8318 |
| Ret + Fusion | 0.8741 | **0.8331** |

## Installation

```bash
pip install torch torchvision torchaudio
pip install numpy scikit-learn tqdm pyyaml
```

## Usage

### Training
```bash
python train_rapm_v2.py
```

### Evaluation
```bash
python evaluate.py --checkpoint ../checkpoints/best_model.pth
```

## Key Findings

1. **Identity Overfitting**: 1024-dim retrieval suffers from timbre interference on unseen speakers
2. **Subspace Advantage**: 103-dim retrieval improves generalization (+0.0032)
3. **Small Data Trap**: Fusion networks struggle when data density is low (N < 3k)
4. **PCA Failure**: Variance-based reduction destroys pragmatic structure (0.4456 vs 0.8741)

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

## References

- [DRAL Dataset](https://www.cs.utep.edu/nigel/dral/)
- [Challenge Page](https://www.codabench.org/competitions/12225/)
- [Baseline Code](https://github.com/mdekorte/Pragmatic_Similarity_Computation)

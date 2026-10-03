# Interspeech 2026 TOPI S2ST Challenge: RAPM

**RAPM: Retrieval-Augmented Pragmatic Mapper for Speech-to-Speech Translation**

[简体中文](README.zh-CN.md) | [日本語](README.ja.md)

## Authors

**Xiaoyang Luo**, **Siyuan Jiang**, **Shuya Yang**, **Dengfeng Ke**, **Yanlu Xie**, **Jinsong Zhang** (corresponding author)

Speech Acquisition and Intelligent Technology Laboratory (SAIT LAB)

Beijing Language and Culture University, Beijing, China

## Paper and overview

[Read the latest paper PDF](InterspeechPaperRAPM.tex.pdf). This version displays the authors and affiliation and fixes the Translatotron 2 citation.

RAPM transfers pragmatic intent and prosody from English to Spanish. It takes 1024-dimensional English HuBERT features, retrieves aligned Spanish exemplars, and predicts the 101-dimensional pragmatic feature subset selected by `spanish_winners`. We compare pure retrieval with retrieval plus a learned residual fusion network.

Config A searches the full 1024-dimensional English space. Config B searches the 103-dimensional `english_winners` subspace. Both retrieve full Spanish features before selecting the 101 output dimensions. Fusion combines the original 1024-dimensional English input and the 101-dimensional retrieved prior: `1125 → 256 → 128 → 101`, with LayerNorm and GELU. The final prediction is the retrieved prior plus the learned correction.

## Main results

The following values are reported in the paper: means over 10 runs, with standard deviations of ±0.002. Gain is the absolute improvement from adding fusion.

| System | Retrieval dimensions | Internal: seen speakers | Fusion gain | Official: unseen speakers | Fusion gain |
|---|---:|---:|---:|---:|---:|
| Baseline MLP | — | 0.8732 | — | **0.8574** | — |
| Config A: pure retrieval | 1024 | 0.8722 | — | 0.8286 | — |
| Config A: retrieval + fusion | 1024 | **0.8742** | +0.0020 | 0.8290 | +0.0004 |
| Config B: pure retrieval | 103 | 0.8730 | — | 0.8318 | — |
| Config B: retrieval + fusion | 103 | 0.8741 | +0.0011 | **0.8331** | +0.0013 |

Config B achieves the best RAPM result on the official challenge set, but remains below the MLP baseline by 0.0243. Four of the five official test speakers are absent from training. The internal split shares speakers with training; these evaluation settings must be kept separate.

### Fusion input ablation

The updated paper reports these results on the **internal split with seen speakers**, not the official blind test set.

| Fusion input | Input dimensions | Cosine similarity |
|---|---:|---:|
| Original fusion: English context + retrieved prior | 1125 | **0.8741** |
| Pure retrieval baseline | — | 0.8730 |
| Prior-only refinement | 101 | 0.8683 |
| Consistent subspace fusion | 204 | 0.8660 |

Restricting source context reduces performance on seen speakers in this study. These scores do not establish an improvement on unseen speakers. Check individual experiment scripts and their data splits before using their output to reproduce the paper tables.

## Architecture

![RAPM system architecture](docs/images/figure1_architecture.png)

![Residual fusion network](docs/images/figure2_fusion.png)

The paper uses cosine retrieval with `K=70` and temperature `0.04`. It reports fusion training for up to 100 epochs with AdamW, learning rate `1e-3`, weight decay `1e-4`, batch size 32, and cosine embedding loss. See the PDF for validation, early stopping, and evaluation details; individual repository scripts may use different training procedures.

## Installation and data

```bash
git clone https://github.com/TheGrSun/Interspeech2026-TOPI-RAPM.git
cd Interspeech2026-TOPI-RAPM
pip install -r requirements.txt
git clone https://github.com/mdekorte/Pragmatic_Similarity_Computation.git official_mdekorte
```

The baseline repository supplies `feature_selection.py` and the official file lists required by the experiment scripts. It is an external dependency, not a configured Git submodule. Obtain the [DRAL dataset](https://www.cs.utep.edu/nigel/dral/) and prepare aligned `EN_*.npy` / `ES_*.npy` features under `dral-features/features/`, with matching filename order. Dataset features and trained `.pth` weights are not included in the current Git tree.

## Running the code

From the repository root, after preparing the data and baseline dependency:

```bash
python src/train_ensemble.py --mode 103_fusion --data_dir dral-features/features --checkpoint_dir checkpoints --top_k 70 --temperature 0.04 --hidden_dims 256 128 --epochs 100 --lr 0.001 --batch_size 32 --device cuda
```

Use `--device cpu` for CPU execution. Other modes are `1024_fusion`, `1024_pure`, `103_pure`, and `all`. This script trains on the supplied feature directory and reports training cosine similarity; its score is not a held-out evaluation.

For the comparison using the baseline repository's train/test file lists:

```bash
python src/compare_all_models_official_split.py
```

Check the script's feature paths and file lists before running it. The baseline file-list test split is the internal evaluation split, distinct from the official challenge's blind test set. The current Git tree has no standalone `src/evaluate.py` or submission-generation script.

## Repository layout

| Path | Contents |
|---|---|
| `InterspeechPaperRAPM.tex.pdf` | Latest paper with authors and affiliation |
| `src/` | Training, comparison, and ablation scripts |
| `experiments/` | Additional experiment scripts |
| `scripts/` | Ensemble training and hyperparameter search |
| `config/` | Configuration examples; check paths and script compatibility |
| `checkpoints/` | Checkpoint documentation; weights are not tracked |
| `docs/images/` | Architecture figures |

## Citation

Bibliographic entry for this system-description manuscript:

```bibtex
@misc{luo2026rapm,
  title={{RAPM: Retrieval-Augmented Pragmatic Mapper for Speech-to-Speech Translation}},
  author={Luo, Xiaoyang and Jiang, Siyuan and Yang, Shuya and Ke, Dengfeng and Xie, Yanlu and Zhang, Jinsong},
  year={2026},
  howpublished={Interspeech 2026 TOPI Challenge system-description manuscript},
  url={https://github.com/TheGrSun/Interspeech2026-TOPI-RAPM}
}
```

## License and acknowledgments

[MIT License](LICENSE). We thank the Interspeech 2026 TOPI S2ST Challenge organizers, the DRAL dataset creators, and the baseline implementation authors.

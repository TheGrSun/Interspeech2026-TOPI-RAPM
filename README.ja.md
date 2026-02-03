# Interspeech 2026 TOPI S2ST Challenge: R-APM

**検索強型プラグマティックマッパーによる言語間プロディ転送**

[English](README.md) | [简体中文](README.zh-CN.md)

---

## 著者

**Xiaoyang Luo**、**Siyuan Jiang**、**Shuya Yang**、**Dengfeng Ke**、**Yanlu Xie**、**Jinsong Zhang**

Speech Acquisition and Intelligent Technology Laboratory (SAIT LAB)
Beijing Language and Culture University, Beijing, China

---

## 概要

R-APM は検索ベースの言語間プロディ転送システムです。英語 HuBERT 特徴量（1024次元）からスペイン語プロディ特徴量（101次元）を予測します。ハイブリッド検索+融合アーキテクチャを採用しています。

> **📄 論文**: [InterspeechPaperRAPM.tex.pdf](InterspeechPaperRAPM.tex.pdf) - Interspeech 2026 TOPI Challenge システム記述

## 主な結果

| システム | 検索次元 | 内部(既見) コサイン類似度 | 改善 | 公式(未見) コサイン類似度 | 改善 |
|--------|----------|-------------------------|------|-------------------------|------|
| **Baseline MLP** | - | 0.8732 | - | **0.8574** | - |
| **Config A: High-Res** | | | | | |
| ─ Pure Ret | 1024 | 0.8722 | - | 0.8286 | - |
| ─ Ret + Fusion | 1024 | **0.8742** | +0.0020 | 0.8290 | +0.0004 |
| **Config B: Subspace** | | | | | |
| ─ Pure Ret | 103 | 0.8730 | - | 0.8318 | - |
| ─ Ret + Fusion | 103 | 0.8741 | +0.0011 | **0.8331** | +0.0013 |

> **注意**: 内部分割は公式訓練/テストファイルリストを使用します。Config B（103次元部分空間）が未見話者を含む公式テストセットで最高性能を達成しました。

## アーキテクチャ

### システムアーキテクチャ

**図 1: R-APM システムアーキテクチャ**

![R-APM アーキテクチャ](docs/images/figure1_architecture.png)

### 融合ネットワーク設計

**図 2: 残差接続付き融合ネットワーク**

![融合ネットワーク](docs/images/figure2_fusion.png)

## インストール

```bash
git clone --recurse-submodules https://github.com/TheGrSun/Interspeech2026-TOPI-RAPM.git
cd Interspeech2026-TOPI-RAPM
pip install -r requirements.txt
```

## 使用方法

```bash
# 訓練
python src/train.py --config config/default.yaml

# 評価
python src/evaluate.py --checkpoint checkpoints/best_model.pth

# 予測生成
cd submit
python generate_submission.py
```

## データセット

DRAL データセットは以下からダウンロードしてください: https://www.cs.utep.edu/nigel/dral/

## 引用

```bibtex
@inproceedings{luo2026rapm,
  title={{R-APM: Retrieval-Augmented Pragmatic Mapper for Cross-Lingual Prosody Transfer}},
  author={Luo, Xiaoyang and Jiang, Siyuan and Yang, Shuya and Ke, Dengfeng and Xie, Yanlu and Zhang, Jinsong},
  booktitle={Interspeech 2026},
  year={2026},
  note={TOPI Challenge System Description}
}
```

## ライセンス

MIT License

## 謝辞

- Interspeech 2026 TOPI S2ST Challenge 主催者
- DRAL データセット作成者

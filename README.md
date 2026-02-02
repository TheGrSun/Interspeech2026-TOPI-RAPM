# Interspeech 2026 TOPI S2ST Challenge: R-APM

**Retrieval-Augmented Pragmatic Mapper for Cross-Lingual Prosody Transfer**

[English](#english) | [简体中文](#简体中文) | [日本語](#日本語)

---

## English

### Overview

R-APM is a retrieval-based system for cross-lingual prosody transfer from English to Spanish. It predicts Spanish HuBERT prosodic features (101-dim) from English HuBERT features (1024-dim) using a hybrid retrieval + fusion architecture.

> **📄 System Description**: [R-APM: Retrieval-Augmented Pragmatic Mapper for Cross-Lingual Prosody Transfer.PDF](docs/R-APM_System_Description.pdf) (Coming Soon)

### Key Results

| Model | Internal Split | Official Test |
|-------|----------------|---------------|
| **1024-Fusion** (Submission) | **0.8742** | **0.8288** |
| 1024-Pure | 0.8722 | - |
| 103-Fusion | 0.8654 | - |
| 103-Pure | 0.8642 | - |
| MLP Baseline | 0.8732 | - |

### Architecture

#### Pure Retrieval Mode

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
│   │  • Method: Variance-based Selection                         │          │
│   │  • Output: 101-dim (Official competition format)            │          │
│   └─────────────────────────────────────────────────────────────┘          │
│       │                                                                     │
│       ▼                                                                     │
│   Output: ES_101 (Spanish Prosodic Features, 101-dim)                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Fusion Mode ⭐ **SUBMISSION MODEL**

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

### 1024 vs 103 Retrieval: What's the Difference?

| Aspect | **1024 Retrieval** | **103 Retrieval** |
|--------|-------------------|-------------------|
| **Search Space** | Full HuBERT features (1024-dim) | Pre-selected variance features (103-dim) |
| **Semantic Info** | ✅ Preserves complete semantic information | ❌ Loses some semantic information |
| **Paralinguistic Info** | ✅ Preserves complete paralinguistic cues | ⚠️ Partially preserved |
| **Performance** | **0.8742** (Best) | 0.8654 (-1.01%) |
| **Use Case** | **Final submission** | Ablation study only |

**Key Finding**: 1024-dim retrieval space preserves complete information and achieves best performance. The 103-dim variant was used for ablation to understand the contribution of full HuBERT features.

### Installation

```bash
git clone --recurse-submodules https://github.com/TheGrSun/Interspeech2026-TOPI-RAPM.git
cd Interspeech2026-TOPI-RAPM
pip install -r requirements.txt
```

### Usage

```bash
# Training
python src/train.py --config config/default.yaml

# Evaluation
python src/evaluate.py --checkpoint checkpoints/best_model.pth

# Generate submission
cd submit
python generate_submission.py
```

### Dataset

Download the DRAL dataset from: https://www.cs.utep.edu/nigel/dral/

### Citation

```bibtex
@inproceedings{rapm2026,
  title={Retrieval-Augmented Pragmatic Mapper for Cross-Lingual Prosody Transfer},
  author={Your Name},
  booktitle={Interspeech 2026},
  year={2026}
}
```

---

## 简体中文

### 概述

R-APM是一个基于检索的跨语言韵律迁移系统，从英语HuBERT特征(1024维)预测西班牙语韵律特征(101维)，采用混合检索+融合架构。

> **📄 系统描述**: [R-APM: Retrieval-Augmented Pragmatic Mapper for Cross-Lingual Prosody Transfer.PDF](docs/R-APM_System_Description.pdf) (即将发布)

### 主要结果

| 模型 | 内部分割 | 官方测试集 |
|-------|----------------|---------------|
| **1024-Fusion** (提交模型) | **0.8742** | **0.8288** |
| 1024-Pure | 0.8722 | - |
| 103-Fusion | 0.8654 | - |
| 103-Pure | 0.8642 | - |
| MLP Baseline | 0.8732 | - |

### 架构

#### 纯检索模式 (Pure Retrieval)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              纯检索模式 (Pure)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   输入: EN_1024 (英语HuBERT特征, 1024维)                                    │
│       │                                                                     │
│       ▼                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐          │
│   │         增强检索模块                                        │          │
│   │  • 相似度: 余弦相似度                                       │          │
│   │  • Top-K检索: K=70                                         │          │
│   │  • 温度参数: 0.04 (锐化注意力)                              │          │
│   └─────────────────────────────────────────────────────────────┘          │
│       │                                                                     │
│       ▼                                                                     │
│   ES_retrieved_1024 (检索到的西班牙语特征, 1024维)                          │
│       │                                                                     │
│       ▼                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐          │
│   │         特征选择模块                                        │          │
│   │  • 方法: 基于方差的特征选择                                 │          │
│   │  • 输出: 101维 (官方竞赛格式)                               │          │
│   └─────────────────────────────────────────────────────────────┘          │
│       │                                                                     │
│       ▼                                                                     │
│   输出: ES_101 (西班牙语韵律特征, 101维)                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 融合模式 (Fusion Mode) ⭐ **提交模型**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           融合模式 (Fusion) - 提交模型                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   输入: EN_1024 (英语HuBERT特征, 1024维)                                    │
│       │                                                                     │
│       ├────────────────────────────────────────┐                            │
│       ▼                                        ▼                            │
│   ┌─────────────────────┐         ┌─────────────────────────┐             │
│   │   增强检索          │         │   直接投影               │             │
│   │ • Top-K=70          │         │   EN_1024 → 101         │             │
│   │ • Temp=0.04         │         └─────────────────────────┘             │
│   └─────────────────────┘                    │                             │
│       │                                      │                             │
│       ▼                                      │                             │
│   ES_retrieved_1024                          │                             │
│       │                                      │                             │
│       ▼                                      │                             │
│   特征选择 (101维)                           │                             │
│       │                                      │                             │
│       ▼                                      ▼                             │
│   ┌─────────────────────────────────────────────────────┐                  │
│   │              融合网络 (Fusion Network)               │                  │
│   │  ┌─────────────────────────────────────────────┐    │                  │
│   │  │  • 多头自注意力机制 (8 heads)               │    │                  │
│   │  │  • 多尺度 MLP: [256, 128, 64]               │    │                  │
│   │  │  • 门控机制 (基于注意力)                     │    │                  │
│   │  │  • 层归一化 + Dropout(0.0)                  │    │                  │
│   │  └─────────────────────────────────────────────┘    │                  │
│   │                                                     │                  │
│   │  输出 = ES_retrieved + Delta(EN_input)             │                  │
│   └─────────────────────────────────────────────────────┘                  │
│       │                                                                     │
│       ▼                                                                     │
│   输出: ES_101_fused (西班牙语韵律特征, 101维)                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1024检索 vs 103检索: 有什么区别？

| 方面 | **1024检索** | **103检索** |
|------|-------------|-------------|
| **检索空间** | 完整HuBERT特征 (1024维) | 预选方差特征 (103维) |
| **语义信息** | ✅ 保留完整语义信息 | ❌ 丢失部分语义信息 |
| **副语言信息** | ✅ 保留完整副语言线索 | ⚠️ 部分保留 |
| **性能** | **0.8742** (最佳) | 0.8654 (-1.01%) |
| **用途** | **最终提交** | 仅用于消融研究 |

**关键发现**: 1024维检索空间保留完整信息并取得最佳性能。103维变体仅用于消融研究，以理解完整HuBERT特征的贡献。

### 安装

```bash
git clone --recurse-submodules https://github.com/TheGrSun/Interspeech2026-TOPI-RAPM.git
cd Interspeech2026-TOPI-RAPM
pip install -r requirements.txt
```

### 使用方法

```bash
# 训练
python src/train.py --config config/default.yaml

# 评估
python src/evaluate.py --checkpoint checkpoints/best_model.pth

# 生成提交
cd submit
python generate_submission.py
```

### 数据集

从以下地址下载DRAL数据集: https://www.cs.utep.edu/nigel/dral/

### 引用

```bibtex
@inproceedings{rapm2026,
  title={检索增强的韵律映射器用于跨语言韵律迁移},
  author={您的名字},
  booktitle={Interspeech 2026},
  year={2026}
}
```

---

## 日本語

### 概要

R-APMは検索ベースの言語間プロディ転送システムです。英語HuBERT特徴量(1024次元)からスペイン語プロディ特徴量(101次元)を予測します。ハイブリッド検索+融合アーキテクチャを採用しています。

> **📄 システム記述**: [R-APM: Retrieval-Augmented Pragmatic Mapper for Cross-Lingual Prosody Transfer.PDF](docs/R-APM_System_Description.pdf) (近日公開)

### 主な結果

| モデル | 内部分割 | 公式テスト |
|-------|----------------|---------------|
| **1024-Fusion** (提出モデル) | **0.8742** | **0.8288** |
| 1024-Pure | 0.8722 | - |
| 103-Fusion | 0.8654 | - |
| 103-Pure | 0.8642 | - |
| MLP Baseline | 0.8732 | - |

### アーキテクチャ

#### 純粋検索モード (Pure Retrieval)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              純粋検索モード (Pure)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   入力: EN_1024 (英語HuBERT特徴量, 1024次元)                                 │
│       │                                                                     │
│       ▼                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐          │
│   │         強化検索モジュール                                   │          │
│   │  • 類似度: コサイン類似度                                    │          │
│   │  • Top-K検索: K=70                                          │          │
│   │  • 温度パラメータ: 0.04 (鋭い注意力)                        │          │
│   └─────────────────────────────────────────────────────────────┘          │
│       │                                                                     │
│       ▼                                                                     │
│   ES_retrieved_1024 (検索されたスペイン語特徴, 1024次元)                    │
│       │                                                                     │
│       ▼                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐          │
│   │         特徴選択モジュール                                   │          │
│   │  • 方法: 分散に基づく特徴選択                               │          │
│   │  • 出力: 101次元 (公式コンテスト形式)                       │          │
│   └─────────────────────────────────────────────────────────────┘          │
│       │                                                                     │
│       ▼                                                                     │
│   出力: ES_101 (スペイン語プロディ特徴, 101次元)                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 融合モード (Fusion Mode) ⭐ **提出モデル**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           融合モード (Fusion) - 提出モデル                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   入力: EN_1024 (英語HuBERT特徴量, 1024次元)                                 │
│       │                                                                     │
│       ├────────────────────────────────────────┐                            │
│       ▼                                        ▼                            │
│   ┌─────────────────────┐         ┌─────────────────────────┐             │
│   │   強化検索          │         │   直接射影               │             │
│   │ • Top-K=70          │         │   EN_1024 → 101         │             │
│   │ • Temp=0.04         │         └─────────────────────────┘             │
│   └─────────────────────┘                    │                             │
│       │                                      │                             │
│       ▼                                      │                             │
│   ES_retrieved_1024                          │                             │
│       │                                      │                             │
│       ▼                                      │                             │
│   特徴選択 (101次元)                         │                             │
│       │                                      │                             │
│       ▼                                      ▼                             │
│   ┌─────────────────────────────────────────────────────┐                  │
│   │              融合ネットワーク (Fusion Network)        │                  │
│   │  ┌─────────────────────────────────────────────┐    │                  │
│   │  │  • マルチヘッド自己注意 (8 heads)            │    │                  │
│   │  │  • マルチスケール MLP: [256, 128, 64]        │    │                  │
│   │  │  • ゲーティング機構 (注意ベース)              │    │                  │
│   │  │  • 層正規化 + Dropout(0.0)                   │    │                  │
│   │  └─────────────────────────────────────────────┘    │                  │
│   │                                                     │                  │
│   │  出力 = ES_retrieved + Delta(EN_input)              │                  │
│   └─────────────────────────────────────────────────────┘                  │
│       │                                                                     │
│       ▼                                                                     │
│   出力: ES_101_fused (スペイン語プロディ特徴, 101次元)                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1024検索 vs 103検索: 何が違いますか？

| 側面 | **1024検索** | **103検索** |
|------|-------------|-------------|
| **検索空間** | 完全HuBERT特徴 (1024次元) | 事前選択分散特徴 (103次元) |
| **意味情報** | ✅ 完全な意味情報を保持 | ❌ 一部の意味情報が失われる |
| **副言語情報** | ✅ 完全な副言語の手がかりを保持 | ⚠️ 部分的に保持 |
| **性能** | **0.8742** (最高) | 0.8654 (-1.01%) |
| **用途** | **最終提出** | アブレーション研究のみ |

**重要な発見**: 1024次元検索空間は完全な情報を保持し、最高の性能を達成します。103次元変体は、完全なHuBERT特徴の寄与を理解するためのアブレーション研究にのみ使用されました。

### インストール

```bash
git clone --recurse-submodules https://github.com/TheGrSun/Interspeech2026-TOPI-RAPM.git
cd Interspeech2026-TOPI-RAPM
pip install -r requirements.txt
```

### 使用方法

```bash
# 訓練
python src/train.py --config config/default.yaml

# 評価
python src/evaluate.py --checkpoint checkpoints/best_model.pth

# 予測生成
cd submit
python generate_submission.py
```

### データセット

DRALデータセットは以下からダウンロードしてください: https://www.cs.utep.edu/nigel/dral/

### 引用

```bibtex
@inproceedings{rapm2026,
  title={検索強型プラグマティックマッパーによる言語間プロディ転送},
  author={あなたの名前},
  booktitle={Interspeech 2026},
  year={2026}
}
```

---

## License

MIT License

## Acknowledgments

- Interspeech 2026 TOPI S2ST Challenge organizers
- DRAL Dataset creators

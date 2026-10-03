# Interspeech 2026 TOPI S2ST Challenge: RAPM

**RAPM：面向语音到语音翻译的检索增强语用映射器**

[English](README.md) | [日本語](README.ja.md)

## 作者

**Xiaoyang Luo**、**Siyuan Jiang**、**Shuya Yang**、**Dengfeng Ke**、**Yanlu Xie**、**Jinsong Zhang**（通讯作者）

语音获取与智能技术实验室（SAIT LAB）

北京语言大学，中国北京

## 论文与概述

[阅读最新版论文 PDF](InterspeechPaperRAPM.tex.pdf)。当前版本已显示作者和单位，并修正 Translatotron 2 的引用。

论文标题：**RAPM: Retrieval-Augmented Pragmatic Mapper for Speech-to-Speech Translation**。

RAPM 用于英语到西班牙语的语用意图与韵律迁移。系统接收 1024 维英语 HuBERT 特征，检索对齐的西班牙语样本，预测由 `spanish_winners` 选出的 101 维语用特征，并对比纯检索与加入残差融合网络的方案。

Config A 在完整的 1024 维英语空间中检索，Config B 在 `english_winners` 选出的 103 维子空间中检索。两者均先检索完整的西班牙语特征，再选出 101 维输出。融合网络拼接原始 1024 维英语输入和 101 维检索先验，采用 `1125 → 256 → 128 → 101` 的 MLP、LayerNorm 和 GELU。最终预测为检索先验加上学习得到的残差修正。

## 主要结果

下表为论文报告的结果：10 次运行的均值，标准差为 ±0.002。融合增益表示加入融合网络后的绝对提升。

| 系统 | 检索维度 | 内部测试：已见说话人 | 融合增益 | 官方测试：未见说话人 | 融合增益 |
|---|---:|---:|---:|---:|---:|
| Baseline MLP | — | 0.8732 | — | **0.8574** | — |
| Config A: 纯检索 | 1024 | 0.8722 | — | 0.8286 | — |
| Config A: 检索 + 融合 | 1024 | **0.8742** | +0.0020 | 0.8290 | +0.0004 |
| Config B: 纯检索 | 103 | 0.8730 | — | 0.8318 | — |
| Config B: 检索 + 融合 | 103 | 0.8741 | +0.0011 | **0.8331** | +0.0013 |

Config B 在 RAPM 各方案中取得最佳官方测试结果，但仍比 MLP 基线低 0.0243。官方测试集的五位说话人中有四位未出现在训练语料中；内部测试集与训练集共享说话人，两种评估条件需要区分。

### 融合输入消融

新版论文将以下结果报告为**内部测试集（已见说话人）**上的实验，不属于官方盲测结果。

| 融合输入 | 输入维度 | 余弦相似度 |
|---|---:|---:|
| 原始融合：英语上下文 + 检索先验 | 1125 | **0.8741** |
| 纯检索基线 | — | 0.8730 |
| 仅先验修正 | 101 | 0.8683 |
| 一致子空间融合 | 204 | 0.8660 |

在该实验中，限制源语言上下文会降低已见说话人上的性能。这些分数不能证明未见说话人上的性能有所提升。使用独立实验脚本复现论文表格前，需要核对脚本和数据划分。

## 系统架构

![RAPM 系统架构](docs/images/figure1_architecture.png)

![残差融合网络](docs/images/figure2_fusion.png)

论文采用余弦检索，`K=70`、温度 `0.04`。融合训练最多运行 100 个 epoch，使用 AdamW、学习率 `1e-3`、权重衰减 `1e-4`、批大小 32 和余弦嵌入损失。验证集、早停和评估细节以 PDF 为准；仓库内不同脚本的训练流程可能不同。

## 安装与数据准备

```bash
git clone https://github.com/TheGrSun/Interspeech2026-TOPI-RAPM.git
cd Interspeech2026-TOPI-RAPM
pip install -r requirements.txt
git clone https://github.com/mdekorte/Pragmatic_Similarity_Computation.git official_mdekorte
```

基线仓库提供实验所需的 `feature_selection.py` 和官方文件列表，它是外部依赖，当前项目没有将其配置为 Git 子模块。获取 [DRAL 数据集](https://www.cs.utep.edu/nigel/dral/)，将对齐的 `EN_*.npy` / `ES_*.npy` 特征放入 `dral-features/features/`，确保两种语言按文件名排序后正确配对。当前 Git 文件树不包含数据特征或训练好的 `.pth` 权重。

## 运行代码

准备好数据和基线依赖后，在仓库根目录运行：

```bash
python src/train_ensemble.py --mode 103_fusion --data_dir dral-features/features --checkpoint_dir checkpoints --top_k 70 --temperature 0.04 --hidden_dims 256 128 --epochs 100 --lr 0.001 --batch_size 32 --device cuda
```

使用 CPU 时改为 `--device cpu`。其他模式包括 `1024_fusion`、`1024_pure`、`103_pure` 和 `all`。该脚本在传入目录的特征上训练，报告的是训练集余弦相似度，不能作为独立测试结果。

按基线仓库的训练/测试文件列表进行方法比较：

```bash
python src/compare_all_models_official_split.py
```

运行前检查脚本中的特征路径和文件列表。基线文件列表对应内部评估划分，与挑战赛官方盲测集不同。当前 Git 文件树没有独立的 `src/evaluate.py` 或提交文件生成脚本。

## 仓库结构

| 路径 | 内容 |
|---|---|
| `InterspeechPaperRAPM.tex.pdf` | 包含作者和单位的最新版论文 |
| `src/` | 训练、方法比较及消融脚本 |
| `experiments/` | 其他实验脚本 |
| `scripts/` | 集成训练与超参数搜索 |
| `config/` | 配置示例；使用前核对路径及脚本兼容性 |
| `checkpoints/` | 权重说明文档；权重尚未纳入版本控制 |
| `docs/images/` | 架构图 |

## 引用

以下条目用于引用本系统描述稿：

```bibtex
@misc{luo2026rapm,
  title={{RAPM: Retrieval-Augmented Pragmatic Mapper for Speech-to-Speech Translation}},
  author={Luo, Xiaoyang and Jiang, Siyuan and Yang, Shuya and Ke, Dengfeng and Xie, Yanlu and Zhang, Jinsong},
  year={2026},
  howpublished={Interspeech 2026 TOPI Challenge system-description manuscript},
  url={https://github.com/TheGrSun/Interspeech2026-TOPI-RAPM}
}
```

## 许可证与致谢

[MIT License](LICENSE)。感谢 Interspeech 2026 TOPI S2ST Challenge 组织者、DRAL 数据集创建者及基线实现作者。

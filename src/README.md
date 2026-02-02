# Retrieval-Augmented Pragmatic Mapper

PyTorch implementation for cross-lingual prosody transfer using retrieval-augmented learning.

## 项目结构

```
src/
├── models/
│   ├── encoder.py       # 语用编码器 (101→64)
│   ├── retrieval.py     # 检索模块 (top-k=5)
│   ├── fusion.py        # 融合网络 (266→101)
│   ├── gating.py        # 门控机制 (167→1)
│   └── mapper.py        # 完整模型组装
├── data/
│   ├── pca_reducer.py   # PCA降维器 (1024→101)
│   └── dataset.py       # 数据集加载器
├── losses/
│   └── losses.py        # 损失函数
├── config.py            # 超参数配置
├── train.py             # 训练脚本
├── evaluate.py          # 评估脚本
└── README.md            # 本文件
```

## 关键维度说明

经过分析官方数据和baseline，确认的维度流程：

| 阶段 | 维度 | 说明 |
|------|------|------|
| 原始HuBERT特征 | 1024 | 从dral-features下载的原始数据 |
| PCA降维 | **101** | 使用PCA保留85%方差，**这是提交格式** |
| 语用编码器输出 | 64 | 跨语言语用表示空间 |
| 模型输出 | **101** | 预测的西班牙语特征（可直接提交）|

## 快速开始

### 1. 安装依赖

```bash
pip install torch torchvision torchaudio
pip install numpy scikit-learn tqdm
```

### 2. 准备PCA降维模型

PCA模型已自动生成在 `../checkpoints/pca_reducer_101.pkl`

如需重新生成：

```bash
cd data
python pca_reducer.py
```

### 3. 训练模型（三阶段）

```bash
python train.py
```

训练分为三个阶段：
- **Stage 1 (Epochs 1-30)**: 语用空间预训练，仅对比学习
- **Stage 2 (Epochs 31-150)**: 联合训练，所有损失
- **Stage 3 (Epochs 151-200)**: 精调，降低学习率

### 4. 评估和生成提交文件

```bash
python evaluate.py
```

生成的预测文件位于 `../submissions/predictions/`，格式为 `ES_xxx_x.npy` (101维)

## 模型架构

### 数据流

```
EN特征(101) 
    ↓
语用编码器 → 语用表示(64)
    ↓
检索模块 → 检索SP特征(101) + 置信度 + 熵
    ↓
融合网络 ← [EN特征, 检索SP, 语用表示]
    ↓
生成SP特征(101)
    ↓
门控机制 → 权重g ∈ [0,1]
    ↓
最终输出 = g·检索 + (1-g)·生成
```

### 核心参数

- **编码器**: 101→256→128→64, L2归一化
- **检索**: top-k=5, temperature=0.1
- **融合**: 266→512→256→101, 带残差旁路
- **门控**: 167→64→32→1, Sigmoid输出

## 损失函数

| 损失项 | 权重 | 说明 |
|--------|------|------|
| 重建损失 | 1.0 | MSE(pred, target) |
| 对比损失 | 0.5 | InfoNCE，语用空间对齐 |
| 分布对齐 | 0.1 | 均值+方差对齐 |
| 门控正则 | 0.05 | 熵正则，防止退化 |
| L2正则 | 0.01 | 参数正则化 |

## 训练配置

### Stage 1 (对比学习预训练)
- Learning Rate: 1e-3
- Batch Size: 64
- Epochs: 30
- 只训练编码器，冻结融合和门控

### Stage 2 (联合训练)
- Learning Rate: 1e-3 (encoder), 5e-4 (others)
- Batch Size: 64
- Epochs: 120
- 全部模块激活，使用所有损失

### Stage 3 (微调)
- Learning Rate: 1e-4
- Batch Size: 64
- Epochs: 50
- 降低学习率精调

## 数据增强

训练时应用以下增强（概率性）：
- 高斯噪声 (σ=0.05, p=0.3)
- 特征Dropout (p_drop=0.1, p=0.2)
- 特征缩放 (scale∈[0.9,1.1], p=0.2)
- 特征偏移 (shift∈[-0.1,0.1], p=0.1)

## 评估指标

- **MSE**: Mean Squared Error (主要指标)
- **Euclidean Distance**: 欧氏距离 (官方评估指标)

## Checkpoint

模型checkpoint保存在 `../checkpoints/`:
- `pca_reducer_101.pkl`: PCA降维模型
- `stage1_epoch_X.pth`: Stage 1 checkpoint
- `stage2_epoch_X.pth`: Stage 2 checkpoint
- `best_epoch_X.pth`: 验证集最佳模型

## 提交格式

生成的文件满足CodaBench要求：
- 每个EN输入对应一个ES输出
- 文件名: `ES_xxx_x.npy`
- 维度: (101,) numpy array
- 需要压缩成zip后提交

## 理论依据

详见项目根目录的 `theory.md` 和 `plan.md`：
- 检索增强学习的理论基础
- 语用表示空间的构建
- 门控机制的动态权衡
- 三阶段课程学习策略

## 引用

基于 DRAL 数据集：
```
@inproceedings{avila23_interspeech,
  author={Jonathan E. Avila and Nigel G. Ward},
  title={{Towards Cross-Language Prosody Transfer for Dialog}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
  pages={2143--2147}
}
```

## 参考资源

- [DRAL Dataset](https://www.cs.utep.edu/nigel/dral/)
- [GitHub: DRAL](https://github.com/joneavila/DRAL)
- [Challenge Page](https://www.codabench.org/competitions/12225/)
- [Baseline Code](https://github.com/mdekorte/Pragmatic_Similarity_Computation)


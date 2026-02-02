"""
消融实验：聚合函数和Dropout的最佳配置

测试的聚合方法：
1. Softmax (current, with temperature)
2. Top-K直接平均 (无权重)
3. 余弦加权 (不做softmax)

测试的Dropout值：
1. 0.1 (current)
2. 0.3
3. 0.5

使用官方filelist划分
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.feature_selector import FeatureSelector

# ============================================================================
# 数据加载
# ============================================================================
def load_filelist(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def load_features_from_filelist(filelist, feature_dir):
    features = []
    for filename in tqdm(filelist, desc="Loading features"):
        npy_name = filename.replace('.wav', '_features.npy')
        feature_path = os.path.join(feature_dir, npy_name)
        if os.path.exists(feature_path):
            features.append(np.load(feature_path))
    return np.array(features)


def load_data_official_split():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    feature_dir = os.path.join(project_root, "dral-features/features")
    filelists_dir = os.path.join(project_root, "official_mdekorte/data/filelists")

    print("Loading data with official split...")
    en_train_files = load_filelist(os.path.join(filelists_dir, "train_filelist_en.txt"))
    es_train_files = load_filelist(os.path.join(filelists_dir, "train_filelist_es.txt"))
    en_test_files = load_filelist(os.path.join(filelists_dir, "test_filelist_en.txt"))
    es_test_files = load_filelist(os.path.join(filelists_dir, "test_filelist_es.txt"))

    print(f"  Train: {len(en_train_files)} samples")
    print(f"  Test:  {len(en_test_files)} samples")

    en_train_1024 = load_features_from_filelist(en_train_files, feature_dir)
    es_train_1024 = load_features_from_filelist(es_train_files, feature_dir)
    en_test_1024 = load_features_from_filelist(en_test_files, feature_dir)
    es_test_1024 = load_features_from_filelist(es_test_files, feature_dir)

    es_selector = FeatureSelector.from_official('spanish')
    es_train_101 = es_selector.transform(es_train_1024)
    es_test_101 = es_selector.transform(es_test_1024)

    spanish_indices = torch.from_numpy(es_selector.selected_indices).long()

    return {
        'train': {
            'en_1024': torch.tensor(en_train_1024).float(),
            'es_1024': torch.tensor(es_train_1024).float(),
            'es_101': torch.tensor(es_train_101).float()
        },
        'test': {
            'en_1024': torch.tensor(en_test_1024).float(),
            'es_1024': torch.tensor(es_test_1024).float(),
            'es_101': torch.tensor(es_test_101).float()
        },
        'spanish_indices': spanish_indices
    }


# ============================================================================
# 不同的检索聚合方法
# ============================================================================
class RetrievalSoftmax(nn.Module):
    """标准: Softmax加权聚合"""
    def __init__(self, top_k=70, temperature=0.04):
        super().__init__()
        self.top_k = top_k
        self.temperature = temperature
        self.en_db = None
        self.es_db = None

    def set_database(self, en_features, es_features):
        self.en_db = torch.tensor(en_features).float() if isinstance(en_features, np.ndarray) else en_features
        self.es_db = torch.tensor(es_features).float() if isinstance(es_features, np.ndarray) else es_features

    def forward(self, query):
        device = query.device
        en_db = self.en_db.to(device)
        es_db = self.es_db.to(device)

        query_norm = F.normalize(query, dim=-1)
        db_norm = F.normalize(en_db, dim=-1)
        similarities = torch.matmul(query_norm, db_norm.T)

        k = min(self.top_k, en_db.shape[0])
        topk_sims, topk_indices = torch.topk(similarities, k=k, dim=1)

        # Softmax加权
        attn_weights = F.softmax(topk_sims / self.temperature, dim=1)
        topk_es = es_db[topk_indices]
        retrieved = torch.bmm(attn_weights.unsqueeze(1), topk_es).squeeze(1)

        return retrieved


class RetrievalDirectAvg(nn.Module):
    """Top-K直接平均（无权重）"""
    def __init__(self, top_k=70):
        super().__init__()
        self.top_k = top_k
        self.en_db = None
        self.es_db = None

    def set_database(self, en_features, es_features):
        self.en_db = torch.tensor(en_features).float() if isinstance(en_features, np.ndarray) else en_features
        self.es_db = torch.tensor(es_features).float() if isinstance(es_features, np.ndarray) else es_features

    def forward(self, query):
        device = query.device
        en_db = self.en_db.to(device)
        es_db = self.es_db.to(device)

        query_norm = F.normalize(query, dim=-1)
        db_norm = F.normalize(en_db, dim=-1)
        similarities = torch.matmul(query_norm, db_norm.T)

        k = min(self.top_k, en_db.shape[0])
        topk_sims, topk_indices = torch.topk(similarities, k=k, dim=1)

        # 均匀权重（直接平均）
        attn_weights = torch.ones_like(topk_sims) / k
        topk_es = es_db[topk_indices]
        retrieved = torch.bmm(attn_weights.unsqueeze(1), topk_es).squeeze(1)

        return retrieved


class RetrievalCosineWeighted(nn.Module):
    """余弦加权（不做softmax，只归一化）"""
    def __init__(self, top_k=70):
        super().__init__()
        self.top_k = top_k
        self.en_db = None
        self.es_db = None

    def set_database(self, en_features, es_features):
        self.en_db = torch.tensor(en_features).float() if isinstance(en_features, np.ndarray) else en_features
        self.es_db = torch.tensor(es_features).float() if isinstance(es_features, np.ndarray) else es_features

    def forward(self, query):
        device = query.device
        en_db = self.en_db.to(device)
        es_db = self.es_db.to(device)

        query_norm = F.normalize(query, dim=-1)
        db_norm = F.normalize(en_db, dim=-1)
        similarities = torch.matmul(query_norm, db_norm.T)

        k = min(self.top_k, en_db.shape[0])
        topk_sims, topk_indices = torch.topk(similarities, k=k, dim=1)

        # 归一化到[0,1]再归一化和为1（不做softmax的锐化）
        sims_normalized = (topk_sims + 1) / 2  # [-1,1] -> [0,1]
        attn_weights = sims_normalized / sims_normalized.sum(dim=1, keepdim=True)

        topk_es = es_db[topk_indices]
        retrieved = torch.bmm(attn_weights.unsqueeze(1), topk_es).squeeze(1)

        return retrieved


# ============================================================================
# Fusion网络（可配置dropout）
# ============================================================================
class FusionNetwork(nn.Module):
    def __init__(self, en_dim=1024, es_dim=101, dropout=0.1):
        super().__init__()
        input_dim = en_dim + es_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, es_dim)
        )
        self.residual_weight = nn.Parameter(torch.tensor(0.8))

    def forward(self, en_features, retrieved_es):
        combined = torch.cat([en_features, retrieved_es], dim=-1)
        delta = self.net(combined)
        w = torch.sigmoid(self.residual_weight)
        output = w * retrieved_es + (1 - w) * (retrieved_es + delta)
        return output


class RetrievalFusionModel(nn.Module):
    def __init__(self, retrieval, fusion, selector_indices):
        super().__init__()
        self.retrieval = retrieval
        self.fusion = fusion
        self.register_buffer('selector_indices', selector_indices)

    def forward(self, en_features):
        retrieved_1024 = self.retrieval(en_features)
        retrieved_101 = retrieved_1024[:, self.selector_indices]
        output = self.fusion(en_features, retrieved_101)
        return output, retrieved_101


# ============================================================================
# 评估函数
# ============================================================================
def compute_cosine_similarity(pred, target):
    pred_norm = F.normalize(pred, dim=-1)
    target_norm = F.normalize(target, dim=-1)
    return (pred_norm * target_norm).sum(dim=-1).mean().item()


def evaluate_retrieval(retrieval, data, device):
    """评估纯检索"""
    retrieval.eval()
    with torch.no_grad():
        test_en = data['test']['en_1024'].to(device)
        test_es = data['test']['es_101'].to(device)
        spanish_indices = data['spanish_indices'].to(device)

        retrieved_1024 = retrieval(test_en)
        retrieved_101 = retrieved_1024[:, spanish_indices]
        test_cos = compute_cosine_similarity(retrieved_101, test_es)

    return test_cos


def train_and_evaluate_fusion(data, retrieval_cls, dropout, device, epochs=100):
    """训练Fusion并评估"""
    # 创建检索模块
    retrieval = retrieval_cls()
    retrieval.set_database(
        data['train']['en_1024'].cpu().numpy(),
        data['train']['es_1024'].cpu().numpy()
    )

    # 创建Fusion模块
    fusion = FusionNetwork(en_dim=1024, es_dim=101, dropout=dropout)
    model = RetrievalFusionModel(retrieval, fusion, data['spanish_indices']).to(device)

    # 划分验证集
    val_size = len(data['train']['en_1024']) // 10
    indices = torch.randperm(len(data['train']['en_1024']))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    train_en = data['train']['en_1024'][train_idx].to(device)
    train_es = data['train']['es_101'][train_idx].to(device)
    val_en = data['train']['en_1024'][val_idx].to(device)
    val_es = data['train']['es_101'][val_idx].to(device)

    optimizer = optim.AdamW(model.fusion.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_cos = 0
    best_state = None

    # 训练
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        output, _ = model(train_en)
        pred_norm = F.normalize(output, dim=-1)
        target_norm = F.normalize(train_es, dim=-1)
        loss = 1 - (pred_norm * target_norm).sum(dim=-1).mean()

        loss.backward()
        optimizer.step()
        scheduler.step()

    # 测试集评估
    model.eval()
    with torch.no_grad():
        test_en = data['test']['en_1024'].to(device)
        test_es = data['test']['es_101'].to(device)
        output, _ = model(test_en)
        test_cos = compute_cosine_similarity(output, test_es)

    return test_cos


# ============================================================================
# 主实验
# ============================================================================
def main():
    device = 'cpu'  # 使用CPU
    print("=" * 70)
    print("消融实验：聚合函数 vs Dropout")
    print("=" * 70)
    print(f"Device: {device}\n")

    # 加载数据
    data = load_data_official_split()

    results = {}
    retrieval_methods = {
        'softmax': RetrievalSoftmax,
        'direct_avg': RetrievalDirectAvg,
        'cosine_weighted': RetrievalCosineWeighted
    }
    dropout_values = [0.1, 0.3, 0.5]

    print("\n" + "=" * 70)
    print("第一部分：纯检索（无Fusion，无Dropout影响）")
    print("=" * 70)

    for method_name, retrieval_cls in retrieval_methods.items():
        retrieval = retrieval_cls()
        retrieval.set_database(
            data['train']['en_1024'].cpu().numpy(),
            data['train']['es_1024'].cpu().numpy()
        )
        test_cos = evaluate_retrieval(retrieval, data, device)
        results[f'{method_name}_pure'] = test_cos
        print(f"  {method_name:<20} Test Cosine: {test_cos:.4f}")

    print("\n" + "=" * 70)
    print("第二部分：检索+Fusion（测试不同Dropout）")
    print("=" * 70)

    for method_name, retrieval_cls in retrieval_methods.items():
        print(f"\n--- {method_name} + Fusion ---")
        for dropout in dropout_values:
            test_cos = train_and_evaluate_fusion(data, retrieval_cls, dropout, device, epochs=50)
            results[f'{method_name}_fusion_do{dropout}'] = test_cos
            print(f"  Dropout={dropout}  Test Cosine: {test_cos:.4f}")

    # 结果汇总
    print("\n" + "=" * 70)
    print("最终结果汇总")
    print("=" * 70)
    print(f"\n{'方法':<35} {'测试集余弦相似度':<15}")
    print("-" * 50)

    # 按类别分组显示
    print("\n【纯检索】")
    for method in retrieval_methods.keys():
        key = f'{method}_pure'
        print(f"  {method:<35} {results[key]:<15.4f}")

    print("\n【检索 + Fusion】")
    header = "聚合方法\\Dropout"
    print(f"{header:<20} {'0.1':<10} {'0.3':<10} {'0.5':<10}")
    print("-" * 50)
    for method in retrieval_methods.keys():
        row = f"  {method:<20}"
        for dropout in dropout_values:
            key = f'{method}_fusion_do{dropout}'
            row += f" {results[key]:<10.4f}"
        print(row)

    # 找出最佳配置
    best_method = max(results, key=results.get)
    best_score = results[best_method]

    print("\n" + "=" * 70)
    print(f"最佳配置: {best_method}")
    print(f"最佳分数: {best_score:.4f}")
    print("=" * 70)

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)

    json_file = os.path.join(results_dir, f'ablation_aggregation_dropout_{timestamp}.json')
    save_data = {
        'timestamp': timestamp,
        'device': device,
        'data_split': 'official_filelist',
        'results': results,
        'best_method': best_method,
        'best_score': best_score
    }

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到: {json_file}")


if __name__ == '__main__':
    main()

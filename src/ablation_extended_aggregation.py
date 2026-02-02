"""
扩展聚合函数实验：测试更多聚合方法

测试方法：
1. Softmax (baseline)
2. Sparsemax (稀疏权重)
3. Top-K截断 (只取Top-M)
4. 指数加权 (exp(sim), 不归一化)
5. 硬Top-1 (直接取最相似的)
6. Gumbel-Softmax (可微分硬选择)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.feature_selector import FeatureSelector


# ============================================================================
# Sparsemax实现
# ============================================================================
def sparsemax(logits, dim=-1):
    """
    Sparsemax激活函数：产生稀疏的概率分布

    From: "From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification"
    """
    sorted_logits, _ = torch.sort(logits, descending=True, dim=dim)

    # 找到截断点k
    cumsum = torch.cumsum(sorted_logits, dim=dim)
    k = torch.arange(1, logits.size(dim) + 1, device=logits.device).float()
    support = (k * sorted_logits) > (cumsum - 1)

    # 计算阈值
    k_z = support.sum(dim=dim, keepdim=True).float()
    tau = (cumsum.gather(dim, (k_z - 1).long()) - 1) / k_z

    # 计算sparsemax
    p = torch.clamp(logits - tau, min=0)

    # 归一化
    p = p / p.sum(dim=dim, keepdim=True)

    return p


# ============================================================================
# 不同的检索聚合方法
# ============================================================================
class RetrievalSoftmax(nn.Module):
    """基准: Softmax加权"""
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

        attn_weights = F.softmax(topk_sims / self.temperature, dim=1)
        topk_es = es_db[topk_indices]
        retrieved = torch.bmm(attn_weights.unsqueeze(1), topk_es).squeeze(1)

        return retrieved


class RetrievalSparsemax(nn.Module):
    """Sparsemax: 稀疏权重"""
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

        # Sparsemax产生稀疏权重
        attn_weights = sparsemax(topk_sims / self.temperature, dim=1)
        topk_es = es_db[topk_indices]
        retrieved = torch.bmm(attn_weights.unsqueeze(1), topk_es).squeeze(1)

        return retrieved


class RetrievalTopTruncated(nn.Module):
    """Top-M截断: 只用Top-M个，M < K"""
    def __init__(self, top_k=70, top_m=20):
        super().__init__()
        self.top_k = top_k
        self.top_m = min(top_m, top_k)
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

        # 只用Top-M个
        sims_m = topk_sims[:, :self.top_m]
        indices_m = topk_indices[:, :self.top_m]
        es_m = es_db[indices_m]

        # Softmax加权（只对Top-M）
        attn_weights = F.softmax(sims_m / 0.04, dim=1)
        retrieved = torch.bmm(attn_weights.unsqueeze(1), es_m).squeeze(1)

        return retrieved


class RetrievalExponential(nn.Module):
    """指数加权: exp(sim), 不做softmax归一化"""
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

        # 指数加权，不归一化（更锐化）
        attn_weights = torch.exp(topk_sims / self.temperature)
        attn_weights = attn_weights / attn_weights.sum(dim=1, keepdim=True)
        topk_es = es_db[topk_indices]
        retrieved = torch.bmm(attn_weights.unsqueeze(1), topk_es).squeeze(1)

        return retrieved


class RetrievalHardTop1(nn.Module):
    """硬Top-1: 直接取最相似的一个"""
    def __init__(self):
        super().__init__()
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

        # 只取Top-1
        top1_indices = torch.argmax(similarities, dim=1)
        retrieved = es_db[top1_indices]

        return retrieved


class RetrievalGumbelSoftmax(nn.Module):
    """Gumbel-Softmax: 可微分的硬选择"""
    def __init__(self, top_k=70, temperature=0.1):
        super().__init__()
        self.top_k = top_k
        self.temperature = temperature  # Gumbel温度，越小越接近hard
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

        # Gumbel-Softmax采样
        gumbels = -torch.log(-torch.log(torch.rand_like(topk_sims) + 1e-10) + 1e-10)
        logits = (topk_sims + gumbels) / self.temperature
        attn_weights = F.softmax(logits, dim=1)

        topk_es = es_db[topk_indices]
        retrieved = torch.bmm(attn_weights.unsqueeze(1), topk_es).squeeze(1)

        return retrieved


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
    es_test_101 = es_selector.transform(es_test_1024)

    spanish_indices = torch.from_numpy(es_selector.selected_indices).long()

    return {
        'train': {
            'en_1024': torch.tensor(en_train_1024).float(),
            'es_1024': torch.tensor(es_train_1024).float(),
        },
        'test': {
            'en_1024': torch.tensor(en_test_1024).float(),
            'es_101': torch.tensor(es_test_101).float()
        },
        'spanish_indices': spanish_indices
    }


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


# ============================================================================
# 主实验
# ============================================================================
def main():
    device = 'cpu'
    print("=" * 70)
    print("扩展聚合函数实验")
    print("=" * 70)
    print(f"Device: {device}\n")

    # 加载数据
    data = load_data_official_split()

    # 定义所有聚合方法
    retrieval_methods = {
        'softmax': RetrievalSoftmax(top_k=70, temperature=0.04),
        'sparsemax': RetrievalSparsemax(top_k=70, temperature=0.04),
        'top20_truncated': RetrievalTopTruncated(top_k=70, top_m=20),
        'exponential': RetrievalExponential(top_k=70, temperature=0.04),
        'hard_top1': RetrievalHardTop1(),
        'gumbel_softmax': RetrievalGumbelSoftmax(top_k=70, temperature=0.1),
    }

    results = {}

    print("\n" + "=" * 70)
    print("测试所有聚合方法")
    print("=" * 70)

    for method_name, retrieval in retrieval_methods.items():
        print(f"\n--- {method_name} ---")
        retrieval.set_database(
            data['train']['en_1024'].cpu().numpy(),
            data['train']['es_1024'].cpu().numpy()
        )
        test_cos = evaluate_retrieval(retrieval, data, device)
        results[method_name] = test_cos
        print(f"  Test Cosine: {test_cos:.4f}")

    # 结果汇总
    print("\n" + "=" * 70)
    print("最终结果汇总")
    print("=" * 70)
    print(f"\n{'聚合方法':<25} {'测试集余弦相似度':<15} {'相对Softmax':<15}")
    print("-" * 55)

    baseline = results['softmax']
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    for method_name, score in sorted_results:
        diff = score - baseline
        marker = " [BEST]" if method_name == sorted_results[0][0] else ""
        print(f"  {method_name:<25} {score:<15.4f} {diff:+.4f}        {marker}")

    print("-" * 55)

    best_method = sorted_results[0][0]
    best_score = sorted_results[0][1]

    print("\n" + "=" * 70)
    print(f"最佳方法: {best_method}")
    print(f"最佳分数: {best_score:.4f}")
    print(f"相比Softmax: {best_score - baseline:+.4f}")
    print("=" * 70)

    # 方法说明
    print("\n" + "=" * 70)
    print("方法说明")
    print("=" * 70)
    descriptions = {
        'softmax': 'Softmax加权 + 温度缩放 (T=0.04)',
        'sparsemax': 'Sparsemax稀疏权重 + 温度缩放',
        'top20_truncated': 'Top-70中只用Top-20',
        'exponential': '指数加权 exp(sim/T)',
        'hard_top1': '硬选择：只用最相似的1个',
        'gumbel_softmax': 'Gumbel-Softmax随机采样',
    }
    for method, desc in descriptions.items():
        print(f"  {method:<25} {desc}")

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)

    json_file = os.path.join(results_dir, f'extended_aggregation_{timestamp}.json')
    save_data = {
        'timestamp': timestamp,
        'device': device,
        'results': results,
        'best_method': best_method,
        'best_score': best_score,
        'descriptions': descriptions
    }

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到: {json_file}")


if __name__ == '__main__':
    main()

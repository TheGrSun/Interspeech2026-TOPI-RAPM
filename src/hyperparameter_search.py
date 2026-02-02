"""
超参数搜索实验：为每个聚合函数找到其最优超参数

搜索所有6个聚合函数：
1. Softmax
2. Sparsemax
3. Top-Truncated
4. Exponential
5. Hard Top-1
6. Gumbel-Softmax

使用5折交叉验证确保稳健性，最终在测试集上评估。
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
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.feature_selector import FeatureSelector


# ============================================================================
# 从ablation_extended_aggregation.py复用聚合函数
# ============================================================================
def sparsemax(logits, dim=-1):
    """Sparsemax激活函数"""
    sorted_logits, _ = torch.sort(logits, descending=True, dim=dim)
    cumsum = torch.cumsum(sorted_logits, dim=dim)
    k = torch.arange(1, logits.size(dim) + 1, device=logits.device).float()
    support = (k * sorted_logits) > (cumsum - 1)
    k_z = support.sum(dim=dim, keepdim=True).float()
    tau = (cumsum.gather(dim, (k_z - 1).long()) - 1) / k_z
    p = torch.clamp(logits - tau, min=0)
    p = p / p.sum(dim=dim, keepdim=True)
    return p


class RetrievalSoftmax(nn.Module):
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

        attn_weights = sparsemax(topk_sims / self.temperature, dim=1)
        topk_es = es_db[topk_indices]
        retrieved = torch.bmm(attn_weights.unsqueeze(1), topk_es).squeeze(1)

        return retrieved


class RetrievalTopTruncated(nn.Module):
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

        sims_m = topk_sims[:, :self.top_m]
        indices_m = topk_indices[:, :self.top_m]
        es_m = es_db[indices_m]

        attn_weights = F.softmax(sims_m / 0.04, dim=1)
        retrieved = torch.bmm(attn_weights.unsqueeze(1), es_m).squeeze(1)

        return retrieved


class RetrievalExponential(nn.Module):
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

        attn_weights = torch.exp(topk_sims / self.temperature)
        attn_weights = attn_weights / attn_weights.sum(dim=1, keepdim=True)
        topk_es = es_db[topk_indices]
        retrieved = torch.bmm(attn_weights.unsqueeze(1), topk_es).squeeze(1)

        return retrieved


class RetrievalHardTop1(nn.Module):
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

        top1_indices = torch.argmax(similarities, dim=1)
        retrieved = es_db[top1_indices]

        return retrieved


class RetrievalGumbelSoftmax(nn.Module):
    def __init__(self, top_k=70, temperature=0.1):
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

        gumbels = -torch.log(-torch.log(torch.rand_like(topk_sims) + 1e-10) + 1e-10)
        logits = (topk_sims + gumbels) / self.temperature
        attn_weights = F.softmax(logits, dim=1)

        topk_es = es_db[topk_indices]
        retrieved = torch.bmm(attn_weights.unsqueeze(1), topk_es).squeeze(1)

        return retrieved


# ============================================================================
# 超参数搜索空间
# ============================================================================
SEARCH_SPACES = {
    'softmax': {
        'top_k': [30, 50, 70, 90, 110],
        'temperature': [0.02, 0.04, 0.06, 0.08, 0.10]
    },
    'sparsemax': {
        'top_k': [50, 70, 90],
        'temperature': [0.04, 0.08, 0.15, 0.30]
    },
    'top_truncated': {
        'top_k': [50, 70, 90],
        'top_m': [5, 10, 15, 20, 30, 40]
    },
    'exponential': {
        'top_k': [50, 70, 90],
        'temperature': [0.02, 0.04, 0.06]
    },
    'hard_top1': {
        # Hard Top-1没有超参数
    },
    'gumbel_softmax': {
        'top_k': [50, 70, 90],
        'temperature': [0.01, 0.05, 0.1, 0.5, 1.0]
    }
}

RETRIEVAL_CLASSES = {
    'softmax': RetrievalSoftmax,
    'sparsemax': RetrievalSparsemax,
    'top_truncated': RetrievalTopTruncated,
    'exponential': RetrievalExponential,
    'hard_top1': RetrievalHardTop1,
    'gumbel_softmax': RetrievalGumbelSoftmax
}


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
            'es_101': torch.tensor(es_test_101).float()
        },
        'spanish_indices': spanish_indices
    }


# ============================================================================
# 评估函数
# ============================================================================
def compute_cosine_similarity(pred, target):
    pred_norm = F.normalize(pred, dim=-1)
    target_norm = F.normalize(target, dim=-1)
    return (pred_norm * target_norm).sum(dim=-1).mean().item()


# ============================================================================
# K折交叉验证
# ============================================================================
def create_k_folds(n_samples, k_folds=5, seed=42):
    """创建K折交叉验证的索引"""
    np.random.seed(seed)
    indices = np.random.permutation(n_samples)
    fold_size = n_samples // k_folds

    folds = []
    for i in range(k_folds):
        start_idx = i * fold_size
        end_idx = start_idx + fold_size if i < k_folds - 1 else n_samples

        val_indices = indices[start_idx:end_idx]
        train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])

        folds.append({
            'train': train_indices,
            'val': val_indices
        })

    return folds


def evaluate_config_with_cv(retrieval_cls, config, data, folds, device):
    """对单个配置执行K折交叉验证"""
    scores = []

    for fold_idx, fold in enumerate(folds):
        # 创建检索模块
        retrieval = retrieval_cls(**config)

        # 设置数据库（使用训练折）
        train_en = data['train']['en_1024'][fold['train']]
        train_es = data['train']['es_1024'][fold['train']]
        retrieval.set_database(train_en.cpu().numpy(), train_es.cpu().numpy())

        # 在验证折上评估
        retrieval.eval()
        with torch.no_grad():
            val_en = data['train']['en_1024'][fold['val']].to(device)
            val_es = data['train']['es_101'][fold['val']].to(device)
            spanish_indices = data['spanish_indices'].to(device)

            retrieved_1024 = retrieval(val_en)
            retrieved_101 = retrieved_1024[:, spanish_indices]
            score = compute_cosine_similarity(retrieved_101, val_es)
            scores.append(score)

    return np.mean(scores), np.std(scores)


# ============================================================================
# 网格搜索
# ============================================================================
def grid_search_method(method_name, data, folds, device):
    """对单个方法执行网格搜索"""
    print(f"\n{'=' * 70}")
    print(f"[Grid Search] {method_name}")
    print(f"{'=' * 70}")

    search_space = SEARCH_SPACES[method_name]
    retrieval_cls = RETRIEVAL_CLASSES[method_name]

    # 生成所有配置组合
    if not search_space:
        # 没有超参数的方法
        all_configs = [{}]
    else:
        param_names = list(search_space.keys())
        param_values = list(search_space.values())
        all_configs = []

        for combination in product(*param_values):
            config = dict(zip(param_names, combination))
            all_configs.append(config)

    print(f"Total configs: {len(all_configs)}")
    print(f"Configs per fold: {len(all_configs)} × {len(folds)} = {len(all_configs) * len(folds)} evaluations")

    results = []

    for config_idx, config in enumerate(all_configs, 1):
        # 格式化配置显示
        config_str = ", ".join([f"{k}={v}" for k, v in config.items()])
        print(f"\nConfig {config_idx}/{len(all_configs)}: {config_str}")

        # 执行交叉验证
        mean_score, std_score = evaluate_config_with_cv(
            retrieval_cls, config, data, folds, device
        )

        print(f"  CV Score: {mean_score:.4f} ± {std_score:.4f}")

        results.append({
            'config': config,
            'mean_score': mean_score,
            'std_score': std_score
        })

    # 找到最佳配置
    best_result = max(results, key=lambda x: x['mean_score'])

    print(f"\n[BEST] {method_name}")
    print(f"  Config: {best_result['config']}")
    print(f"  CV Score: {best_result['mean_score']:.4f} ± {best_result['std_score']:.4f}")

    return results, best_result


# ============================================================================
# 最终测试集评估
# ============================================================================
def final_test_evaluation(best_configs, data, device):
    """在测试集上评估每个方法的最佳配置"""
    print(f"\n{'=' * 70}")
    print(f"[Final Test Evaluation]")
    print(f"{'=' * 70}\n")

    test_results = {}

    for method_name, config_info in best_configs.items():
        config = config_info['config']
        retrieval_cls = RETRIEVAL_CLASSES[method_name]

        # 使用全部训练数据
        retrieval = retrieval_cls(**config)
        retrieval.set_database(
            data['train']['en_1024'].cpu().numpy(),
            data['train']['es_1024'].cpu().numpy()
        )

        # 测试集评估
        retrieval.eval()
        with torch.no_grad():
            test_en = data['test']['en_1024'].to(device)
            test_es = data['test']['es_101'].to(device)
            spanish_indices = data['spanish_indices'].to(device)

            retrieved_1024 = retrieval(test_en)
            retrieved_101 = retrieved_1024[:, spanish_indices]
            test_score = compute_cosine_similarity(retrieved_101, test_es)

        test_results[method_name] = {
            'config': config,
            'test_score': test_score,
            'cv_score': config_info['cv_score']
        }

        print(f"{method_name:<20} Test: {test_score:.4f}  (CV: {config_info['cv_score']:.4f})")

    return test_results


# ============================================================================
# 主函数
# ============================================================================
def main():
    device = 'cpu'
    print("=" * 70)
    print("超参数搜索实验 - 为每个聚合函数找到最优配置")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 加载数据
    data = load_data_official_split()

    # 创建5折交叉验证划分
    n_train = len(data['train']['en_1024'])
    folds = create_k_folds(n_train, k_folds=5, seed=42)
    print(f"\n5-Fold CV: {n_train} training samples")
    print(f"  Average fold size: {n_train // 5} samples per validation fold")

    # 存储所有结果
    all_results = {}
    best_configs = {}

    # 对每个方法执行网格搜索
    for method_name in SEARCH_SPACES.keys():
        results, best_result = grid_search_method(method_name, data, folds, device)

        all_results[method_name] = results
        best_configs[method_name] = {
            'config': best_result['config'],
            'cv_score': best_result['mean_score'],
            'cv_std': best_result['std_score']
        }

    # 最终测试集评估
    test_results = final_test_evaluation(best_configs, data, device)

    # 生成排名
    ranked = sorted(
        test_results.items(),
        key=lambda x: x[1]['test_score'],
        reverse=True
    )

    print(f"\n{'=' * 70}")
    print("【最终排名 - 基于测试集分数】")
    print(f"{'=' * 70}\n")

    print(f"{'排名':<5} {'方法':<20} {'测试分数':<12} {'CV分数':<12} {'配置':<30}")
    print("-" * 85)

    for rank, (method_name, result) in enumerate(ranked, 1):
        config_str = str(result['config'])
        if len(config_str) > 28:
            config_str = config_str[:28] + ".."

        marker = " [BEST]" if rank == 1 else ""
        print(f"{rank:<5} {method_name:<20} {result['test_score']:<12.4f} {result['cv_score']:<12.4f} {config_str:<30}{marker}")

    print("-" * 85)

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)

    json_file = os.path.join(results_dir, f'hypersearch_fair_comparison_{timestamp}.json')

    save_data = {
        'timestamp': timestamp,
        'device': device,
        'search_method': 'grid_search_5fold_cv',
        'total_experiments': sum(len(r) for r in all_results.values()) * 5,
        'search_spaces': SEARCH_SPACES,
        'all_results': all_results,
        'best_configs': best_configs,
        'test_results': test_results,
        'ranking': [{'method': m, 'test_score': r['test_score'], 'config': r['config']} for m, r in ranked]
    }

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到: {json_file}")
    print("\n" + "=" * 70)
    print("实验完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()

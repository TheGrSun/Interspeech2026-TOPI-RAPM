"""
完整对比实验：使用官方filelist划分，对比所有5种方法

1. 官方MLP (1024 -> 101)
2. 1024维纯检索
3. 1024维 + Fusion
4. 103维纯检索
5. 103维 + Fusion

使用官方filelist划分：
- 训练集: 2315对
- 测试集: 577对
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
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.feature_selector import FeatureSelector

# ============================================================================
# 官方MLP模型
# ============================================================================
class MLP(nn.Module):
    """官方MLP模型: 1024 -> 500 -> 250 -> 125 -> 101"""
    def __init__(self):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, 500),
            nn.ReLU(),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Linear(250, 125),
            nn.ReLU(),
            nn.Linear(125, 101)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================================
# 检索模型
# ============================================================================
class SimpleRetrieval(nn.Module):
    """Top-K检索模块"""
    def __init__(self, top_k=70, temperature=0.04):
        super().__init__()
        self.top_k = top_k
        self.temperature = temperature
        self.en_db = None
        self.es_db = None

    def set_database(self, en_features, es_features):
        if isinstance(en_features, np.ndarray):
            en_features = torch.tensor(en_features).float()
        if isinstance(es_features, np.ndarray):
            es_features = torch.tensor(es_features).float()
        self.en_db = en_features
        self.es_db = es_features

    def forward(self, query):
        if self.en_db is None:
            raise ValueError("Database not set")

        device = query.device
        en_db = self.en_db.to(device)
        es_db = self.es_db.to(device)

        # 余弦相似度
        query_norm = F.normalize(query, dim=-1)
        db_norm = F.normalize(en_db, dim=-1)
        similarities = torch.matmul(query_norm, db_norm.T)

        # Top-K
        k = min(self.top_k, en_db.shape[0])
        topk_sims, topk_indices = torch.topk(similarities, k=k, dim=1)

        # 加权聚合
        attn_weights = F.softmax(topk_sims / self.temperature, dim=1)
        topk_es = es_db[topk_indices]
        retrieved = torch.bmm(attn_weights.unsqueeze(1), topk_es).squeeze(1)

        return retrieved


class FusionNetwork(nn.Module):
    """融合网络"""
    def __init__(self, en_dim=1024, es_dim=101):
        super().__init__()
        input_dim = en_dim + es_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
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
    """检索 + 融合模型"""
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
# 数据加载（使用官方filelist）
# ============================================================================
def load_filelist(file_path):
    """加载filelist"""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def load_features_from_filelist(filelist, feature_dir):
    """根据filelist加载特征"""
    features = []
    for filename in tqdm(filelist, desc="Loading features"):
        npy_name = filename.replace('.wav', '_features.npy')
        feature_path = os.path.join(feature_dir, npy_name)
        if os.path.exists(feature_path):
            features.append(np.load(feature_path))
        else:
            print(f"Warning: {feature_path} not found")
    return np.array(features)


def load_data_official_split(cfg):
    """使用官方filelist加载数据"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    feature_dir = os.path.join(project_root, "dral-features/features")
    filelists_dir = os.path.join(project_root, "official_mdekorte/data/filelists")

    print("[1/5] Loading official filelists...")
    en_train_files = load_filelist(os.path.join(filelists_dir, "train_filelist_en.txt"))
    es_train_files = load_filelist(os.path.join(filelists_dir, "train_filelist_es.txt"))
    en_test_files = load_filelist(os.path.join(filelists_dir, "test_filelist_en.txt"))
    es_test_files = load_filelist(os.path.join(filelists_dir, "test_filelist_es.txt"))

    print(f"  Train: {len(en_train_files)} samples")
    print(f"  Test:  {len(en_test_files)} samples")

    print("[2/5] Loading 1024-dim features...")
    en_train_1024 = load_features_from_filelist(en_train_files, feature_dir)
    es_train_1024 = load_features_from_filelist(es_train_files, feature_dir)
    en_test_1024 = load_features_from_filelist(en_test_files, feature_dir)
    es_test_1024 = load_features_from_filelist(es_test_files, feature_dir)

    print(f"  Loaded shape: EN_train {en_train_1024.shape}, ES_train {es_train_1024.shape}")
    print(f"               EN_test  {en_test_1024.shape}, ES_test  {es_test_1024.shape}")

    print("[3/5] Extracting 103-dim English features (english_winners)...")
    en_selector = FeatureSelector.from_official('english')
    en_train_103 = en_selector.transform(en_train_1024)
    en_test_103 = en_selector.transform(en_test_1024)
    print(f"  EN: 1024 -> 103 dims")

    print("[4/5] Extracting 101-dim Spanish features (spanish_winners)...")
    es_selector = FeatureSelector.from_official('spanish')
    es_train_101 = es_selector.transform(es_train_1024)
    es_test_101 = es_selector.transform(es_test_1024)
    print(f"  ES: 1024 -> 101 dims")

    print("[5/5] Preparing data...")
    spanish_indices = torch.from_numpy(es_selector.selected_indices).long()

    return {
        'train': {
            'en_1024': torch.tensor(en_train_1024).float(),
            'en_103': torch.tensor(en_train_103).float(),
            'es_1024': torch.tensor(es_train_1024).float(),
            'es_101': torch.tensor(es_train_101).float()
        },
        'test': {
            'en_1024': torch.tensor(en_test_1024).float(),
            'en_103': torch.tensor(en_test_103).float(),
            'es_1024': torch.tensor(es_test_1024).float(),
            'es_101': torch.tensor(es_test_101).float()
        },
        'spanish_indices': spanish_indices
    }


# ============================================================================
# 评估和训练函数
# ============================================================================
def compute_cosine_similarity(pred, target):
    """计算余弦相似度"""
    pred_norm = F.normalize(pred, dim=-1)
    target_norm = F.normalize(target, dim=-1)
    return (pred_norm * target_norm).sum(dim=-1).mean().item()


# ============================================================================
# 实验函数
# ============================================================================
def run_official_mlp(data, device, epochs=500):
    """实验1: 官方MLP"""
    print("\n" + "=" * 70)
    print("Method 1: Official MLP (1024 -> 500 -> 250 -> 125 -> 101)")
    print("=" * 70)

    model = MLP().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    # 训练集split 10%作为验证集
    train_data = data['train']
    val_size = len(train_data['en_1024']) // 10
    indices = torch.randperm(len(train_data['en_1024']))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    train_en = train_data['en_1024'][train_idx].to(device)
    train_es = train_data['es_101'][train_idx].to(device)
    val_en = train_data['en_1024'][val_idx].to(device)
    val_es = train_data['es_101'][val_idx].to(device)

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(train_en)
        loss = criterion(pred, train_es)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(val_en)
                val_loss = criterion(val_pred, val_es).item()
            print(f"  Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.6f}, Val Loss: {val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # 加载最佳模型
    model.load_state_dict(best_state)

    # 测试集评估
    model.eval()
    with torch.no_grad():
        test_en = data['test']['en_1024'].to(device)
        test_es = data['test']['es_101'].to(device)
        pred = model(test_en)
        test_cos = compute_cosine_similarity(pred, test_es)

    print(f"  Test Cosine: {test_cos:.4f}")
    return test_cos


def run_pure_retrieval_1024(data, device, top_k=70, temperature=0.04):
    """实验2: 1024维纯检索"""
    print("\n" + "=" * 70)
    print("Method 2: 1024-dim Pure Retrieval")
    print("=" * 70)

    retrieval = SimpleRetrieval(top_k=top_k, temperature=temperature)
    retrieval.set_database(
        data['train']['en_1024'].cpu().numpy(),
        data['train']['es_1024'].cpu().numpy()
    )

    retrieval.eval()
    with torch.no_grad():
        test_en = data['test']['en_1024'].to(device)
        test_es = data['test']['es_101'].to(device)
        spanish_indices = data['spanish_indices'].to(device)

        retrieved_1024 = retrieval(test_en)
        retrieved_101 = retrieved_1024[:, spanish_indices]
        test_cos = compute_cosine_similarity(retrieved_101, test_es)

    print(f"  Test Cosine: {test_cos:.4f}")
    return test_cos


def run_fusion_1024(data, device, top_k=70, temperature=0.04, epochs=100):
    """实验3: 1024维 + Fusion"""
    print("\n" + "=" * 70)
    print("Method 3: 1024-dim + Fusion")
    print("=" * 70)

    retrieval = SimpleRetrieval(top_k=top_k, temperature=temperature)
    retrieval.set_database(
        data['train']['en_1024'].cpu().numpy(),
        data['train']['es_1024'].cpu().numpy()
    )

    fusion = FusionNetwork(en_dim=1024, es_dim=101)
    model = RetrievalFusionModel(retrieval, fusion, data['spanish_indices']).to(device)

    # 只训练fusion
    optimizer = optim.AdamW(model.fusion.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_cos = 0
    best_state = None

    # 划分验证集
    val_size = len(data['train']['en_1024']) // 10
    indices = torch.randperm(len(data['train']['en_1024']))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    train_en = data['train']['en_1024'][train_idx].to(device)
    train_es = data['train']['es_101'][train_idx].to(device)
    val_en = data['train']['en_1024'][val_idx].to(device)
    val_es = data['train']['es_101'][val_idx].to(device)

    print("  Training Fusion network...")
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

        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                val_output, _ = model(val_en)
                val_cos = compute_cosine_similarity(val_output, val_es)
            print(f"    Epoch {epoch+1}/{epochs}, Val Cosine: {val_cos:.4f}")

            if val_cos > best_val_cos:
                best_val_cos = val_cos
                best_state = {k: v.cpu().clone() for k, v in model.fusion.state_dict().items()}

    # 加载最佳模型
    model.fusion.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_en = data['test']['en_1024'].to(device)
        test_es = data['test']['es_101'].to(device)
        output, _ = model(test_en)
        test_cos = compute_cosine_similarity(output, test_es)

    print(f"  Test Cosine: {test_cos:.4f}")
    return test_cos


def run_pure_retrieval_103(data, device, top_k=70, temperature=0.04):
    """实验4: 103维纯检索"""
    print("\n" + "=" * 70)
    print("Method 4: 103-dim Pure Retrieval")
    print("=" * 70)

    retrieval = SimpleRetrieval(top_k=top_k, temperature=temperature)
    retrieval.set_database(
        data['train']['en_103'].cpu().numpy(),
        data['train']['es_1024'].cpu().numpy()  # 注意：ES仍是1024维
    )

    retrieval.eval()
    with torch.no_grad():
        test_en = data['test']['en_103'].to(device)
        test_es = data['test']['es_101'].to(device)
        spanish_indices = data['spanish_indices'].to(device)

        retrieved_1024 = retrieval(test_en)  # 返回1024维
        retrieved_101 = retrieved_1024[:, spanish_indices]
        test_cos = compute_cosine_similarity(retrieved_101, test_es)

    print(f"  Test Cosine: {test_cos:.4f}")
    return test_cos


def run_fusion_103(data, device, top_k=70, temperature=0.04, epochs=100):
    """实验5: 103维 + Fusion"""
    print("\n" + "=" * 70)
    print("Method 5: 103-dim + Fusion")
    print("=" * 70)

    retrieval = SimpleRetrieval(top_k=top_k, temperature=temperature)
    retrieval.set_database(
        data['train']['en_103'].cpu().numpy(),
        data['train']['es_1024'].cpu().numpy()
    )

    fusion = FusionNetwork(en_dim=103, es_dim=101)  # 注意：en_dim=103
    model = RetrievalFusionModel(retrieval, fusion, data['spanish_indices']).to(device)

    optimizer = optim.AdamW(model.fusion.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_cos = 0
    best_state = None

    val_size = len(data['train']['en_103']) // 10
    indices = torch.randperm(len(data['train']['en_103']))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    train_en = data['train']['en_103'][train_idx].to(device)
    train_es = data['train']['es_101'][train_idx].to(device)
    val_en = data['train']['en_103'][val_idx].to(device)
    val_es = data['train']['es_101'][val_idx].to(device)

    print("  Training Fusion network...")
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

        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                val_output, _ = model(val_en)
                val_cos = compute_cosine_similarity(val_output, val_es)
            print(f"    Epoch {epoch+1}/{epochs}, Val Cosine: {val_cos:.4f}")

            if val_cos > best_val_cos:
                best_val_cos = val_cos
                best_state = {k: v.cpu().clone() for k, v in model.fusion.state_dict().items()}

    model.fusion.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_en = data['test']['en_103'].to(device)
        test_es = data['test']['es_101'].to(device)
        output, _ = model(test_en)
        test_cos = compute_cosine_similarity(output, test_es)

    print(f"  Test Cosine: {test_cos:.4f}")
    return test_cos


# ============================================================================
# 主函数
# ============================================================================
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 70)
    print("完整对比实验：使用官方filelist划分")
    print("=" * 70)
    print(f"Device: {device}\n")

    # 加载数据
    data = load_data_official_split(None)

    results = {}

    # 运行所有实验
    results['official_mlp'] = run_official_mlp(data, device, epochs=500)
    results['1024_pure'] = run_pure_retrieval_1024(data, device)
    results['1024_fusion'] = run_fusion_1024(data, device, epochs=100)
    results['103_pure'] = run_pure_retrieval_103(data, device)
    results['103_fusion'] = run_fusion_103(data, device, epochs=100)

    # 结果汇总
    print("\n" + "=" * 70)
    print("最终结果汇总")
    print("=" * 70)
    print(f"\n{'方法':<30} {'测试集余弦相似度':<20}")
    print("-" * 50)
    print(f"{'Official MLP (1024->101)':<30} {results['official_mlp']:<20.4f}")
    print(f"{'1024维 纯检索':<30} {results['1024_pure']:<20.4f}")
    print(f"{'1024维 + Fusion':<30} {results['1024_fusion']:<20.4f}")
    print(f"{'103维 纯检索':<30} {results['103_pure']:<20.4f}")
    print(f"{'103维 + Fusion':<30} {results['103_fusion']:<20.4f}")
    print("-" * 50)

    # 对比分析
    print("\n对比分析：")
    print(f"  纯检索: 103维 vs 1024维 = {results['103_pure'] - results['1024_pure']:+.4f}")
    print(f"  Fusion:  103维 vs 1024维 = {results['103_fusion'] - results['1024_fusion']:+.4f}")
    print(f"  1024维: Fusion vs 纯检索 = {results['1024_fusion'] - results['1024_pure']:+.4f}")
    print(f"  103维:  Fusion vs 纯检索 = {results['103_fusion'] - results['103_pure']:+.4f}")
    print(f"  最佳方法 vs MLP = {max(results.values()) - results['official_mlp']:+.4f}")

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)

    json_file = os.path.join(results_dir, f'comparison_official_split_{timestamp}.json')
    save_data = {
        'timestamp': timestamp,
        'device': device,
        'data_split': 'official_filelist',
        'train_samples': len(data['train']['en_1024']),
        'test_samples': len(data['test']['en_1024']),
        'results': results,
        'best_method': max(results, key=results.get),
        'best_score': max(results.values())
    }

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到: {json_file}")


if __name__ == '__main__':
    main()

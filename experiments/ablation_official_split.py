"""
消融实验：103维检索 vs 1024维检索 vs 官方MLP
使用官方的数据划分进行公平对比

数据划分：
- 训练集：2315个样本（官方train_filelist）
- 测试集：577个样本（官方test_filelist）
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
import json
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.feature_selector import FeatureSelector


# ============================================================================
# 模型定义
# ============================================================================

class SimpleRetrieval(nn.Module):
    """简单的Top-K检索模块"""
    def __init__(self, top_k=70, temperature=0.05):
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

        query_norm = F.normalize(query, dim=-1)
        db_norm = F.normalize(en_db, dim=-1)
        similarities = torch.matmul(query_norm, db_norm.T)

        k = min(self.top_k, en_db.shape[0])
        topk_sims, topk_indices = torch.topk(similarities, k=k, dim=1)

        attn_weights = F.softmax(topk_sims / self.temperature, dim=1)
        topk_es = es_db[topk_indices]
        retrieved = torch.bmm(attn_weights.unsqueeze(1), topk_es).squeeze(1)

        return retrieved


class FusionNetwork(nn.Module):
    """融合网络（与系统描述一致）"""
    def __init__(self, en_dim=1024, es_dim=101, hidden_dims=[256, 128]):
        super().__init__()

        input_dim = en_dim + es_dim
        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                # 无Dropout（系统描述：移除dropout效果更好）
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, es_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, en_features, retrieved_es):
        combined = torch.cat([en_features, retrieved_es], dim=-1)
        delta = self.net(combined)
        # 简单残差连接（系统描述：ES_retrieved + Delta）
        output = retrieved_es + delta
        return output


class OfficialMLP(nn.Module):
    """官方MLP模型：1024 -> 500 -> 250 -> 125 -> 101"""
    def __init__(self):
        super().__init__()
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
# 数据加载（使用官方划分）
# ============================================================================

def load_official_split():
    """加载官方的数据划分"""
    official_dir = 'E:/interspeech2026/official_mdekorte/data/filelists'

    with open(os.path.join(official_dir, 'train_filelist_en.txt'), 'r') as f:
        train_files = [line.strip().replace('.wav', '_features.npy').replace('EN_', '') for line in f]

    with open(os.path.join(official_dir, 'test_filelist_en.txt'), 'r') as f:
        test_files = [line.strip().replace('.wav', '_features.npy').replace('EN_', '') for line in f]

    return train_files, test_files


def load_data_official():
    """使用官方划分加载数据"""
    data_dir = 'E:/interspeech2026/dral-features/features'

    print("[1/4] Loading official data split...")
    train_files, test_files = load_official_split()
    print(f"  Train: {len(train_files)}, Test: {len(test_files)}")

    print("[2/4] Loading 1024-dim features...")
    # 加载训练集
    en_1024_train = []
    es_1024_train = []
    for f in tqdm(train_files, desc="  Train"):
        en_path = os.path.join(data_dir, f'EN_{f}')
        es_path = os.path.join(data_dir, f'ES_{f}')
        if os.path.exists(en_path) and os.path.exists(es_path):
            en_1024_train.append(np.load(en_path))
            es_1024_train.append(np.load(es_path))

    # 加载测试集
    en_1024_test = []
    es_1024_test = []
    for f in tqdm(test_files, desc="  Test"):
        en_path = os.path.join(data_dir, f'EN_{f}')
        es_path = os.path.join(data_dir, f'ES_{f}')
        if os.path.exists(en_path) and os.path.exists(es_path):
            en_1024_test.append(np.load(en_path))
            es_1024_test.append(np.load(es_path))

    en_1024_train = np.array(en_1024_train)
    es_1024_train = np.array(es_1024_train)
    en_1024_test = np.array(en_1024_test)
    es_1024_test = np.array(es_1024_test)

    print(f"  Loaded: Train={len(en_1024_train)}, Test={len(en_1024_test)}")

    print("[3/4] Extracting 103-dim English features...")
    en_selector = FeatureSelector.from_official('english')
    en_103_train = en_selector.transform(en_1024_train)
    en_103_test = en_selector.transform(en_1024_test)

    print("[4/4] Extracting 101-dim Spanish features...")
    es_selector = FeatureSelector.from_official('spanish')
    es_101_train = es_selector.transform(es_1024_train)
    es_101_test = es_selector.transform(es_1024_test)

    return {
        'train': {
            'en_1024': en_1024_train,
            'en_103': en_103_train,
            'es_1024': es_1024_train,
            'es_101': es_101_train
        },
        'test': {
            'en_1024': en_1024_test,
            'en_103': en_103_test,
            'es_1024': es_1024_test,
            'es_101': es_101_test
        },
        'spanish_indices': es_selector.selected_indices
    }


# ============================================================================
# 评估和训练函数
# ============================================================================

def compute_cosine_similarity(pred, target):
    pred_norm = F.normalize(pred, dim=-1)
    target_norm = F.normalize(target, dim=-1)
    return (pred_norm * target_norm).sum(dim=-1).mean().item()


def train_mlp(model, train_en, train_es, epochs=500, lr=0.001, batch_size=500, val_split=0.1):
    """训练MLP模型（带早停机制）"""
    device = train_en.device

    # 划分验证集
    n = train_en.shape[0]
    n_val = int(n * val_split)
    perm = torch.randperm(n, device=device)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_en_split = train_en[train_idx]
    train_es_split = train_es[train_idx]
    val_en = train_en[val_idx]
    val_es = train_es[val_idx]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    best_epoch = 0
    n_train = train_en_split.shape[0]

    for epoch in range(epochs):
        # Training
        model.train()
        perm = torch.randperm(n_train, device=device)
        total_loss = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            batch_en = train_en_split[idx]
            batch_es = train_es_split[idx]

            optimizer.zero_grad()
            output = model(batch_en)
            loss = criterion(output, batch_es)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_output = model(val_en)
            val_loss = criterion(val_output, val_es).item()

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1

        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}, Best: epoch {best_epoch}")

    # 恢复最佳模型
    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"    Loaded best model from epoch {best_epoch} (val_loss={best_val_loss:.4f})")
    return model, best_epoch


def train_fusion(model, retrieval, train_en, train_es_1024, train_es_101, spanish_indices,
                 epochs=100, lr=0.001):
    """训练Fusion网络"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # 检索
        with torch.no_grad():
            retrieved_1024 = retrieval(train_en)
            retrieved_101 = retrieved_1024[:, spanish_indices]

        # Fusion
        output = model(train_en, retrieved_101)

        # 余弦损失
        pred_norm = F.normalize(output, dim=-1)
        target_norm = F.normalize(train_es_101, dim=-1)
        loss = 1 - (pred_norm * target_norm).sum(dim=-1).mean()

        loss.backward()
        optimizer.step()

    return model


# ============================================================================
# 主实验
# ============================================================================

def run_ablation():
    """运行消融实验"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # 创建日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = 'E:/interspeech2026/results'
    os.makedirs(results_dir, exist_ok=True)
    log_file = os.path.join(results_dir, f'ablation_official_split_{timestamp}.log')
    json_file = os.path.join(results_dir, f'ablation_official_split_{timestamp}.json')

    log_lines = []
    log_lines.append("=" * 80)
    log_lines.append("Ablation Study: 103-dim vs 1024-dim vs Official MLP")
    log_lines.append("Using Official Data Split")
    log_lines.append("=" * 80)
    log_lines.append(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_lines.append(f"Device: {device}")
    log_lines.append("")

    # 加载数据
    data = load_data_official()

    log_lines.append(f"[Data Info - Official Split]")
    log_lines.append(f"  Train: {len(data['train']['en_1024'])}")
    log_lines.append(f"  Test: {len(data['test']['en_1024'])}")
    log_lines.append("")

    # 转换为tensor
    def to_tensor(arr):
        return torch.tensor(arr).float().to(device)

    train = {k: to_tensor(v) for k, v in data['train'].items()}
    test = {k: to_tensor(v) for k, v in data['test'].items()}
    spanish_indices = torch.from_numpy(data['spanish_indices']).long().to(device)

    results = {}
    top_k = 70
    temperature = 0.04  # 系统描述中的值

    log_lines.append("[Experiment Config]")
    log_lines.append(f"  Top-K: {top_k}")
    log_lines.append(f"  Temperature: {temperature}")
    log_lines.append("")

    # ========== 实验1: 官方MLP ==========
    log_lines.append("=" * 80)
    log_lines.append("Experiment 1: Official MLP (1024->500->250->125->101)")
    log_lines.append("=" * 80)
    print("\n" + "=" * 80)
    print("Experiment 1: Official MLP")
    print("=" * 80)

    mlp = OfficialMLP().to(device)
    print("  Training MLP (500 epochs with early stopping)...")
    mlp, best_epoch = train_mlp(mlp, train['en_1024'], train['es_101'], epochs=500)

    mlp.eval()
    with torch.no_grad():
        test_output = mlp(test['en_1024'])
        test_cos_mlp = compute_cosine_similarity(test_output, test['es_101'])

    results['official_mlp'] = {'test': test_cos_mlp, 'best_epoch': best_epoch}
    log_lines.append(f"  Best Epoch: {best_epoch}")
    log_lines.append(f"  Test Cosine: {test_cos_mlp:.4f}")
    print(f"  Best Epoch: {best_epoch}")
    print(f"  Test Cosine: {test_cos_mlp:.4f}")

    # ========== 实验2: 1024维纯检索 ==========
    log_lines.append("")
    log_lines.append("=" * 80)
    log_lines.append("Experiment 2: 1024-dim Pure Retrieval")
    log_lines.append("=" * 80)
    print("\n" + "=" * 80)
    print("Experiment 2: 1024-dim Pure Retrieval")
    print("=" * 80)

    retrieval_1024 = SimpleRetrieval(top_k=top_k, temperature=temperature)
    retrieval_1024.set_database(train['en_1024'], train['es_1024'])

    with torch.no_grad():
        retrieved = retrieval_1024(test['en_1024'])
        retrieved_101 = retrieved[:, spanish_indices]
        test_cos_1024_pure = compute_cosine_similarity(retrieved_101, test['es_101'])

    results['1024_pure'] = {'test': test_cos_1024_pure}
    log_lines.append(f"  Test Cosine: {test_cos_1024_pure:.4f}")
    print(f"  Test Cosine: {test_cos_1024_pure:.4f}")

    # ========== 实验3: 1024维 + Fusion ==========
    log_lines.append("")
    log_lines.append("=" * 80)
    log_lines.append("Experiment 3: 1024-dim + Fusion")
    log_lines.append("=" * 80)
    print("\n" + "=" * 80)
    print("Experiment 3: 1024-dim + Fusion")
    print("=" * 80)

    retrieval_1024 = SimpleRetrieval(top_k=top_k, temperature=temperature)
    retrieval_1024.set_database(train['en_1024'], train['es_1024'])
    fusion_1024 = FusionNetwork(en_dim=1024, es_dim=101).to(device)

    print("  Training Fusion...")
    fusion_1024 = train_fusion(fusion_1024, retrieval_1024, train['en_1024'],
                                train['es_1024'], train['es_101'], spanish_indices)

    fusion_1024.eval()
    with torch.no_grad():
        retrieved = retrieval_1024(test['en_1024'])
        retrieved_101 = retrieved[:, spanish_indices]
        test_output = fusion_1024(test['en_1024'], retrieved_101)
        test_cos_1024_fusion = compute_cosine_similarity(test_output, test['es_101'])

    results['1024_fusion'] = {'test': test_cos_1024_fusion}
    log_lines.append(f"  Test Cosine: {test_cos_1024_fusion:.4f}")
    print(f"  Test Cosine: {test_cos_1024_fusion:.4f}")

    # ========== 实验4: 103维纯检索 ==========
    log_lines.append("")
    log_lines.append("=" * 80)
    log_lines.append("Experiment 4: 103-dim Pure Retrieval")
    log_lines.append("=" * 80)
    print("\n" + "=" * 80)
    print("Experiment 4: 103-dim Pure Retrieval")
    print("=" * 80)

    retrieval_103 = SimpleRetrieval(top_k=top_k, temperature=temperature)
    retrieval_103.set_database(train['en_103'], train['es_1024'])

    with torch.no_grad():
        retrieved = retrieval_103(test['en_103'])
        retrieved_101 = retrieved[:, spanish_indices]
        test_cos_103_pure = compute_cosine_similarity(retrieved_101, test['es_101'])

    results['103_pure'] = {'test': test_cos_103_pure}
    log_lines.append(f"  Test Cosine: {test_cos_103_pure:.4f}")
    print(f"  Test Cosine: {test_cos_103_pure:.4f}")

    # ========== 实验5: 103维 + Fusion ==========
    log_lines.append("")
    log_lines.append("=" * 80)
    log_lines.append("Experiment 5: 103-dim + Fusion")
    log_lines.append("=" * 80)
    print("\n" + "=" * 80)
    print("Experiment 5: 103-dim + Fusion")
    print("=" * 80)

    retrieval_103 = SimpleRetrieval(top_k=top_k, temperature=temperature)
    retrieval_103.set_database(train['en_103'], train['es_1024'])
    fusion_103 = FusionNetwork(en_dim=103, es_dim=101).to(device)

    print("  Training Fusion...")
    fusion_103 = train_fusion(fusion_103, retrieval_103, train['en_103'],
                               train['es_1024'], train['es_101'], spanish_indices)

    fusion_103.eval()
    with torch.no_grad():
        retrieved = retrieval_103(test['en_103'])
        retrieved_101 = retrieved[:, spanish_indices]
        test_output = fusion_103(test['en_103'], retrieved_101)
        test_cos_103_fusion = compute_cosine_similarity(test_output, test['es_101'])

    results['103_fusion'] = {'test': test_cos_103_fusion}
    log_lines.append(f"  Test Cosine: {test_cos_103_fusion:.4f}")
    print(f"  Test Cosine: {test_cos_103_fusion:.4f}")

    # ========== 结果汇总 ==========
    log_lines.append("")
    log_lines.append("=" * 80)
    log_lines.append("SUMMARY (Official Data Split)")
    log_lines.append("=" * 80)
    log_lines.append("")
    log_lines.append(f"{'Method':<30} {'Test Cosine':<15} {'Note':<20}")
    log_lines.append("-" * 65)
    log_lines.append(f"{'Official MLP':<30} {results['official_mlp']['test']:<15.4f} {'(best epoch '+str(results['official_mlp']['best_epoch'])+')':<20}")
    log_lines.append(f"{'1024-dim Pure Retrieval':<30} {results['1024_pure']['test']:<15.4f}")
    log_lines.append(f"{'1024-dim + Fusion':<30} {results['1024_fusion']['test']:<15.4f}")
    log_lines.append(f"{'103-dim Pure Retrieval':<30} {results['103_pure']['test']:<15.4f}")
    log_lines.append(f"{'103-dim + Fusion':<30} {results['103_fusion']['test']:<15.4f}")
    log_lines.append("-" * 65)

    print("\n" + "=" * 80)
    print("SUMMARY (Official Data Split)")
    print("=" * 80)
    print(f"\n{'Method':<30} {'Test Cosine':<15} {'Note':<20}")
    print("-" * 65)
    print(f"{'Official MLP':<30} {results['official_mlp']['test']:<15.4f} {'(best epoch '+str(results['official_mlp']['best_epoch'])+')':<20}")
    print(f"{'1024-dim Pure Retrieval':<30} {results['1024_pure']['test']:<15.4f}")
    print(f"{'1024-dim + Fusion':<30} {results['1024_fusion']['test']:<15.4f}")
    print(f"{'103-dim Pure Retrieval':<30} {results['103_pure']['test']:<15.4f}")
    print(f"{'103-dim + Fusion':<30} {results['103_fusion']['test']:<15.4f}")
    print("-" * 65)

    log_lines.append("")
    log_lines.append(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_lines.append("=" * 80)

    # 保存日志
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))
    print(f"\nLog saved to: {log_file}")

    # 保存JSON
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({'timestamp': timestamp, 'results': results}, f, indent=2)
    print(f"JSON saved to: {json_file}")

    return results


if __name__ == '__main__':
    print("=" * 80)
    print("Ablation Study: 103-dim vs 1024-dim vs Official MLP")
    print("Using Official Data Split (Train: 2315, Test: 577)")
    print("=" * 80)

    run_ablation()

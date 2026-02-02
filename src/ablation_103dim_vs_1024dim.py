"""
消融实验：103维检索 vs 1024维检索
包含纯检索和Retrieval+Fusion两种模式的公平对比

实验配置：
- 1024维纯检索：EN_1024 → 在1024维空间检索 → ES_1024 → spanish_winners → ES_101
- 103维纯检索：EN_1024 → english_winners → EN_103 → 在103维空间检索 → ES_1024 → spanish_winners → ES_101
- 1024维+Fusion：纯检索 + FusionNetwork训练
- 103维+Fusion：纯检索 + FusionNetwork训练

评测指标：余弦相似度（与官方评测一致）
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
# 模型定义（从train_final_submission.py复制）
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
    """融合网络：结合EN特征和检索结果"""
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
                nn.Dropout(0.1)
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, es_dim))
        self.net = nn.Sequential(*layers)

        # 残差权重
        self.residual_weight = nn.Parameter(torch.tensor(0.8))

    def forward(self, en_features, retrieved_es):
        combined = torch.cat([en_features, retrieved_es], dim=-1)
        delta = self.net(combined)
        w = torch.sigmoid(self.residual_weight)
        output = w * retrieved_es + (1 - w) * (retrieved_es + delta)
        return output


class RetrievalFusionModel(nn.Module):
    """完整模型：检索 + 融合"""
    def __init__(self, retrieval, fusion, selector_indices):
        super().__init__()
        self.retrieval = retrieval
        self.fusion = fusion
        self.register_buffer('selector_indices', selector_indices)

    def forward(self, en_features):
        # 检索（返回1024维）
        retrieved_1024 = self.retrieval(en_features)
        # 降维到101维
        retrieved_101 = retrieved_1024[:, self.selector_indices]
        # 融合
        output = self.fusion(en_features, retrieved_101)
        return output, retrieved_101


# ============================================================================
# 数据加载
# ============================================================================

def load_data(cfg):
    """加载数据"""
    data_dir = cfg['data']['data_dir']

    print("[1/4] Loading 1024-dim features...")
    en_files = sorted([f for f in os.listdir(data_dir) if f.startswith('EN_')])
    es_files = sorted([f for f in os.listdir(data_dir) if f.startswith('ES_')])

    en_1024 = np.array([np.load(os.path.join(data_dir, f)) for f in tqdm(en_files, desc="  EN_1024")])
    es_1024 = np.array([np.load(os.path.join(data_dir, f)) for f in tqdm(es_files, desc="  ES_1024")])
    print(f"  Loaded: {len(en_1024)} EN, {len(es_1024)} ES samples")

    print("[2/4] Extracting 103-dim English features (english_winners)...")
    en_selector = FeatureSelector.from_official('english')
    en_103 = en_selector.transform(en_1024)
    print(f"  EN: 1024 -> 103 dims")

    print("[3/4] Extracting 101-dim Spanish features (spanish_winners)...")
    es_selector = FeatureSelector.from_official('spanish')
    es_101 = es_selector.transform(es_1024)
    print(f"  ES: 1024 -> 101 dims")

    print("[4/4] Splitting data (seed=42, 80/10/10)...")
    np.random.seed(cfg['data']['seed'])
    n = len(en_files)
    idx = np.random.permutation(n)
    n_train = int(n * cfg['data']['split_ratio']['train'])
    n_val = int(n * cfg['data']['split_ratio']['val'])

    splits = {
        'train': idx[:n_train],
        'val': idx[n_train:n_train+n_val],
        'test': idx[n_train+n_val:]
    }
    print(f"  Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")

    return {
        'en_1024': en_1024,
        'en_103': en_103,
        'es_1024': es_1024,
        'es_101': es_101,
        'splits': splits,
        'spanish_indices': es_selector.selected_indices
    }


# ============================================================================
# 评估函数
# ============================================================================

def compute_cosine_similarity(pred, target):
    """计算余弦相似度"""
    pred_norm = F.normalize(pred, dim=-1)
    target_norm = F.normalize(target, dim=-1)
    return (pred_norm * target_norm).sum(dim=-1).mean().item()


def evaluate_pure_retrieval(retrieval, query, es_true, spanish_indices, device):
    """评估纯检索性能"""
    retrieval.eval()
    with torch.no_grad():
        retrieved_1024 = retrieval(query)
        retrieved_101 = retrieved_1024[:, spanish_indices]
        cos_sim = compute_cosine_similarity(retrieved_101, es_true)
    return cos_sim


def train_fusion_model(model, train_data, val_data, device, epochs=100, lr=0.001):
    """训练Fusion模型"""
    optimizer = torch.optim.AdamW(model.fusion.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_cos = 0
    best_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()

        output, _ = model(train_data['en'])
        loss = 1 - compute_cosine_similarity(output, train_data['es_101'])
        loss_tensor = torch.tensor(loss, requires_grad=True)

        # 使用MSE作为可微分的代理损失
        pred_norm = F.normalize(output, dim=-1)
        target_norm = F.normalize(train_data['es_101'], dim=-1)
        loss_tensor = 1 - (pred_norm * target_norm).sum(dim=-1).mean()

        loss_tensor.backward()
        optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_output, _ = model(val_data['en'])
            val_cos = compute_cosine_similarity(val_output, val_data['es_101'])

        if val_cos > best_val_cos:
            best_val_cos = val_cos
            best_state = {k: v.clone() for k, v in model.fusion.state_dict().items()}

    # 恢复最佳模型
    if best_state is not None:
        model.fusion.load_state_dict(best_state)

    return best_val_cos


# ============================================================================
# 主实验
# ============================================================================

def run_ablation(cfg):
    """运行消融实验"""
    device = 'cuda' if torch.cuda.is_available() and cfg['training']['device'] == 'cuda' else 'cpu'
    print(f"\nDevice: {device}")

    # 创建日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)
    log_file = os.path.join(results_dir, f'ablation_103_vs_1024_{timestamp}.log')
    json_file = os.path.join(results_dir, f'ablation_103_vs_1024_{timestamp}.json')

    log_lines = []
    log_lines.append("=" * 80)
    log_lines.append("Ablation Study: 103-dim vs 1024-dim Retrieval")
    log_lines.append("=" * 80)
    log_lines.append(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_lines.append(f"Device: {device}")
    log_lines.append("")

    # 加载数据
    data = load_data(cfg)

    log_lines.append(f"[Data Info]")
    log_lines.append(f"  Total samples: {len(data['en_1024'])}")
    log_lines.append(f"  Train: {len(data['splits']['train'])}")
    log_lines.append(f"  Val: {len(data['splits']['val'])}")
    log_lines.append(f"  Test: {len(data['splits']['test'])}")
    log_lines.append("")

    # 转换为tensor
    def to_tensor(arr, idx):
        return torch.tensor(arr[idx]).float().to(device)

    splits = data['splits']
    tensors = {}
    for name, idx in splits.items():
        tensors[name] = {
            'en_1024': to_tensor(data['en_1024'], idx),
            'en_103': to_tensor(data['en_103'], idx),
            'es_1024': to_tensor(data['es_1024'], idx),
            'es_101': to_tensor(data['es_101'], idx)
        }

    spanish_indices = torch.from_numpy(data['spanish_indices']).long().to(device)

    # 实验参数
    top_k = 70
    temperature = 0.05

    log_lines.append("[Experiment Config]")
    log_lines.append(f"  Top-K: {top_k}")
    log_lines.append(f"  Temperature: {temperature}")
    log_lines.append(f"  Fusion epochs: 100")
    log_lines.append("")

    results = {}

    # ========== 实验1: 1024维纯检索 ==========
    log_lines.append("=" * 80)
    log_lines.append("Experiment 1: 1024-dim Pure Retrieval")
    log_lines.append("=" * 80)
    print("\n" + "=" * 80)
    print("Experiment 1: 1024-dim Pure Retrieval")
    print("=" * 80)

    retrieval_1024 = SimpleRetrieval(top_k=top_k, temperature=temperature)
    retrieval_1024.set_database(tensors['train']['en_1024'], tensors['train']['es_1024'])

    val_cos_1024_pure = evaluate_pure_retrieval(
        retrieval_1024, tensors['val']['en_1024'], tensors['val']['es_101'], spanish_indices, device
    )
    test_cos_1024_pure = evaluate_pure_retrieval(
        retrieval_1024, tensors['test']['en_1024'], tensors['test']['es_101'], spanish_indices, device
    )

    results['1024_pure'] = {'val': val_cos_1024_pure, 'test': test_cos_1024_pure}
    log_lines.append(f"  Val Cosine:  {val_cos_1024_pure:.4f}")
    log_lines.append(f"  Test Cosine: {test_cos_1024_pure:.4f}")
    print(f"  Val Cosine:  {val_cos_1024_pure:.4f}")
    print(f"  Test Cosine: {test_cos_1024_pure:.4f}")

    # ========== 实验2: 1024维 + Fusion ==========
    log_lines.append("")
    log_lines.append("=" * 80)
    log_lines.append("Experiment 2: 1024-dim + Fusion")
    log_lines.append("=" * 80)
    print("\n" + "=" * 80)
    print("Experiment 2: 1024-dim + Fusion")
    print("=" * 80)

    retrieval_1024 = SimpleRetrieval(top_k=top_k, temperature=temperature)
    retrieval_1024.set_database(tensors['train']['en_1024'], tensors['train']['es_1024'])
    fusion_1024 = FusionNetwork(en_dim=1024, es_dim=101)
    model_1024 = RetrievalFusionModel(retrieval_1024, fusion_1024, spanish_indices).to(device)

    print("  Training Fusion network...")
    train_data_1024 = {'en': tensors['train']['en_1024'], 'es_101': tensors['train']['es_101']}
    val_data_1024 = {'en': tensors['val']['en_1024'], 'es_101': tensors['val']['es_101']}
    train_fusion_model(model_1024, train_data_1024, val_data_1024, device)

    model_1024.eval()
    with torch.no_grad():
        val_output, _ = model_1024(tensors['val']['en_1024'])
        val_cos_1024_fusion = compute_cosine_similarity(val_output, tensors['val']['es_101'])
        test_output, _ = model_1024(tensors['test']['en_1024'])
        test_cos_1024_fusion = compute_cosine_similarity(test_output, tensors['test']['es_101'])

    results['1024_fusion'] = {'val': val_cos_1024_fusion, 'test': test_cos_1024_fusion}
    log_lines.append(f"  Val Cosine:  {val_cos_1024_fusion:.4f}")
    log_lines.append(f"  Test Cosine: {test_cos_1024_fusion:.4f}")
    print(f"  Val Cosine:  {val_cos_1024_fusion:.4f}")
    print(f"  Test Cosine: {test_cos_1024_fusion:.4f}")

    # ========== 实验3: 103维纯检索 ==========
    log_lines.append("")
    log_lines.append("=" * 80)
    log_lines.append("Experiment 3: 103-dim Pure Retrieval")
    log_lines.append("=" * 80)
    print("\n" + "=" * 80)
    print("Experiment 3: 103-dim Pure Retrieval")
    print("=" * 80)

    retrieval_103 = SimpleRetrieval(top_k=top_k, temperature=temperature)
    retrieval_103.set_database(tensors['train']['en_103'], tensors['train']['es_1024'])

    val_cos_103_pure = evaluate_pure_retrieval(
        retrieval_103, tensors['val']['en_103'], tensors['val']['es_101'], spanish_indices, device
    )
    test_cos_103_pure = evaluate_pure_retrieval(
        retrieval_103, tensors['test']['en_103'], tensors['test']['es_101'], spanish_indices, device
    )

    results['103_pure'] = {'val': val_cos_103_pure, 'test': test_cos_103_pure}
    log_lines.append(f"  Val Cosine:  {val_cos_103_pure:.4f}")
    log_lines.append(f"  Test Cosine: {test_cos_103_pure:.4f}")
    print(f"  Val Cosine:  {val_cos_103_pure:.4f}")
    print(f"  Test Cosine: {test_cos_103_pure:.4f}")

    # ========== 实验4: 103维 + Fusion ==========
    log_lines.append("")
    log_lines.append("=" * 80)
    log_lines.append("Experiment 4: 103-dim + Fusion")
    log_lines.append("=" * 80)
    print("\n" + "=" * 80)
    print("Experiment 4: 103-dim + Fusion")
    print("=" * 80)

    retrieval_103 = SimpleRetrieval(top_k=top_k, temperature=temperature)
    retrieval_103.set_database(tensors['train']['en_103'], tensors['train']['es_1024'])
    fusion_103 = FusionNetwork(en_dim=103, es_dim=101)  # 注意：en_dim=103
    model_103 = RetrievalFusionModel(retrieval_103, fusion_103, spanish_indices).to(device)

    print("  Training Fusion network...")
    train_data_103 = {'en': tensors['train']['en_103'], 'es_101': tensors['train']['es_101']}
    val_data_103 = {'en': tensors['val']['en_103'], 'es_101': tensors['val']['es_101']}
    train_fusion_model(model_103, train_data_103, val_data_103, device)

    model_103.eval()
    with torch.no_grad():
        val_output, _ = model_103(tensors['val']['en_103'])
        val_cos_103_fusion = compute_cosine_similarity(val_output, tensors['val']['es_101'])
        test_output, _ = model_103(tensors['test']['en_103'])
        test_cos_103_fusion = compute_cosine_similarity(test_output, tensors['test']['es_101'])

    results['103_fusion'] = {'val': val_cos_103_fusion, 'test': test_cos_103_fusion}
    log_lines.append(f"  Val Cosine:  {val_cos_103_fusion:.4f}")
    log_lines.append(f"  Test Cosine: {test_cos_103_fusion:.4f}")
    print(f"  Val Cosine:  {val_cos_103_fusion:.4f}")
    print(f"  Test Cosine: {test_cos_103_fusion:.4f}")

    # ========== 结果汇总 ==========
    log_lines.append("")
    log_lines.append("=" * 80)
    log_lines.append("SUMMARY")
    log_lines.append("=" * 80)
    log_lines.append("")
    log_lines.append(f"{'Method':<25} {'Val Cosine':<15} {'Test Cosine':<15}")
    log_lines.append("-" * 55)
    log_lines.append(f"{'1024-dim Pure Retrieval':<25} {results['1024_pure']['val']:<15.4f} {results['1024_pure']['test']:<15.4f}")
    log_lines.append(f"{'1024-dim + Fusion':<25} {results['1024_fusion']['val']:<15.4f} {results['1024_fusion']['test']:<15.4f}")
    log_lines.append(f"{'103-dim Pure Retrieval':<25} {results['103_pure']['val']:<15.4f} {results['103_pure']['test']:<15.4f}")
    log_lines.append(f"{'103-dim + Fusion':<25} {results['103_fusion']['val']:<15.4f} {results['103_fusion']['test']:<15.4f}")
    log_lines.append("-" * 55)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n{'Method':<25} {'Val Cosine':<15} {'Test Cosine':<15}")
    print("-" * 55)
    print(f"{'1024-dim Pure Retrieval':<25} {results['1024_pure']['val']:<15.4f} {results['1024_pure']['test']:<15.4f}")
    print(f"{'1024-dim + Fusion':<25} {results['1024_fusion']['val']:<15.4f} {results['1024_fusion']['test']:<15.4f}")
    print(f"{'103-dim Pure Retrieval':<25} {results['103_pure']['val']:<15.4f} {results['103_pure']['test']:<15.4f}")
    print(f"{'103-dim + Fusion':<25} {results['103_fusion']['val']:<15.4f} {results['103_fusion']['test']:<15.4f}")
    print("-" * 55)

    # 对比分析
    diff_pure = results['103_pure']['test'] - results['1024_pure']['test']
    diff_fusion = results['103_fusion']['test'] - results['1024_fusion']['test']

    log_lines.append("")
    log_lines.append("Comparison (Test Set):")
    log_lines.append(f"  Pure Retrieval: 103-dim vs 1024-dim = {diff_pure:+.4f} ({diff_pure/results['1024_pure']['test']*100:+.2f}%)")
    log_lines.append(f"  With Fusion:    103-dim vs 1024-dim = {diff_fusion:+.4f} ({diff_fusion/results['1024_fusion']['test']*100:+.2f}%)")
    log_lines.append("")
    log_lines.append(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_lines.append("=" * 80)

    print(f"\nComparison (Test Set):")
    print(f"  Pure Retrieval: 103-dim vs 1024-dim = {diff_pure:+.4f} ({diff_pure/results['1024_pure']['test']*100:+.2f}%)")
    print(f"  With Fusion:    103-dim vs 1024-dim = {diff_fusion:+.4f} ({diff_fusion/results['1024_fusion']['test']*100:+.2f}%)")

    # 保存日志
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))
    print(f"\nLog saved to: {log_file}")

    # 保存JSON
    save_data = {
        'timestamp': timestamp,
        'device': device,
        'config': {
            'top_k': top_k,
            'temperature': temperature,
            'fusion_epochs': 100
        },
        'results': results,
        'comparison': {
            'pure_retrieval_diff': diff_pure,
            'fusion_diff': diff_fusion
        }
    }

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"JSON saved to: {json_file}")

    return save_data


if __name__ == '__main__':
    config_path = 'E:/interspeech2026/config.yaml'
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    print("=" * 80)
    print("Ablation Study: 103-dim vs 1024-dim Retrieval")
    print("(Pure Retrieval + Fusion)")
    print("=" * 80)
    print(f"Config: {config_path}")

    run_ablation(cfg)

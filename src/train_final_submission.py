"""
Final Submission Training Script
使用最优超参数和官方索引训练 Retrieval + Fusion 模型

最优配置 (来自 system_description.md):
- Top-K: 70
- Temperature: 0.04
- Similarity: cosine
- Fusion Network: [256, 128], no dropout
- 全量数据训练
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.feature_selector import FeatureSelector, OFFICIAL_SPANISH_WINNERS


class SimpleRetrieval:
    """Simple cosine similarity retrieval module"""

    def __init__(self, top_k=70, temperature=0.04):
        self.top_k = top_k
        self.temperature = temperature
        self.en_db = None
        self.es_db = None

    def set_database(self, en_features, es_features):
        """Set retrieval database"""
        self.en_db = F.normalize(en_features, dim=-1)
        self.es_db = es_features

    def retrieve(self, query_en):
        """Retrieve ES features for query EN features"""
        query_norm = F.normalize(query_en, dim=-1)

        # Cosine similarity
        similarities = torch.mm(query_norm, self.en_db.t())

        # Top-K selection
        top_k_sims, top_k_idx = torch.topk(similarities, self.top_k, dim=-1)

        # Temperature-scaled softmax
        weights = F.softmax(top_k_sims / self.temperature, dim=-1)

        # Weighted aggregation
        batch_size = query_en.shape[0]
        es_retrieved = torch.zeros(batch_size, self.es_db.shape[1], device=query_en.device)

        for i in range(batch_size):
            es_retrieved[i] = torch.sum(
                self.es_db[top_k_idx[i]] * weights[i].unsqueeze(-1),
                dim=0
            )

        return es_retrieved


class FusionNetwork(nn.Module):
    """
    Fusion Network: combines EN features and retrieved ES features
    Output = retrieved + delta (residual connection)
    """

    def __init__(self, en_dim=1024, es_dim=101, hidden_dims=[256, 128]):
        super().__init__()

        input_dim = en_dim + es_dim  # 1125

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                # No dropout (best performance)
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, es_dim))
        self.mlp = nn.Sequential(*layers)

        # Zero-initialize last layer (start with no correction)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, en_1024, retrieved_es_101):
        """
        Args:
            en_1024: (B, 1024) - English features
            retrieved_es_101: (B, 101) - Retrieved ES features
        Returns:
            output: (B, 101) - Final prediction
            delta: (B, 101) - The correction
        """
        features = torch.cat([en_1024, retrieved_es_101], dim=-1)
        delta = self.mlp(features)
        output = retrieved_es_101 + delta
        return output, delta


class RetrievalFusionModel(nn.Module):
    """Complete Retrieval + Fusion model"""

    def __init__(self, top_k=70, temperature=0.04, hidden_dims=[256, 128]):
        super().__init__()
        self.retrieval = SimpleRetrieval(top_k=top_k, temperature=temperature)
        self.fusion = FusionNetwork(hidden_dims=hidden_dims)
        self.selector_indices = None

    def set_database(self, en_1024, es_1024):
        """Set retrieval database with 1024-dim features"""
        self.retrieval.set_database(en_1024, es_1024)

    def set_selector_indices(self, indices):
        """Set feature selection indices"""
        self.selector_indices = indices

    def forward(self, en_1024):
        """
        Args:
            en_1024: (B, 1024) - English HuBERT features
        Returns:
            es_pred_101: (B, 101) - Predicted Spanish features
            es_retrieved_101: (B, 101) - Retrieved features
            delta: (B, 101) - Correction applied
        """
        # Step 1: Retrieve in 1024-dim space
        es_retrieved_1024 = self.retrieval.retrieve(en_1024)

        # Step 2: Select to 101-dim using official indices
        if self.selector_indices is not None:
            indices = self.selector_indices
            if isinstance(indices, np.ndarray):
                indices = torch.from_numpy(indices).long().to(es_retrieved_1024.device)
            es_retrieved_101 = es_retrieved_1024[:, indices]
        else:
            es_retrieved_101 = es_retrieved_1024

        # Step 3: Apply fusion network
        es_pred_101, delta = self.fusion(en_1024, es_retrieved_101)

        return es_pred_101, es_retrieved_101, delta


def load_data(data_dir, use_official_indices=True):
    """Load training data"""
    print(f"[1/2] Loading features from {data_dir}...")

    en_files = sorted([f for f in os.listdir(data_dir) if f.startswith('EN_')])
    es_files = sorted([f for f in os.listdir(data_dir) if f.startswith('ES_')])

    en_1024 = np.array([np.load(os.path.join(data_dir, f)) for f in tqdm(en_files, desc="  EN")])
    es_1024 = np.array([np.load(os.path.join(data_dir, f)) for f in tqdm(es_files, desc="  ES")])

    print(f"  Loaded: {len(en_files)} samples, EN shape: {en_1024.shape}, ES shape: {es_1024.shape}")

    print(f"[2/2] Setting up feature selector...")
    if use_official_indices:
        # Use official spanish_winners indices
        selector_indices = np.array(OFFICIAL_SPANISH_WINNERS, dtype=np.int64)
        print(f"  Using OFFICIAL spanish_winners indices ({len(selector_indices)} dims)")
    else:
        # Variance-based selection (NOT recommended)
        selector = FeatureSelector(n_components=101)
        selector.fit(es_1024)
        selector_indices = selector.selected_indices
        print(f"  Using variance-based selection ({len(selector_indices)} dims)")

    # Apply selection to ES features
    es_101 = es_1024[:, selector_indices]
    print(f"  ES_101 shape: {es_101.shape}")

    return en_1024, es_1024, es_101, selector_indices, en_files


def train_full_dataset(
    data_dir="E:/interspeech2026/dral-features/features",
    checkpoint_dir="E:/interspeech2026/checkpoints",
    top_k=70,
    temperature=0.04,
    hidden_dims=[256, 128],
    epochs=100,
    lr=0.001,
    batch_size=32,
    device='cuda'
):
    """Train on full dataset with optimal hyperparameters"""

    print("="*70)
    print("Final Submission Training")
    print("="*70)
    print(f"Config: top_k={top_k}, temperature={temperature}, hidden_dims={hidden_dims}")
    print(f"Training: epochs={epochs}, lr={lr}, batch_size={batch_size}")
    print(f"Device: {device}")
    print()

    # Load data
    en_1024, es_1024, es_101, selector_indices, en_files = load_data(
        data_dir, use_official_indices=True
    )

    # Convert to tensors
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    en_tensor = torch.tensor(en_1024, dtype=torch.float32).to(device)
    es_1024_tensor = torch.tensor(es_1024, dtype=torch.float32).to(device)
    es_101_tensor = torch.tensor(es_101, dtype=torch.float32).to(device)

    # Create model
    print("\nCreating Retrieval + Fusion model...")
    model = RetrievalFusionModel(
        top_k=top_k,
        temperature=temperature,
        hidden_dims=hidden_dims
    ).to(device)

    # Set retrieval database (full dataset)
    model.set_database(en_tensor, es_1024_tensor)
    model.set_selector_indices(selector_indices)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Training
    print(f"\nTraining for {epochs} epochs on {len(en_files)} samples...")
    n_samples = len(en_files)
    best_cos = 0

    for epoch in range(1, epochs + 1):
        model.train()

        # Shuffle
        perm = torch.randperm(n_samples)
        total_loss = 0
        total_cos = 0
        n_batches = (n_samples + batch_size - 1) // batch_size

        for i in range(n_batches):
            batch_idx = perm[i*batch_size : min((i+1)*batch_size, n_samples)]

            en_batch = en_tensor[batch_idx]
            es_batch = es_101_tensor[batch_idx]

            optimizer.zero_grad()

            es_pred, es_ret, delta = model(en_batch)

            # Cosine loss (directly optimize competition metric)
            cos_sim = F.cosine_similarity(es_pred, es_batch, dim=-1)
            loss = (1 - cos_sim).mean()

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_cos += cos_sim.mean().item()

        avg_loss = total_loss / n_batches
        avg_cos = total_cos / n_batches

        # Log
        if epoch % 10 == 0 or epoch <= 5:
            print(f"Epoch {epoch:3d}: Loss={avg_loss:.4f}, Cosine={avg_cos:.4f}")

        # Save best
        if avg_cos > best_cos:
            best_cos = avg_cos
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_cos': best_cos,
                'config': {
                    'top_k': top_k,
                    'temperature': temperature,
                    'hidden_dims': hidden_dims,
                    'selector_indices': selector_indices.tolist()
                }
            }, os.path.join(checkpoint_dir, 'best_final_submission.pth'))

    print("\n" + "="*70)
    print(f"Training Complete! Best Cosine: {best_cos:.4f}")
    print(f"Model saved to: {checkpoint_dir}/best_final_submission.pth")
    print("="*70)

    return model, selector_indices


if __name__ == '__main__':
    train_full_dataset()

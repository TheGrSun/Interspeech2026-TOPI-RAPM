"""
Ensemble Training Script
训练4个模型：1024/103维 × 有/无Fusion

最优配置 (来自 hyperparameter_search_report.md):
- top_k: 90 (不是70!)
- temperature: 0.04
- 使用官方特征索引
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np
from tqdm import tqdm
import argparse
import importlib.util

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 加载官方特征索引
official_feature_selection_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'official_mdekorte', 'feature_selection.py'
)
spec = importlib.util.spec_from_file_location("official_features", official_feature_selection_path)
official_features = importlib.util.module_from_spec(spec)
spec.loader.exec_module(official_features)

SPANISH_WINNERS = np.array(official_features.spanish_winners, dtype=np.int64)
ENGLISH_WINNERS = np.array(official_features.english_winners, dtype=np.int64)

print(f"Loaded official indices:")
print(f"  Spanish winners: {len(SPANISH_WINNERS)} dims")
print(f"  English winners: {len(ENGLISH_WINNERS)} dims")


class SimpleRetrieval:
    """Cosine similarity retrieval module

    Supports two modes:
    - 1024-dim: retrieve in 1024-dim space, return 1024-dim features
    - 103-dim: retrieve in 103-dim space (using english_winners), return 1024-dim features
    """

    def __init__(self, top_k=90, temperature=0.04):
        self.top_k = top_k
        self.temperature = temperature
        self.en_db = None          # Normalized EN features for similarity
        self.es_db_full = None     # Full 1024-dim ES features for retrieval
        self.indices_103 = None    # English indices for 103-dim mode

    def set_database(self, en_features, es_features_full, indices_103=None):
        """Set retrieval database

        Args:
            en_features: EN features (1024-dim or 103-dim if indices_103 provided)
            es_features_full: Full 1024-dim ES features (for retrieval)
            indices_103: English indices for 103-dim mode (optional)
        """
        self.en_db = F.normalize(en_features, dim=-1)
        self.es_db_full = es_features_full
        self.indices_103 = indices_103

    def retrieve(self, query_en, es_1024_database=None):
        """Retrieve ES features for query EN features

        Args:
            query_en: Query EN features (1024-dim or 103-dim)
            es_1024_database: Full 1024-dim ES database (required if not set in set_database)

        Returns:
            Retrieved 1024-dim ES features
        """
        query_norm = F.normalize(query_en, dim=-1)
        similarities = torch.mm(query_norm, self.en_db.t())

        top_k_sims, top_k_idx = torch.topk(similarities, self.top_k, dim=-1)
        weights = F.softmax(top_k_sims / self.temperature, dim=-1)

        batch_size = query_en.shape[0]

        # Use full 1024-dim ES database for retrieval
        es_db = self.es_db_full if es_1024_database is None else es_1024_database
        es_retrieved = torch.zeros(batch_size, es_db.shape[1], device=query_en.device)

        for i in range(batch_size):
            es_retrieved[i] = torch.sum(
                es_db[top_k_idx[i]] * weights[i].unsqueeze(-1),
                dim=0
            )

        return es_retrieved


class FusionNetwork(nn.Module):
    """Fusion Network: combines EN and retrieved ES features"""

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
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, es_dim))
        self.mlp = nn.Sequential(*layers)

        # Zero-initialize last layer
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, en_features, retrieved_es):
        features = torch.cat([en_features, retrieved_es], dim=-1)
        delta = self.mlp(features)
        output = retrieved_es + delta
        return output, delta


class RetrievalModel(nn.Module):
    """Base retrieval model supporting 1024/103 dimensional retrieval"""

    def __init__(self, mode='1024_fusion', top_k=90, temperature=0.04, hidden_dims=[256, 128]):
        super().__init__()
        self.mode = mode
        self.en_dim = 103 if mode.startswith('103') else 1024
        self.use_fusion = 'fusion' in mode

        # Spanish indices for output
        self.spanish_indices = SPANISH_WINNERS

        # English indices for 103-dim mode
        if self.en_dim == 103:
            self.english_indices = ENGLISH_WINNERS
        else:
            self.english_indices = None

        # Retrieval module
        self.retrieval = SimpleRetrieval(top_k=top_k, temperature=temperature)

        # Fusion network (optional)
        if self.use_fusion:
            self.fusion = FusionNetwork(
                en_dim=1024,  # Fusion always uses 1024-dim EN features
                es_dim=101,
                hidden_dims=hidden_dims
            )

    def set_database(self, en_1024, es_1024):
        """Set retrieval database

        Args:
            en_1024: (N, 1024) English features
            es_1024: (N, 1024) Spanish features
        """
        # For 103-dim mode, reduce EN dimensions for similarity computation
        # But always keep ES as 1024-dim for retrieval
        if self.en_dim == 103:
            en_db = en_1024[:, self.english_indices]
        else:
            en_db = en_1024

        # Set database: EN can be 1024 or 103-dim, ES is always 1024-dim
        self.retrieval.set_database(
            en_features=en_db,
            es_features_full=es_1024,
            indices_103=self.english_indices if self.en_dim == 103 else None
        )

    def forward(self, en_1024):
        """Forward pass

        Args:
            en_1024: (B, 1024) English features

        Returns:
            If use_fusion:
                es_pred_101: (B, 101) Predicted features
                es_retrieved_101: (B, 101) Retrieved features
                delta: (B, 101) Correction
            Else:
                es_pred_101: (B, 101) Direct retrieval result
        """
        # For 103-dim mode, reduce query dimension for similarity computation
        if self.en_dim == 103:
            en_query = en_1024[:, self.english_indices]
        else:
            en_query = en_1024

        # Retrieve: returns 1024-dim ES features
        es_retrieved_1024 = self.retrieval.retrieve(en_query)

        # Select to 101-dim using spanish indices
        indices = self.spanish_indices
        if isinstance(indices, np.ndarray):
            indices = torch.from_numpy(indices).long().to(es_retrieved_1024.device)
        es_retrieved_101 = es_retrieved_1024[:, indices]

        # Apply fusion if enabled
        if self.use_fusion:
            # Fusion uses original 1024-dim EN features + 101-dim retrieved
            es_pred_101, delta = self.fusion(en_1024, es_retrieved_101)
            return es_pred_101, es_retrieved_101, delta
        else:
            # Pure retrieval: return retrieved features directly
            return es_retrieved_101


def load_data(data_dir):
    """Load training data"""
    print(f"[1/2] Loading features from {data_dir}...")

    en_files = sorted([f for f in os.listdir(data_dir) if f.startswith('EN_')])
    es_files = sorted([f for f in os.listdir(data_dir) if f.startswith('ES_')])

    en_1024 = np.array([np.load(os.path.join(data_dir, f)) for f in tqdm(en_files, desc="  EN")])
    es_1024 = np.array([np.load(os.path.join(data_dir, f)) for f in tqdm(es_files, desc="  ES")])

    print(f"  Loaded: {len(en_files)} samples")
    print(f"  EN shape: {en_1024.shape}, ES shape: {es_1024.shape}")

    # Select spanish dimensions for training target
    es_101 = es_1024[:, SPANISH_WINNERS]
    print(f"[2/2] ES_101 shape: {es_101.shape}")

    return en_1024, es_1024, es_101, en_files


def train_mode(mode, data_dir, checkpoint_dir, top_k=90, temperature=0.04,
               hidden_dims=[256, 128], epochs=100, lr=0.001, batch_size=32, device='cuda'):
    """Train a single model configuration"""

    print("=" * 70)
    print(f"Training: {mode}")
    print("=" * 70)
    print(f"Config: top_k={top_k}, temp={temperature}, hidden={hidden_dims}")
    print(f"Mode: {mode}")
    print()

    # Load data
    en_1024, es_1024, es_101, en_files = load_data(data_dir)

    # Convert to tensors
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    en_tensor = torch.tensor(en_1024, dtype=torch.float32).to(device)
    es_1024_tensor = torch.tensor(es_1024, dtype=torch.float32).to(device)
    es_101_tensor = torch.tensor(es_101, dtype=torch.float32).to(device)

    # Create model
    print(f"\nCreating model: {mode}...")
    model = RetrievalModel(
        mode=mode,
        top_k=top_k,
        temperature=temperature,
        hidden_dims=hidden_dims
    ).to(device)

    # Set database
    model.set_database(en_tensor, es_1024_tensor)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # For pure retrieval mode (no trainable parameters), just evaluate directly
    if n_params == 0:
        print(f"\nPure retrieval mode - no training needed, evaluating directly...")

        with torch.no_grad():
            model.eval()
            es_pred = model(en_tensor)
            cos_sim = F.cosine_similarity(es_pred, es_101_tensor, dim=-1)
            best_cos = cos_sim.mean().item()

        print(f"Cosine similarity: {best_cos:.4f}")

        # Save checkpoint
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint = {
            'mode': mode,
            'epoch': 0,
            'best_cos': best_cos,
            'config': {
                'top_k': top_k,
                'temperature': temperature,
                'hidden_dims': hidden_dims,
                'spanish_indices': SPANISH_WINNERS.tolist(),
                'english_indices': ENGLISH_WINNERS.tolist() if model.en_dim == 103 else None
            }
        }
        torch.save(checkpoint, os.path.join(checkpoint_dir, f'model_{mode}.pth'))

        print(f"\nSaved: {checkpoint_dir}/model_{mode}.pth")
        print("=" * 70)
        print()

        return model, best_cos

    # Training mode (with fusion network)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    print(f"\nTraining for {epochs} epochs...")
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

            # Forward pass (different return types for fusion vs pure)
            if model.use_fusion:
                es_pred, es_ret, delta = model(en_batch)
            else:
                es_pred = model(en_batch)

            # Cosine loss
            cos_sim = F.cosine_similarity(es_pred, es_batch, dim=-1)
            loss = (1 - cos_sim).mean()

            # Only backward if there are trainable parameters
            if n_params > 0:
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

            checkpoint = {
                'mode': mode,
                'epoch': epoch,
                'best_cos': best_cos,
                'config': {
                    'top_k': top_k,
                    'temperature': temperature,
                    'hidden_dims': hidden_dims,
                    'spanish_indices': SPANISH_WINNERS.tolist(),
                    'english_indices': ENGLISH_WINNERS.tolist() if model.en_dim == 103 else None
                }
            }

            # Save model state if fusion enabled
            if model.use_fusion:
                checkpoint['model_state_dict'] = model.state_dict()

            torch.save(checkpoint, os.path.join(checkpoint_dir, f'model_{mode}.pth'))

    print(f"\nBest Cosine: {best_cos:.4f}")
    print(f"Saved: {checkpoint_dir}/model_{mode}.pth")
    print("=" * 70)
    print()

    return model, best_cos


def main():
    parser = argparse.ArgumentParser(description='Train ensemble models')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', '1024_fusion', '1024_pure', '103_fusion', '103_pure'],
                        help='Which model(s) to train')
    parser.add_argument('--data_dir', type=str, default='E:/interspeech2026/dral-features/features')
    parser.add_argument('--checkpoint_dir', type=str, default='E:/interspeech2026/checkpoints')
    parser.add_argument('--top_k', type=int, default=90)
    parser.add_argument('--temperature', type=float, default=0.04)
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 128])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    # Determine which modes to train
    if args.mode == 'all':
        modes = ['1024_fusion', '1024_pure', '103_fusion', '103_pure']
    else:
        modes = [args.mode]

    # Train each mode
    results = {}
    for mode in modes:
        model, best_cos = train_mode(
            mode=mode,
            data_dir=args.data_dir,
            checkpoint_dir=args.checkpoint_dir,
            top_k=args.top_k,
            temperature=args.temperature,
            hidden_dims=args.hidden_dims,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=args.device
        )
        results[mode] = best_cos

    # Summary
    print("\n" + "=" * 70)
    print("Training Summary")
    print("=" * 70)
    for mode, cos in results.items():
        print(f"  {mode:<20} {cos:.4f}")
    print("=" * 70)


if __name__ == '__main__':
    main()

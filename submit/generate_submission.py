"""
Generate submission for CodaBench
Input: Test EN features (1024-dim)
Output: Predicted ES features (101-dim) as .npy files
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.feature_selector import FeatureSelector


class RetrievalPredictor:
    """K-NN retrieval in 1024-dim space"""
    
    def __init__(self, top_k=20, temperature=0.1):
        self.top_k = top_k
        self.temperature = temperature
        self.db_en = None
        self.db_es = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def fit(self, train_en_1024, train_es_1024):
        """Set retrieval database"""
        self.db_en = torch.tensor(train_en_1024).float().to(self.device)
        self.db_es = torch.tensor(train_es_1024).float().to(self.device)
        print(f"Database: {self.db_en.shape[0]} samples")
    
    def predict(self, test_en_1024):
        """Predict ES features for test EN"""
        query = torch.tensor(test_en_1024).float().to(self.device)
        if query.ndim == 1:
            query = query.unsqueeze(0)
        
        # Normalize
        q_norm = F.normalize(query, dim=-1)
        db_norm = F.normalize(self.db_en, dim=-1)
        
        # Cosine similarity
        sims = torch.matmul(q_norm, db_norm.T)
        
        # Top-k
        k = min(self.top_k, self.db_en.shape[0])
        topk_sims, topk_idx = torch.topk(sims, k, dim=1)
        
        # Weighted average
        weights = F.softmax(topk_sims / self.temperature, dim=1)
        topk_es = self.db_es[topk_idx]
        retrieved = torch.bmm(weights.unsqueeze(1), topk_es).squeeze(1)
        
        return retrieved.cpu().numpy()


def main():
    # Paths
    train_dir = "/home/luoxiaoyang/interspeech2026/dral-features/features"
    test_dir = "/home/luoxiaoyang/interspeech2026/test-features"  # 测试集特征路径
    output_dir = "/home/luoxiaoyang/interspeech2026/submit/predictions"
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("Generating Submission")
    print("="*60)
    
    # Load training data
    print("\n[1/4] Loading training data...")
    train_en_files = sorted([f for f in os.listdir(train_dir) if f.startswith('EN_')])
    train_es_files = sorted([f for f in os.listdir(train_dir) if f.startswith('ES_')])
    
    train_en_1024 = np.array([np.load(os.path.join(train_dir, f)) for f in tqdm(train_en_files)])
    train_es_1024 = np.array([np.load(os.path.join(train_dir, f)) for f in tqdm(train_es_files)])
    
    print(f"  Train EN: {train_en_1024.shape}")
    print(f"  Train ES: {train_es_1024.shape}")
    
    # Feature selector (fit on training ES)
    print("\n[2/4] Fitting feature selector...")
    selector = FeatureSelector(n_components=101)
    selector.fit(train_es_1024)
    
    # Create predictor
    print("\n[3/4] Setting up retrieval predictor...")
    predictor = RetrievalPredictor(top_k=20, temperature=0.1)
    predictor.fit(train_en_1024, train_es_1024)
    
    # Load test data and predict
    print("\n[4/4] Generating predictions...")
    
    if not os.path.exists(test_dir):
        print(f"\n⚠️ 测试集目录不存在: {test_dir}")
        print("请将测试集的EN特征放到该目录")
        print("\n使用训练集演示...")
        test_en_files = train_en_files[:10]  # Demo with first 10
        test_dir = train_dir
    else:
        test_en_files = sorted([f for f in os.listdir(test_dir) if f.startswith('EN_')])
    
    for en_file in tqdm(test_en_files, desc="Predicting"):
        # Load test EN feature
        test_en = np.load(os.path.join(test_dir, en_file))
        
        # Predict ES (1024-dim)
        es_pred_1024 = predictor.predict(test_en)
        
        # Select 101 dimensions
        es_pred_101 = selector.transform(es_pred_1024).squeeze()
        
        # Save with ES_ prefix
        es_file = en_file.replace('EN_', 'ES_').replace('_features.npy', '_features.npy')
        np.save(os.path.join(output_dir, es_file), es_pred_101.astype(np.float64))
    
    # Create zip
    print("\n[5/5] Creating submission zip...")
    zip_path = "/home/luoxiaoyang/interspeech2026/submit/submission.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in os.listdir(output_dir):
            if f.endswith('.npy'):
                zf.write(os.path.join(output_dir, f), f)
    
    print(f"\n✅ Submission created: {zip_path}")
    print(f"   Files: {len(os.listdir(output_dir))} .npy files")
    
    # Verify format
    sample = np.load(os.path.join(output_dir, os.listdir(output_dir)[0]))
    print(f"   Shape: {sample.shape} (should be (101,))")
    print(f"   Dtype: {sample.dtype}")


if __name__ == '__main__':
    main()


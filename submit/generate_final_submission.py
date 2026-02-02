"""
Generate Final Submission
使用训练好的 Retrieval + Fusion 模型生成提交文件
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train_final_submission import RetrievalFusionModel


def generate_submission(
    model_path="E:/interspeech2026/checkpoints/best_final_submission.pth",
    train_dir="E:/interspeech2026/dral-features/features",
    test_dir="E:/interspeech2026/test-features",
    output_dir="E:/interspeech2026/submit/predictions_final",
    zip_path="E:/interspeech2026/submit/submission_final.zip",
    device='cuda'
):
    """Generate submission files"""

    print("="*70)
    print("Generating Final Submission")
    print("="*70)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1. Load model
    print("\n[1/4] Loading model...")
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']

    model = RetrievalFusionModel(
        top_k=config['top_k'],
        temperature=config['temperature'],
        hidden_dims=config['hidden_dims']
    ).to(device)

    # Load fusion network weights only
    model_state = checkpoint['model_state_dict']
    fusion_state = {k.replace('fusion.', ''): v for k, v in model_state.items() if k.startswith('fusion.')}
    model.fusion.load_state_dict(fusion_state)
    model.eval()

    # Set selector indices
    selector_indices = np.array(config['selector_indices'], dtype=np.int64)
    model.set_selector_indices(selector_indices)

    print(f"  Config: top_k={config['top_k']}, temp={config['temperature']}")
    print(f"  Best training cosine: {checkpoint['best_cos']:.4f}")

    # 2. Setup retrieval database
    print("\n[2/4] Setting up retrieval database...")
    en_files = sorted([f for f in os.listdir(train_dir) if f.startswith('EN_')])
    es_files = sorted([f for f in os.listdir(train_dir) if f.startswith('ES_')])

    train_en = np.array([np.load(os.path.join(train_dir, f)) for f in tqdm(en_files, desc="  EN")])
    train_es = np.array([np.load(os.path.join(train_dir, f)) for f in tqdm(es_files, desc="  ES")])

    train_en_tensor = torch.tensor(train_en, dtype=torch.float32).to(device)
    train_es_tensor = torch.tensor(train_es, dtype=torch.float32).to(device)
    model.set_database(train_en_tensor, train_es_tensor)
    print(f"  Database: {len(en_files)} samples")

    # 3. Generate predictions
    print("\n[3/4] Generating predictions...")
    os.makedirs(output_dir, exist_ok=True)

    test_files = sorted([f for f in os.listdir(test_dir) if f.startswith('EN_') and f.endswith('.npy')])
    print(f"  Test files: {len(test_files)}")

    with torch.no_grad():
        for en_file in tqdm(test_files, desc="  Predicting"):
            # Load test feature
            en_path = os.path.join(test_dir, en_file)
            test_en = np.load(en_path)

            if test_en.ndim == 1:
                test_en = test_en.reshape(1, -1)

            test_en_tensor = torch.tensor(test_en, dtype=torch.float32).to(device)

            # Predict
            es_pred, _, _ = model(test_en_tensor)
            es_pred_np = es_pred.cpu().numpy().squeeze()

            # Ensure shape and dtype
            es_pred_np = es_pred_np.reshape(101,).astype(np.float64)

            # Output filename: EN_xxx_xx_features.npy -> ES_xxx_xx.npy
            base = en_file.replace('.npy', '').replace('_features', '')
            es_file = base.replace('EN_', 'ES_') + '.npy'

            np.save(os.path.join(output_dir, es_file), es_pred_np)

    # 4. Create zip
    print("\n[4/4] Creating submission zip...")
    output_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.npy')])

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in tqdm(output_files, desc="  Zipping"):
            zf.write(os.path.join(output_dir, f), f)

    # Verification
    print("\n" + "="*70)
    print("Submission Verification")
    print("="*70)

    sample = np.load(os.path.join(output_dir, output_files[0]))
    print(f"Total files: {len(output_files)}")
    print(f"Sample shape: {sample.shape} (expected: (101,))")
    print(f"Sample dtype: {sample.dtype} (expected: float64)")
    print(f"Sample range: [{sample.min():.4f}, {sample.max():.4f}]")

    if sample.shape == (101,) and sample.dtype == np.float64:
        print("\n✅ Submission ready!")
        print(f"   Zip: {zip_path}")
    else:
        print("\n❌ Verification failed!")

    return zip_path


if __name__ == '__main__':
    generate_submission()

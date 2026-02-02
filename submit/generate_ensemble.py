"""
Generate Ensemble Submissions
批量生成4个模型的提交文件

使用训练好的模型处理测试集，生成符合比赛格式的提交文件
"""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import zipfile
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入训练脚本中的模型定义
from src.train_ensemble import RetrievalModel, SPANISH_WINNERS, ENGLISH_WINNERS


def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        model: Loaded model
        config: Model configuration
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    mode = checkpoint['mode']

    print(f"  Mode: {mode}")
    print(f"  Config: top_k={config['top_k']}, temp={config['temperature']}")
    print(f"  Best cosine: {checkpoint['best_cos']:.4f}")

    # Create model
    model = RetrievalModel(
        mode=mode,
        top_k=config['top_k'],
        temperature=config['temperature'],
        hidden_dims=config.get('hidden_dims', [256, 128])
    ).to(device)

    # Load state dict if fusion model
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    return model, config


def setup_database(model, train_dir, device):
    """Setup retrieval database from training data

    Args:
        model: RetrievalModel instance
        train_dir: Directory containing training features
        device: Device to use
    """
    print("  Loading training data for database...")

    en_files = sorted([f for f in os.listdir(train_dir) if f.startswith('EN_')])
    es_files = sorted([f for f in os.listdir(train_dir) if f.startswith('ES_')])

    en_1024 = np.array([np.load(os.path.join(train_dir, f)) for f in tqdm(en_files, desc="    EN")])
    es_1024 = np.array([np.load(os.path.join(train_dir, f)) for f in tqdm(es_files, desc="    ES")])

    en_tensor = torch.tensor(en_1024, dtype=torch.float32).to(device)
    es_tensor = torch.tensor(es_1024, dtype=torch.float32).to(device)

    model.set_database(en_tensor, es_tensor)

    print(f"    Database: {len(en_files)} samples")


def generate_submission(mode, model, test_dir, output_dir, device):
    """Generate submission for a single model

    Args:
        mode: Model mode (e.g., '1024_fusion')
        model: Trained model
        test_dir: Directory containing test features
        output_dir: Directory to save predictions
        device: Device to use
    """
    print(f"\n[3/4] Generating predictions for {mode}...")

    # Create mode-specific output directory
    mode_output_dir = os.path.join(output_dir, f"predictions_{mode}")
    os.makedirs(mode_output_dir, exist_ok=True)

    # Get test files
    test_files = sorted([f for f in os.listdir(test_dir)
                        if f.startswith('EN_') and f.endswith('.npy')])
    print(f"  Test files: {len(test_files)}")

    with torch.no_grad():
        for en_file in tqdm(test_files, desc=f"  {mode}"):
            # Load test feature
            en_path = os.path.join(test_dir, en_file)
            test_en = np.load(en_path)

            if test_en.ndim == 1:
                test_en = test_en.reshape(1, -1)

            test_en_tensor = torch.tensor(test_en, dtype=torch.float32).to(device)

            # Predict
            if model.use_fusion:
                es_pred, _, _ = model(test_en_tensor)
            else:
                es_pred = model(test_en_tensor)

            es_pred_np = es_pred.cpu().numpy().squeeze()

            # Ensure shape and dtype
            es_pred_np = es_pred_np.reshape(101,).astype(np.float64)

            # Output filename: EN_xxx_xx_features.npy -> ES_xxx_xx.npy
            base = en_file.replace('.npy', '').replace('_features', '')
            es_file = base.replace('EN_', 'ES_') + '.npy'

            np.save(os.path.join(mode_output_dir, es_file), es_pred_np)

    print(f"  Saved to: {mode_output_dir}")
    return mode_output_dir


def create_zip(predictions_dir, zip_path):
    """Create zip file from predictions

    Args:
        predictions_dir: Directory containing .npy prediction files
        zip_path: Path to save zip file
    """
    print(f"\n[4/4] Creating submission zip...")

    output_files = sorted([f for f in os.listdir(predictions_dir) if f.endswith('.npy')])

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in tqdm(output_files, desc="  Zipping"):
            zf.write(os.path.join(predictions_dir, f), f)

    # Verification
    sample = np.load(os.path.join(predictions_dir, output_files[0]))
    print(f"\n  Verification:")
    print(f"    Total files: {len(output_files)}")
    print(f"    Sample shape: {sample.shape} (expected: (101,))")
    print(f"    Sample dtype: {sample.dtype} (expected: float64)")
    print(f"    Sample range: [{sample.min():.4f}, {sample.max():.4f}]")

    if sample.shape == (101,) and sample.dtype == np.float64:
        print(f"  OK: {zip_path}")
    else:
        print(f"  ERROR: Verification failed!")


def generate_all_submissions(
    checkpoint_dir="E:/interspeech2026/checkpoints",
    train_dir="E:/interspeech2026/dral-features/features",
    test_dir="E:/interspeech2026/test-features",
    output_dir="E:/interspeech2026/submit/submissions",
    modes=['1024_fusion', '1024_pure', '103_fusion', '103_pure'],
    device='cuda'
):
    """Generate submissions for all models"""

    print("=" * 70)
    print("Generating Ensemble Submissions")
    print("=" * 70)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    os.makedirs(output_dir, exist_ok=True)

    # Process each mode
    for mode in modes:
        print("-" * 70)
        print(f"Processing: {mode}")
        print("-" * 70)

        # [1/4] Load model
        print("[1/4] Loading model...")
        checkpoint_path = os.path.join(checkpoint_dir, f"model_{mode}.pth")

        if not os.path.exists(checkpoint_path):
            print(f"  WARNING: Checkpoint not found: {checkpoint_path}")
            print(f"  Skipping {mode}")
            continue

        model, config = load_model(checkpoint_path, device)

        # [2/4] Setup database
        print("\n[2/4] Setting up retrieval database...")
        setup_database(model, train_dir, device)

        # [3/4] Generate predictions
        predictions_dir = generate_submission(mode, model, test_dir, output_dir, device)

        # [4/4] Create zip
        zip_path = os.path.join(output_dir, f"submission_{mode}.zip")
        create_zip(predictions_dir, zip_path)

    print("\n" + "=" * 70)
    print("All submissions generated!")
    print(f"Output directory: {output_dir}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Generate ensemble submissions')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='E:/interspeech2026/checkpoints',
                        help='Directory containing model checkpoints')
    parser.add_argument('--train_dir', type=str,
                        default='E:/interspeech2026/dral-features/features',
                        help='Directory containing training features')
    parser.add_argument('--test_dir', type=str,
                        default='E:/interspeech2026/test-features',
                        help='Directory containing test features')
    parser.add_argument('--output_dir', type=str,
                        default='E:/interspeech2026/submit/submissions',
                        help='Directory to save submissions')
    parser.add_argument('--modes', type=str, nargs='+',
                        default=['1024_fusion', '1024_pure', '103_fusion', '103_pure'],
                        choices=['1024_fusion', '1024_pure', '103_fusion', '103_pure', 'all'],
                        help='Which modes to generate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    # Handle 'all' option
    if 'all' in args.modes:
        args.modes = ['1024_fusion', '1024_pure', '103_fusion', '103_pure']

    generate_all_submissions(
        checkpoint_dir=args.checkpoint_dir,
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        modes=args.modes,
        device=args.device
    )


if __name__ == '__main__':
    main()

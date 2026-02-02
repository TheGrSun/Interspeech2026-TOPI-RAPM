"""
从测试音频文件提取1024维HuBERT特征

使用本地HuBERT模型：HoloRAG/Aura-VAE/model/hubert-large-ls960-ft
输出：EN_xxx_x_features.npy (1024维特征文件)
"""

import os
import sys
import argparse
import numpy as np
import torch
import torchaudio
from transformers import HubertModel, Wav2Vec2FeatureExtractor
from tqdm import tqdm

def extract_features_from_audio(wav_path, model, feature_extractor, device='cuda'):
    """
    从单个音频文件提取1024维HuBERT特征
    
    Args:
        wav_path: 音频文件路径
        model: HuBERT模型
        feature_extractor: 特征提取器
        device: 设备
    
    Returns:
        features: (T, 1024) 特征数组，T是时间步数
    """
    # 加载音频
    waveform, sample_rate = torchaudio.load(wav_path)
    
    # 确保是单声道
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # 转换为numpy数组（16kHz采样率）
    audio_array = waveform.squeeze().numpy()
    
    # 如果采样率不是16kHz，需要重采样
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        audio_array = waveform.squeeze().numpy()
    
    # 使用feature_extractor处理音频
    inputs = feature_extractor(
        audio_array,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )
    
    # 移动到设备
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 提取特征
    with torch.no_grad():
        outputs = model(**inputs)
        # 获取最后一层的隐藏状态
        features = outputs.last_hidden_state  # (batch, seq_len, 1024)
    
    # 转换为numpy并移除batch维度
    features = features.squeeze(0).cpu().numpy()  # (seq_len, 1024)
    
    # 对时间维度求平均，得到全局特征向量
    # 或者返回所有时间步的特征（这里我们返回平均特征）
    global_feature = np.mean(features, axis=0)  # (1024,)
    
    return global_feature


def main():
    parser = argparse.ArgumentParser(description='Extract 1024-dim HuBERT features from audio files')
    parser.add_argument('--model_path', type=str,
                        default='/home/luoxiaoyang/HoloRAG/Aura-VAE/model/hubert-large-ls960-ft',
                        help='Path to HuBERT model')
    parser.add_argument('--input_dir', type=str,
                        default='/home/luoxiaoyang/interspeech2026/test-features',
                        help='Directory containing input .wav files')
    parser.add_argument('--output_dir', type=str,
                        default='/home/luoxiaoyang/interspeech2026/test-features',
                        help='Directory to save output .npy feature files')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (currently only supports 1)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("="*70)
    print("HuBERT Feature Extraction")
    print("="*70)
    
    # 1. Load model
    print(f"\n[1/4] Loading HuBERT model from {args.model_path}...")
    try:
        model = HubertModel.from_pretrained(args.model_path)
        model.to(device)
        model.eval()
        print(f"✅ Model loaded successfully")
        print(f"   Hidden size: {model.config.hidden_size}")
        print(f"   Number of layers: {model.config.num_hidden_layers}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
    
    # 2. Load feature extractor
    print(f"\n[2/4] Loading feature extractor...")
    try:
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_path)
        print(f"✅ Feature extractor loaded")
    except Exception as e:
        print(f"❌ Error loading feature extractor: {e}")
        sys.exit(1)
    
    # 3. Find input files
    print(f"\n[3/4] Finding input files in {args.input_dir}...")
    if not os.path.exists(args.input_dir):
        print(f"❌ Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    wav_files = sorted([f for f in os.listdir(args.input_dir) 
                        if f.startswith('EN_') and f.endswith('.wav')])
    
    if not wav_files:
        print(f"❌ No EN_*.wav files found in {args.input_dir}")
        sys.exit(1)
    
    print(f"   Found {len(wav_files)} .wav files")
    
    # 4. Extract features
    print(f"\n[4/4] Extracting features...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    success_count = 0
    error_count = 0
    
    for wav_file in tqdm(wav_files, desc="Extracting"):
        wav_path = os.path.join(args.input_dir, wav_file)
        
        # Generate output filename: EN_xxx_x.wav -> EN_xxx_x_features.npy
        base_name = wav_file.replace('.wav', '')
        output_file = base_name + '_features.npy'
        output_path = os.path.join(args.output_dir, output_file)
        
        # Skip if already exists
        if os.path.exists(output_path):
            tqdm.write(f"   Skipping {wav_file} (already exists)")
            success_count += 1
            continue
        
        try:
            # Extract features
            features = extract_features_from_audio(
                wav_path, model, feature_extractor, device
            )
            
            # Verify shape
            if features.shape != (1024,):
                raise ValueError(f"Expected shape (1024,), got {features.shape}")
            
            # Save as float64 (or float32)
            features = features.astype(np.float64)
            
            # Save
            np.save(output_path, features)
            success_count += 1
            
        except Exception as e:
            tqdm.write(f"   ❌ Error processing {wav_file}: {e}")
            error_count += 1
            continue
    
    # 5. Summary
    print("\n" + "="*70)
    print("Extraction Summary")
    print("="*70)
    print(f"   Total files: {len(wav_files)}")
    print(f"   Success: {success_count}")
    print(f"   Errors: {error_count}")
    print(f"   Output directory: {args.output_dir}")
    
    if error_count == 0:
        print("\n✅ All features extracted successfully!")
    else:
        print(f"\n⚠️  {error_count} files failed. Please check the errors above.")
    
    # Verify a sample file
    if success_count > 0:
        sample_files = [f for f in os.listdir(args.output_dir) 
                       if f.endswith('_features.npy') and f.startswith('EN_')]
        if sample_files:
            sample_path = os.path.join(args.output_dir, sample_files[0])
            sample_feat = np.load(sample_path)
            print(f"\n   Sample file: {sample_files[0]}")
            print(f"   Shape: {sample_feat.shape} (expected: (1024,))")
            print(f"   Dtype: {sample_feat.dtype}")


if __name__ == '__main__':
    main()

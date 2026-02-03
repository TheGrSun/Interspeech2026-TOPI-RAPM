# Configuration Files

## Overview

This directory contains YAML configuration files for the R-APM system (Interspeech 2026 TOPI Challenge).

## Files

### `default.yaml`

The default configuration file used for training the R-APM model. This configuration was used to achieve the best results (Config B: 0.8741 on internal split, **0.8331** on official test set).

### Key Parameters

#### Model Configuration

```yaml
model:
  retrieval:
    top_k: 70              # Optimal number (Paper Table III)
    temperature: 0.04      # Optimal temperature (Paper Table III)
    similarity_type: "cosine"  # Cosine similarity works best

  fusion:
    enabled: true          # Enable fusion network for submission model
    hidden_dims: [256, 128, 64]  # MLP architecture: [1125 -> 256 -> 128 -> 101]
    dropout: 0.0           # No dropout performs best
```

#### Training Configuration

```yaml
training:
  loss_type: "cosine"      # Directly optimize competition metric
  learning_rate: 0.001     # Optimal learning rate (Paper Section 4.2)
  epochs: 100              # Maximum training epochs
  batch_size: 32           # Training batch size
```

#### Feature Selection

```yaml
feature_selection:
  method: "official"       # Predefined official indices from baseline
  n_components: 101        # Spanish winners from official feature selection
  query_dim: 103           # English winners for Config B (subspace retrieval)
  source: "official_baseline/feature_selection.py"
```

**Note**: Config B uses 103-dim query projection (`english_winners`) for retrieval, but retrieves full 1024-dim values. Output uses 101 `spanish_winners` indices.

## Usage

```bash
# Train with default configuration (Config B)
python src/train.py --config config/default.yaml

# Override specific parameters
python src/train.py --config config/default.yaml --model.retrieval.top_k 50
```

## Configuration Tips

1. **Top-K Selection**: 70 is optimal (Paper Table III: K=70 achieves 0.872)
2. **Temperature**: 0.04 is optimal (Paper Table III: τ=0.04 achieves 0.872)
3. **Similarity**: Pure cosine similarity outperforms hybrid approaches
4. **Dropout**: Set to 0.0 for best performance (don't use dropout)
5. **Fusion**: Always enable for submission; pure retrieval is for ablation only

## Ablation Configurations

For ablation studies, you can create variations:

- `pure_retrieval.yaml`: Disable fusion network for pure retrieval baseline
- `config_a.yaml`: Use 1024-dim search space (Config A: High-Res)
- `config_b.yaml`: Use 103-dim search space (Config B: Subspace) - **Default**

## Reference

For more details on hyperparameter choices, see the paper:
- [InterspeechPaperRAPM.tex.pdf](../InterspeechPaperRAPM.tex.pdf)
- Section 4.4: Hyperparameter Sensitivity (Table III)
- Section 3.3: Retrieval Configurations

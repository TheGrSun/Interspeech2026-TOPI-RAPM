# Configuration Files

## Overview

This directory contains YAML configuration files for the R-APM system.

## Files

### `default.yaml`

The default configuration file used for training the R-APM model. This configuration was used to achieve the best results (0.8742 cosine similarity on internal split, 0.8288 on official test set).

### Key Parameters

#### Model Configuration

```yaml
model:
  retrieval:
    top_k: 70              # Optimal number of retrieval candidates
    temperature: 0.04      # Sharp attention distribution
    similarity_type: "cosine"  # Cosine similarity works best

  fusion:
    enabled: true          # Enable fusion network for submission model
    hidden_dims: [256, 128, 64]  # Multi-scale MLP architecture
    dropout: 0.0           # No dropout performs best
```

#### Training Configuration

```yaml
training:
  loss_type: "cosine"      # Directly optimize competition metric
  learning_rate: 0.0008    # Optimal learning rate
  epochs: 500              # Maximum training epochs
  batch_size: 64           # Training batch size
```

#### Feature Selection

```yaml
feature_selection:
  method: "official"       # Predefined official indices from baseline
  n_components: 101        # Spanish winners from official feature selection
  source: "official_baseline/feature_selection.py"
```

**Note**: The feature selection uses 101 predefined indices from the official competition baseline (`spanish_winners` in `official_baseline/feature_selection.py`). These indices were selected using a greedy algorithm to find features most useful for detecting pragmatic similarity.

## Usage

```bash
# Train with default configuration
python src/train.py --config config/default.yaml

# Override specific parameters
python src/train.py --config config/default.yaml --model.retrieval.top_k 50
```

## Configuration Tips

1. **Top-K Selection**: Values between 60-100 work well, with 70 being optimal
2. **Temperature**: Lower values (0.04-0.07) produce sharper attention distributions
3. **Similarity**: Pure cosine similarity outperforms hybrid approaches
4. **Dropout**: Set to 0.0 for best performance (don't use dropout)
5. **Fusion**: Always enable for submission; pure retrieval is for ablation only

## Ablation Configurations

For ablation studies, you can create variations:

- `pure_retrieval.yaml`: Disable fusion network for pure retrieval baseline
- `103dim.yaml`: Use 103-dim search space instead of 1024-dim
- `high_temp.yaml`: Test higher temperature values (> 0.1)

## Reference

For more details on hyperparameter choices, see:
- `docs/hyperparameter_search_report.md`
- `docs/ablation_aggregation_dropout_report.md`

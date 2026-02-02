"""
Feature Selection: Extract pragmatically-salient dimensions
Based on baseline feature_selection.py

For Spanish: select 101 dimensions
For English: select 103 dimensions
"""

import numpy as np
import torch
import pickle
import os

# Official predefined indices from mdekorte/Pragmatic_Similarity_Computation
# These are the "pragmatically-salient" dimensions selected by the official baseline
OFFICIAL_SPANISH_WINNERS = [
    41, 48, 67, 85, 151, 226, 227, 261, 346, 347, 354, 357, 411,
    415, 440, 448, 463, 468, 474, 510, 518, 528, 530, 585, 603,
    609, 614, 618, 665, 703, 705, 732, 736, 792, 793, 814, 830,
    835, 852, 858, 864, 895, 900, 906, 912, 915, 937, 971, 977,
    1020, 1023, 174, 177, 206, 245, 249, 266, 397, 490, 494, 666,
    689, 815, 861, 81, 208, 280, 398, 407, 409, 525, 533, 926, 927,
    117, 287, 384, 655, 656, 809, 887, 897, 31, 36, 112, 231, 232,
    472, 807, 881, 932, 944, 1004, 62, 63, 115, 155, 551, 682, 946, 1007
]

OFFICIAL_ENGLISH_WINNERS = [
    0, 2, 41, 54, 63, 67, 102, 155, 159, 172, 173, 193, 197, 248,
    261, 280, 286, 350, 358, 431, 459, 482, 488, 528, 603, 616,
    618, 707, 708, 715, 717, 731, 792, 799, 804, 809, 824, 828,
    870, 878, 880, 900, 903, 959, 45, 113, 120, 121, 204, 206,
    246, 269, 411, 453, 510, 559, 602, 656, 666, 929, 972, 973,
    1013, 1016, 13, 17, 105, 134, 136, 185, 188, 474, 578, 622,
    651, 882, 925, 162, 410, 577, 628, 750, 758, 866, 869,
    952, 963, 965, 50, 168, 436, 470, 513, 527, 557, 660,
    732, 514, 661, 694, 698, 935, 937
]


class FeatureSelector:
    """
    Select most important dimensions based on variance and pragmatic salience
    Compatible with both numpy and torch tensors

    Supports two modes:
    - 'variance': Select dimensions with highest variance (default)
    - 'predefined': Use pre-defined indices from official evaluation
    """

    def __init__(self, n_components=101, method='variance', indices_path=None):
        """
        Args:
            n_components: Number of dimensions to select
            method: 'variance' (use variance) or 'predefined' (use pre-defined indices)
            indices_path: Path to .npy file with pre-defined indices (for method='predefined')
        """
        self.n_components = n_components
        self.method = method
        self.indices_path = indices_path
        self.selected_indices = None

        # If predefined indices path is provided, load them immediately
        if indices_path is not None and os.path.exists(indices_path):
            self.selected_indices = np.load(indices_path)
            self.method = 'predefined'
            self.n_components = len(self.selected_indices)
            print(f"Loaded predefined indices from {indices_path}")
            print(f"  Number of dimensions: {self.n_components}")
            print(f"  Index range: [{self.selected_indices.min()}, {self.selected_indices.max()}]")
        
    def fit(self, features):
        """
        Fit selector on training features

        Args:
            features: numpy array of shape (N, 1024) or list of arrays

        Note: If predefined indices were loaded in __init__, this method does nothing.
        """
        # Skip fitting if predefined indices are already loaded
        if self.method == 'predefined' and self.selected_indices is not None:
            print(f"Using predefined indices ({self.n_components} dimensions), skipping fit")
            return self

        # Convert to numpy if needed
        if isinstance(features, list):
            features = np.array(features)
        if isinstance(features, torch.Tensor):
            features = features.numpy()

        # Compute variance for each dimension
        variances = features.var(axis=0)

        # Select top-k dimensions with highest variance
        self.selected_indices = np.argsort(variances)[-self.n_components:][::-1]

        print(f"Selected {self.n_components} dimensions (variance-based)")
        print(f"  Index range: [{self.selected_indices.min()}, {self.selected_indices.max()}]")
        print(f"  Variance coverage: {variances[self.selected_indices].sum() / variances.sum():.2%}")

        return self
        
    def transform(self, features):
        """
        Transform features by selecting dimensions
        
        Args:
            features: numpy array or torch tensor of shape (N, 1024) or (1024,)
            
        Returns:
            Selected features with same type as input
        """
        if self.selected_indices is None:
            raise RuntimeError("Must call fit() before transform()")
        
        # Handle single sample
        is_tensor = isinstance(features, torch.Tensor)
        if is_tensor:
            device = features.device
            if features.ndim == 1:
                features = features.unsqueeze(0)
                squeeze = True
            else:
                squeeze = False
            # Detach to avoid gradient issues
            features_np = features.detach().cpu().numpy()
        else:
            if isinstance(features, np.ndarray) and features.ndim == 1:
                features = features.reshape(1, -1)
                squeeze = True
            else:
                squeeze = False
            features_np = features
        
        # Select dimensions
        selected = features_np[:, self.selected_indices]
        
        if squeeze:
            selected = selected.squeeze(0)
        
        if is_tensor:
            return torch.from_numpy(selected).float().to(device)
        return selected
    
    def fit_transform(self, features):
        """Fit and transform in one step"""
        return self.fit(features).transform(features)
    
    def save(self, filepath):
        """Save fitted selector"""
        if self.selected_indices is None:
            raise RuntimeError("Cannot save unfitted selector")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'selected_indices': self.selected_indices,
                'n_components': self.n_components,
                'method': self.method
            }, f)
        print(f"Feature selector saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load fitted selector"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        selector = cls(n_components=data['n_components'], method=data.get('method', 'variance'))
        selector.selected_indices = data['selected_indices']
        print(f"Feature selector loaded from {filepath}")
        return selector

    @classmethod
    def from_predefined_indices(cls, indices_path):
        """Create selector from predefined indices file (.npy)"""
        return cls(indices_path=indices_path)

    @classmethod
    def from_official(cls, language='spanish'):
        """Create selector using official predefined indices

        Args:
            language: 'spanish' (101 dims) or 'english' (103 dims)

        Returns:
            FeatureSelector with official indices loaded
        """
        if language == 'spanish':
            indices = OFFICIAL_SPANISH_WINNERS
        else:
            indices = OFFICIAL_ENGLISH_WINNERS

        selector = cls(n_components=len(indices), method='predefined')
        selector.selected_indices = np.array(indices, dtype=np.int64)
        print(f"Using official {language} indices ({len(indices)} dimensions)")
        return selector


def extract_winners(features, language='spanish', selector=None):
    """
    Extract selected dimensions for a language
    
    Args:
        features: numpy array or torch tensor (N, 1024) or (1024,)
        language: 'spanish' (101 dims) or 'english' (103 dims)
        selector: Pre-fitted FeatureSelector (optional)
        
    Returns:
        Selected features
    """
    n_dims = 101 if language == 'spanish' else 103
    
    if selector is None:
        raise ValueError("Must provide a fitted selector")
    
    return selector.transform(features)


if __name__ == '__main__':
    # Test feature selection
    print("Testing Feature Selection:")
    
    # Create dummy features
    features = np.random.randn(100, 1024)
    
    # Spanish selector (101 dims)
    selector_es = FeatureSelector(n_components=101)
    selector_es.fit(features)
    selected_es = selector_es.transform(features)
    print(f"\nSpanish selection: {features.shape} -> {selected_es.shape}")
    
    # Test with torch tensor
    features_torch = torch.randn(10, 1024)
    selected_torch = selector_es.transform(features_torch)
    print(f"Torch tensor selection: {features_torch.shape} -> {selected_torch.shape}")


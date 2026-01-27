"""
PCA (Principal Component Analysis)
Suitable for: Dimensionality reduction, feature extraction, data visualization

Best for:
- Reducing feature dimensions
- Data visualization (2D/3D)
- Noise reduction
- Speeding up training
- Feature extraction

Pros:
- Reduces overfitting (fewer features)
- Faster training
- Removes multicollinearity
- Good for visualization
- Unsupervised (no labels needed)

Cons:
- Loses interpretability (new features are combinations)
- Assumes linear relationships
- Sensitive to feature scaling
- May lose some information
"""

from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt


class PCAModel:
    """
    PCA Model for dimensionality reduction
    """
    
    def __init__(self, config=None):
        """
        Initialize PCA model
        
        Args:
            config: Model configuration dict
        """
        self.config = config or {
            'n_components': None,  # None = keep all, int = keep n, float = variance ratio
            'random_state': 42
        }
        self.model = None
        self.explained_variance_ratio = None
        self.cumulative_variance_ratio = None
        
    def build_model(self):
        """Build PCA model"""
        self.model = PCA(
            n_components=self.config.get('n_components'),
            random_state=self.config.get('random_state', 42)
        )
        return self.model
    
    def find_optimal_components(self, X, variance_threshold=0.95):
        """
        Find optimal number of components to retain variance threshold
        
        Args:
            X: Training data
            variance_threshold: Cumulative variance to retain (0.95 = 95%)
        """
        # Fit PCA with all components
        pca_full = PCA(random_state=self.config.get('random_state', 42))
        pca_full.fit(X)
        
        # Calculate cumulative variance
        cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
        
        # Find number of components
        n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
        
        print(f"\nOptimal components to retain {variance_threshold*100}% variance: {n_components}")
        print(f"Original features: {X.shape[1]}")
        print(f"Reduction: {X.shape[1] - n_components} features ({round((1 - n_components/X.shape[1])*100, 2)}%)")
        
        return n_components
    
    def train(self, X_train, find_optimal=False, variance_threshold=0.95):
        """
        Fit PCA model
        
        Args:
            X_train: Training features
            find_optimal: Whether to find optimal components
            variance_threshold: Variance threshold if finding optimal
        """
        if find_optimal:
            optimal_components = self.find_optimal_components(X_train, variance_threshold)
            self.config['n_components'] = optimal_components
        
        if self.model is None:
            self.build_model()
        
        self.model.fit(X_train)
        
        # Store variance information
        self.explained_variance_ratio = self.model.explained_variance_ratio_
        self.cumulative_variance_ratio = np.cumsum(self.explained_variance_ratio)
        
        print(f"\nPCA fitted with {self.model.n_components_} components")
        print(f"Explained variance: {round(self.cumulative_variance_ratio[-1]*100, 2)}%")
        
        return self.model
    
    def transform(self, X):
        """Transform data to principal components"""
        return self.model.transform(X)
    
    def inverse_transform(self, X_transformed):
        """Transform back to original feature space"""
        return self.model.inverse_transform(X_transformed)
    
    def evaluate(self, X):
        """
        Evaluate PCA transformation
        
        Measures reconstruction error
        """
        X_transformed = self.transform(X)
        X_reconstructed = self.inverse_transform(X_transformed)
        
        reconstruction_error = mean_squared_error(X, X_reconstructed)
        
        results = {
            'n_components': self.model.n_components_,
            'original_features': X.shape[1],
            'explained_variance_ratio': self.explained_variance_ratio.tolist(),
            'cumulative_variance_ratio': self.cumulative_variance_ratio.tolist(),
            'total_variance_explained': float(self.cumulative_variance_ratio[-1]),
            'reconstruction_error': float(reconstruction_error)
        }
        
        return results
    
    def get_component_loadings(self, feature_names=None):
        """
        Get component loadings (how much each original feature contributes)
        
        Args:
            feature_names: List of original feature names
        """
        loadings = self.model.components_
        
        if feature_names:
            # Create DataFrame-like structure
            component_info = {}
            for i, component in enumerate(loadings):
                component_info[f'PC{i+1}'] = dict(zip(feature_names, component))
            return component_info
        
        return loadings
    
    def plot_variance_explained(self, save_path=None):
        """Plot explained variance ratio"""
        try:
            plt.figure(figsize=(10, 6))
            
            # Bar plot
            plt.subplot(1, 2, 1)
            plt.bar(range(1, len(self.explained_variance_ratio) + 1),
                   self.explained_variance_ratio)
            plt.xlabel('Principal Component')
            plt.ylabel('Variance Explained Ratio')
            plt.title('Variance Explained by Each Component')
            
            # Cumulative plot
            plt.subplot(1, 2, 2)
            plt.plot(range(1, len(self.cumulative_variance_ratio) + 1),
                    self.cumulative_variance_ratio, 'bo-')
            plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
            plt.xlabel('Number of Components')
            plt.ylabel('Cumulative Variance Explained')
            plt.title('Cumulative Variance Explained')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()
        except Exception as e:
            print(f"Could not plot: {str(e)}")
    
    def save_model(self, filepath):
        """Save model to file"""
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save config and metrics
        config_data = {
            'config': self.config,
            'explained_variance_ratio': self.explained_variance_ratio.tolist() if self.explained_variance_ratio is not None else None,
            'cumulative_variance_ratio': self.cumulative_variance_ratio.tolist() if self.cumulative_variance_ratio is not None else None
        }
        with open(f"{filepath}_config.json", 'w') as f:
            json.dump(config_data, f)
    
    @classmethod
    def load_model(cls, filepath):
        """Load model from file"""
        # Load model
        with open(f"{filepath}.pkl", 'rb') as f:
            model = pickle.load(f)
        
        # Load config
        with open(f"{filepath}_config.json", 'r') as f:
            config_data = json.load(f)
        
        # Create instance
        instance = cls(config=config_data['config'])
        instance.model = model
        instance.explained_variance_ratio = np.array(config_data.get('explained_variance_ratio', []))
        instance.cumulative_variance_ratio = np.array(config_data.get('cumulative_variance_ratio', []))
        
        return instance


if __name__ == "__main__":
    print("PCA Model for Dimensionality Reduction")
    print("Use this model to reduce feature dimensions")

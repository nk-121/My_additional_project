"""
K-Means Clustering
Suitable for: Unsupervised learning, customer segmentation, pattern discovery

Best for:
- Customer segmentation
- Pattern discovery
- Data exploration
- Anomaly detection

Pros:
- Simple and fast
- Scales well to large datasets
- Works well with spherical clusters

Cons:
- Need to specify number of clusters (k)
- Sensitive to initial centroids
- Assumes clusters are spherical
- Sensitive to outliers
"""

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt


class KMeansModel:
    """
    K-Means Clustering Model
    """
    
    def __init__(self, config=None):
        """
        Initialize K-Means model
        
        Args:
            config: Model configuration dict
        """
        self.config = config or {
            'n_clusters': 3,
            'max_iter': 300,
            'n_init': 10,
            'random_state': 42
        }
        self.model = None
        self.inertia_values = []
        self.silhouette_scores = []
        
    def build_model(self):
        """Build K-Means model"""
        self.model = KMeans(
            n_clusters=self.config.get('n_clusters', 3),
            max_iter=self.config.get('max_iter', 300),
            n_init=self.config.get('n_init', 10),
            random_state=self.config.get('random_state', 42)
        )
        return self.model
    
    def find_optimal_k(self, X, k_range=range(2, 11)):
        """
        Find optimal number of clusters using elbow method and silhouette score
        
        Args:
            X: Training data
            k_range: Range of k values to test
        """
        self.inertia_values = []
        self.silhouette_scores = []
        
        for k in k_range:
            kmeans = KMeans(
                n_clusters=k,
                max_iter=self.config.get('max_iter', 300),
                n_init=self.config.get('n_init', 10),
                random_state=self.config.get('random_state', 42)
            )
            kmeans.fit(X)
            
            self.inertia_values.append(kmeans.inertia_)
            self.silhouette_scores.append(silhouette_score(X, kmeans.labels_))
        
        # Find optimal k (highest silhouette score)
        optimal_k = k_range[np.argmax(self.silhouette_scores)]
        
        print(f"\nOptimal k based on silhouette score: {optimal_k}")
        print(f"Silhouette scores: {dict(zip(k_range, self.silhouette_scores))}")
        
        return optimal_k
    
    def train(self, X_train, find_optimal=False, k_range=range(2, 11)):
        """
        Train K-Means model
        
        Args:
            X_train: Training features
            find_optimal: Whether to find optimal k
            k_range: Range of k values to test
        """
        if find_optimal:
            optimal_k = self.find_optimal_k(X_train, k_range)
            self.config['n_clusters'] = optimal_k
        
        if self.model is None:
            self.build_model()
        
        self.model.fit(X_train)
        
        return self.model
    
    def predict(self, X):
        """Predict cluster labels"""
        return self.model.predict(X)
    
    def evaluate(self, X):
        """
        Evaluate clustering performance
        
        Note: For unsupervised learning, we use internal metrics
        """
        labels = self.predict(X)
        
        results = {
            'n_clusters': self.config['n_clusters'],
            'inertia': float(self.model.inertia_),
            'silhouette_score': float(silhouette_score(X, labels)),
            'davies_bouldin_score': float(davies_bouldin_score(X, labels)),
            'calinski_harabasz_score': float(calinski_harabasz_score(X, labels)),
            'cluster_sizes': {int(i): int(np.sum(labels == i)) for i in range(self.config['n_clusters'])}
        }
        
        return results
    
    def get_cluster_centers(self):
        """Get cluster centroids"""
        return self.model.cluster_centers_
    
    def save_model(self, filepath):
        """Save model to file"""
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save config and metrics
        config_data = {
            'config': self.config,
            'inertia_values': self.inertia_values,
            'silhouette_scores': self.silhouette_scores
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
        instance.inertia_values = config_data.get('inertia_values', [])
        instance.silhouette_scores = config_data.get('silhouette_scores', [])
        
        return instance


if __name__ == "__main__":
    print("K-Means Clustering Model")
    print("Use this model for unsupervised clustering tasks")

"""
Decision Tree Classifier
Suitable for: Classification tasks with interpretable decision rules

Best for:
- Interpretable models (can visualize decision paths)
- Non-linear relationships
- Mixed feature types
- Feature importance analysis

Pros:
- Easy to understand and interpret
- Requires little data preprocessing
- Can handle both numerical and categorical data

Cons:
- Prone to overfitting
- Can be unstable (small changes in data = different tree)
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import json
import numpy as np


class DecisionTreeModel:
    """
    Decision Tree Classifier with hyperparameter tuning
    """
    
    def __init__(self, config=None):
        """
        Initialize Decision Tree model
        
        Args:
            config: Model configuration dict
        """
        self.config = config or {
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'criterion': 'gini',
            'random_state': 42
        }
        self.model = None
        self.best_params = None
        
    def build_model(self):
        """Build Decision Tree model"""
        self.model = DecisionTreeClassifier(
            max_depth=self.config.get('max_depth', 10),
            min_samples_split=self.config.get('min_samples_split', 5),
            min_samples_leaf=self.config.get('min_samples_leaf', 2),
            criterion=self.config.get('criterion', 'gini'),
            random_state=self.config.get('random_state', 42)
        )
        return self.model
    
    def train(self, X_train, y_train, tune_hyperparameters=False):
        """
        Train Decision Tree model
        
        Args:
            X_train: Training features
            y_train: Training labels
            tune_hyperparameters: Whether to perform grid search
        """
        if tune_hyperparameters:
            # Hyperparameter tuning
            param_grid = {
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy']
            }
            
            grid_search = GridSearchCV(
                DecisionTreeClassifier(random_state=self.config.get('random_state', 42)),
                param_grid,
                cv=5,
                scoring='accuracy',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            print(f"Best parameters: {self.best_params}")
        else:
            if self.model is None:
                self.build_model()
            self.model.fit(X_train, y_train)
        
        return self.model
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.predict(X_test)
        
        results = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'feature_importance': self.model.feature_importances_.tolist() if hasattr(self.model, 'feature_importances_') else None
        }
        
        return results
    
    def get_feature_importance(self, feature_names=None):
        """Get feature importance scores"""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            if feature_names:
                return dict(zip(feature_names, importance))
            return importance
        return None
    
    def save_model(self, filepath):
        """Save model to file"""
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save config
        config_data = {
            'config': self.config,
            'best_params': self.best_params
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
        instance.best_params = config_data.get('best_params')
        
        return instance


if __name__ == "__main__":
    print("Decision Tree Classifier")
    print("Use this model for interpretable classification tasks")

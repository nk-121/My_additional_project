"""
SVM (Support Vector Machine)
Suitable for: Binary and multiclass classification

Best for:
- Binary classification
- High-dimensional data
- Clear margin of separation
- Small to medium datasets
- Text classification

Pros:
- Effective in high dimensions
- Works well with clear margin
- Memory efficient (uses subset of training points)
- Versatile (different kernel functions)

Cons:
- Not suitable for large datasets (slow training)
- Doesn't work well with noisy data
- Doesn't directly provide probability estimates
- Sensitive to feature scaling
- Difficult to interpret
"""

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import json
import numpy as np


class SVMModel:
    """
    Support Vector Machine Classifier
    """
    
    def __init__(self, config=None):
        """
        Initialize SVM model
        
        Args:
            config: Model configuration dict
        """
        self.config = config or {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale',
            'probability': True,
            'random_state': 42
        }
        self.model = None
        self.best_params = None
        
    def build_model(self):
        """Build SVM model"""
        self.model = SVC(
            C=self.config.get('C', 1.0),
            kernel=self.config.get('kernel', 'rbf'),
            gamma=self.config.get('gamma', 'scale'),
            probability=self.config.get('probability', True),
            random_state=self.config.get('random_state', 42)
        )
        return self.model
    
    def train(self, X_train, y_train, tune_hyperparameters=False):
        """
        Train SVM model
        
        Args:
            X_train: Training features
            y_train: Training labels
            tune_hyperparameters: Whether to perform grid search
        """
        if tune_hyperparameters:
            # Hyperparameter tuning
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto']
            }
            
            grid_search = GridSearchCV(
                SVC(
                    probability=self.config.get('probability', True),
                    random_state=self.config.get('random_state', 42)
                ),
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
        if self.config.get('probability', True):
            return self.model.predict_proba(X)
        else:
            raise ValueError("Model was not trained with probability=True")
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.predict(X_test)
        
        results = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'n_support_vectors': self.model.n_support_.tolist() if hasattr(self.model, 'n_support_') else None
        }
        
        return results
    
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
    print("Support Vector Machine Classifier")
    print("Use this model for binary and multiclass classification")

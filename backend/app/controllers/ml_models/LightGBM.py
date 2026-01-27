"""
LightGBM (Light Gradient Boosting Machine)
Suitable for: Fast gradient boosting, large datasets, high performance

Best for:
- Large datasets (handles millions of rows efficiently)
- High-dimensional data
- Fast training required
- Competitions and production systems

Pros:
- Very fast training speed
- Low memory usage
- High accuracy
- Handles categorical features natively
- Built-in feature importance

Cons:
- Can overfit on small datasets
- Requires careful hyperparameter tuning
- May need more data than simpler models
"""

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import json
import numpy as np


class LightGBMModel:
    """
    LightGBM Classifier for fast gradient boosting
    """
    
    def __init__(self, config=None):
        """
        Initialize LightGBM model
        
        Args:
            config: Model configuration dict
        """
        self.config = config or {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'n_estimators': 100,
            'max_depth': -1,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbose': -1
        }
        self.model = None
        self.best_params = None
        
    def build_model(self):
        """Build LightGBM model"""
        self.model = lgb.LGBMClassifier(
            objective=self.config.get('objective', 'binary'),
            boosting_type=self.config.get('boosting_type', 'gbdt'),
            num_leaves=self.config.get('num_leaves', 31),
            learning_rate=self.config.get('learning_rate', 0.05),
            n_estimators=self.config.get('n_estimators', 100),
            max_depth=self.config.get('max_depth', -1),
            min_child_samples=self.config.get('min_child_samples', 20),
            subsample=self.config.get('subsample', 0.8),
            colsample_bytree=self.config.get('colsample_bytree', 0.8),
            random_state=self.config.get('random_state', 42),
            verbose=self.config.get('verbose', -1)
        )
        return self.model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, early_stopping_rounds=50):
        """
        Train LightGBM model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            early_stopping_rounds: Early stopping patience
        """
        if self.model is None:
            self.build_model()
        
        # Train with early stopping if validation set provided
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='logloss',
                callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
            )
        else:
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
    
    def get_feature_importance(self, feature_names=None, importance_type='gain'):
        """
        Get feature importance scores
        
        Args:
            feature_names: List of feature names
            importance_type: 'gain' or 'split'
        """
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            if feature_names:
                return dict(zip(feature_names, importance))
            return importance
        return None
    
    def save_model(self, filepath):
        """Save model to file"""
        # Save using LightGBM's native format
        self.model.booster_.save_model(f"{filepath}.txt")
        
        # Also save as pickle
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
    print("LightGBM Classifier")
    print("Use this model for fast gradient boosting on large datasets")

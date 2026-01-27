"""
XGBoost (Extreme Gradient Boosting)
Suitable for: High-performance gradient boosting, competitions

Best for:
- Kaggle competitions (frequently wins)
- Structured/tabular data
- Feature importance analysis
- When accuracy is priority
- Production systems

Pros:
- State-of-the-art accuracy
- Built-in regularization (prevents overfitting)
- Handles missing values
- Parallel processing (fast)
- Feature importance
- Works with sparse data

Cons:
- Can be slow to train on very large datasets
- Requires hyperparameter tuning
- Less interpretable than simple models
- Overkill for simple problems
"""

import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import json
import numpy as np


class XGBoostModel:
    """
    XGBoost Classifier for gradient boosting
    """
    
    def __init__(self, config=None):
        """
        Initialize XGBoost model
        
        Args:
            config: Model configuration dict
        """
        self.config = config or {
            'objective': 'binary:logistic',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'random_state': 42
        }
        self.model = None
        self.best_params = None
        
    def build_model(self):
        """Build XGBoost model"""
        self.model = xgb.XGBClassifier(
            objective=self.config.get('objective', 'binary:logistic'),
            max_depth=self.config.get('max_depth', 6),
            learning_rate=self.config.get('learning_rate', 0.1),
            n_estimators=self.config.get('n_estimators', 100),
            subsample=self.config.get('subsample', 0.8),
            colsample_bytree=self.config.get('colsample_bytree', 0.8),
            gamma=self.config.get('gamma', 0),
            reg_alpha=self.config.get('reg_alpha', 0),
            reg_lambda=self.config.get('reg_lambda', 1),
            random_state=self.config.get('random_state', 42),
            eval_metric='logloss'
        )
        return self.model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, tune_hyperparameters=False, early_stopping_rounds=50):
        """
        Train XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            tune_hyperparameters: Whether to perform grid search
            early_stopping_rounds: Early stopping patience
        """
        if tune_hyperparameters:
            # Hyperparameter tuning
            param_grid = {
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.05, 0.1, 0.3],
                'n_estimators': [50, 100, 200],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0]
            }
            
            grid_search = GridSearchCV(
                xgb.XGBClassifier(
                    random_state=self.config.get('random_state', 42),
                    eval_metric='logloss'
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
            
            # Train with early stopping if validation set provided
            if X_val is not None and y_val is not None:
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=False
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
            importance_type: 'weight', 'gain', or 'cover'
        """
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            if feature_names:
                return dict(zip(feature_names, importance))
            return importance
        return None
    
    def save_model(self, filepath):
        """Save model to file"""
        # Save using XGBoost's native format
        self.model.save_model(f"{filepath}.json")
        
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
    print("XGBoost Classifier")
    print("Use this model for state-of-the-art gradient boosting")

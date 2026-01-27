"""
Random Forest Classifier
Suitable for: Classification with ensemble learning

Best for:
- Robust classification tasks
- Feature importance analysis
- Handling missing values
- Preventing overfitting
- Both categorical and numerical data

Pros:
- Very accurate (ensemble of trees)
- Handles missing values well
- Reduces overfitting compared to single tree
- Provides feature importance
- Works well out-of-the-box

Cons:
- Slower than single tree
- Less interpretable than single tree
- Can be memory intensive
- Prediction slower than linear models
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import json
import numpy as np


class RandomForestModel:
    """
    Random Forest Classifier
    """
    
    def __init__(self, config=None):
        """
        Initialize Random Forest model
        
        Args:
            config: Model configuration dict
        """
        self.config = config or {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1
        }
        self.model = None
        self.best_params = None
        
    def build_model(self):
        """Build Random Forest model"""
        self.model = RandomForestClassifier(
            n_estimators=self.config.get('n_estimators', 100),
            max_depth=self.config.get('max_depth'),
            min_samples_split=self.config.get('min_samples_split', 2),
            min_samples_leaf=self.config.get('min_samples_leaf', 1),
            max_features=self.config.get('max_features', 'sqrt'),
            random_state=self.config.get('random_state', 42),
            n_jobs=self.config.get('n_jobs', -1)
        )
        return self.model
    
    def train(self, X_train, y_train, tune_hyperparameters=False):
        """
        Train Random Forest model
        
        Args:
            X_train: Training features
            y_train: Training labels
            tune_hyperparameters: Whether to perform grid search
        """
        if tune_hyperparameters:
            # Hyperparameter tuning
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
            
            grid_search = GridSearchCV(
                RandomForestClassifier(
                    random_state=self.config.get('random_state', 42),
                    n_jobs=self.config.get('n_jobs', -1)
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
    print("Random Forest Classifier")
    print("Use this model for robust ensemble classification")

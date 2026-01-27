"""
Logistic Regression
Suitable for: Binary and multiclass classification

Best for:
- Binary classification (yes/no, spam/not spam)
- Multiclass classification
- Probability prediction
- Baseline classification model
- When you need interpretable results

Pros:
- Simple and fast
- Outputs probabilities
- Works well for linearly separable data
- Less prone to overfitting
- Good for high-dimensional data

Cons:
- Assumes linear decision boundary
- May underfit complex patterns
- Sensitive to outliers
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import pickle
import json
import numpy as np


class LogisticRegressionModel:
    """
    Logistic Regression Classifier
    """
    
    def __init__(self, config=None):
        """
        Initialize Logistic Regression model
        
        Args:
            config: Model configuration dict
        """
        self.config = config or {
            'penalty': 'l2',
            'C': 1.0,
            'solver': 'lbfgs',
            'max_iter': 1000,
            'multi_class': 'auto',
            'random_state': 42
        }
        self.model = None
        self.best_params = None
        
    def build_model(self):
        """Build Logistic Regression model"""
        self.model = LogisticRegression(
            penalty=self.config.get('penalty', 'l2'),
            C=self.config.get('C', 1.0),
            solver=self.config.get('solver', 'lbfgs'),
            max_iter=self.config.get('max_iter', 1000),
            multi_class=self.config.get('multi_class', 'auto'),
            random_state=self.config.get('random_state', 42)
        )
        return self.model
    
    def train(self, X_train, y_train, tune_hyperparameters=False):
        """
        Train Logistic Regression model
        
        Args:
            X_train: Training features
            y_train: Training labels
            tune_hyperparameters: Whether to perform grid search
        """
        if tune_hyperparameters:
            # Hyperparameter tuning
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
            
            grid_search = GridSearchCV(
                LogisticRegression(
                    max_iter=self.config.get('max_iter', 1000),
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
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)
        
        results = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'coefficients': self.model.coef_.tolist() if hasattr(self.model, 'coef_') else None,
            'intercept': self.model.intercept_.tolist() if hasattr(self.model, 'intercept_') else None
        }
        
        # Add ROC AUC for binary classification
        if len(np.unique(y_test)) == 2:
            results['roc_auc'] = float(roc_auc_score(y_test, y_pred_proba[:, 1]))
        
        return results
    
    def get_coefficients(self, feature_names=None):
        """Get model coefficients"""
        if hasattr(self.model, 'coef_'):
            coef = self.model.coef_[0] if self.model.coef_.shape[0] == 1 else self.model.coef_
            if feature_names:
                return dict(zip(feature_names, coef))
            return coef
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
    print("Logistic Regression Classifier")
    print("Use this model for binary and multiclass classification")

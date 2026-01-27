"""
Linear Regression
Suitable for: Continuous value prediction, relationship modeling

Best for:
- Predicting continuous values (price, temperature, sales)
- Understanding linear relationships
- Fast predictions
- Baseline model

Pros:
- Simple and fast
- Easy to interpret (coefficients show feature impact)
- Works well when relationship is linear
- No hyperparameters to tune

Cons:
- Assumes linear relationship
- Sensitive to outliers
- May underfit complex data
- Assumes features are independent
"""

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import json
import numpy as np


class LinearRegressionModel:
    """
    Linear Regression Model for continuous prediction
    """
    
    def __init__(self, config=None):
        """
        Initialize Linear Regression model
        
        Args:
            config: Model configuration dict
        """
        self.config = config or {
            'model_type': 'linear',  # 'linear', 'ridge', or 'lasso'
            'alpha': 1.0,  # Regularization strength for Ridge/Lasso
            'fit_intercept': True
        }
        self.model = None
        
    def build_model(self):
        """Build Linear Regression model"""
        model_type = self.config.get('model_type', 'linear')
        
        if model_type == 'ridge':
            self.model = Ridge(
                alpha=self.config.get('alpha', 1.0),
                fit_intercept=self.config.get('fit_intercept', True),
                random_state=42
            )
        elif model_type == 'lasso':
            self.model = Lasso(
                alpha=self.config.get('alpha', 1.0),
                fit_intercept=self.config.get('fit_intercept', True),
                random_state=42
            )
        else:
            self.model = LinearRegression(
                fit_intercept=self.config.get('fit_intercept', True)
            )
        
        return self.model
    
    def train(self, X_train, y_train):
        """
        Train Linear Regression model
        
        Args:
            X_train: Training features
            y_train: Training labels (continuous values)
        """
        if self.model is None:
            self.build_model()
        
        self.model.fit(X_train, y_train)
        
        return self.model
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.predict(X_test)
        
        results = {
            'mse': float(mean_squared_error(y_test, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'mae': float(mean_absolute_error(y_test, y_pred)),
            'r2_score': float(r2_score(y_test, y_pred)),
            'coefficients': self.model.coef_.tolist() if hasattr(self.model, 'coef_') else None,
            'intercept': float(self.model.intercept_) if hasattr(self.model, 'intercept_') else None
        }
        
        return results
    
    def get_coefficients(self, feature_names=None):
        """Get model coefficients"""
        if hasattr(self.model, 'coef_'):
            coef = self.model.coef_
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
            'config': self.config
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
        
        return instance


if __name__ == "__main__":
    print("Linear Regression Model")
    print("Use this model for predicting continuous values")

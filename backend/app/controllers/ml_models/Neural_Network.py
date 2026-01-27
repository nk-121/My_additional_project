"""
Neural Network (Deep Learning)
Suitable for: Complex pattern recognition, non-linear relationships

Best for:
- Complex non-linear patterns
- Large datasets
- High-dimensional data
- Feature learning
- When accuracy is top priority

Pros:
- Can learn complex patterns
- Automatic feature learning
- Scales well with data
- Flexible architecture

Cons:
- Requires large datasets
- Computationally expensive
- Black box (less interpretable)
- Prone to overfitting
- Requires careful tuning
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import json


class NeuralNetworkModel:
    """
    Deep Neural Network for classification
    """
    
    def __init__(self, input_dim, num_classes, config=None):
        """
        Initialize Neural Network model
        
        Args:
            input_dim: Number of input features
            num_classes: Number of output classes
            config: Model configuration dict
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.config = config or {
            'hidden_layers': [128, 64, 32],
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'batch_normalization': True
        }
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build Neural Network architecture"""
        model = Sequential()
        
        # Input layer
        model.add(Dense(
            self.config['hidden_layers'][0],
            activation='relu',
            input_dim=self.input_dim
        ))
        
        if self.config.get('batch_normalization', True):
            model.add(BatchNormalization())
        
        model.add(Dropout(self.config['dropout_rate']))
        
        # Hidden layers
        for units in self.config['hidden_layers'][1:]:
            model.add(Dense(units, activation='relu'))
            
            if self.config.get('batch_normalization', True):
                model.add(BatchNormalization())
            
            model.add(Dropout(self.config['dropout_rate']))
        
        # Output layer
        if self.num_classes == 2:
            model.add(Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
        else:
            model.add(Dense(self.num_classes, activation='softmax'))
            loss = 'sparse_categorical_crossentropy'
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss=loss,
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32):
        """
        Train the Neural Network
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        if self.model is None:
            self.build_model()
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def predict(self, X):
        """Make predictions"""
        predictions = self.model.predict(X)
        
        if self.num_classes == 2:
            return (predictions > 0.5).astype(int).flatten()
        else:
            return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.predict(X_test)
        
        results = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        return results
    
    def save_model(self, filepath):
        """Save model to file"""
        self.model.save(f"{filepath}.h5")
        
        # Save config
        config_data = {
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'config': self.config
        }
        with open(f"{filepath}_config.json", 'w') as f:
            json.dump(config_data, f)
    
    @classmethod
    def load_model(cls, filepath):
        """Load model from file"""
        # Load config
        with open(f"{filepath}_config.json", 'r') as f:
            config_data = json.load(f)
        
        # Create instance
        instance = cls(
            input_dim=config_data['input_dim'],
            num_classes=config_data['num_classes'],
            config=config_data['config']
        )
        
        # Load model
        instance.model = keras.models.load_model(f"{filepath}.h5")
        
        return instance


if __name__ == "__main__":
    print("Neural Network Model")
    print("Use this model for complex pattern recognition")

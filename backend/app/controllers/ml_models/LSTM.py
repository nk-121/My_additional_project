"""
LSTM (Long Short-Term Memory)
Suitable for: Sequential data, time series, text data

Best for:
- Time series forecasting
- Text classification
- Sequential pattern recognition
- Speech recognition
- Any data with temporal dependencies

Pros:
- Handles sequential dependencies
- Can learn long-term patterns
- Works with variable-length sequences
- Avoids vanishing gradient problem

Cons:
- Computationally expensive
- Requires more data
- Slower training than simpler models
- Can be difficult to tune
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
import pickle
import json


class LSTMModel:
    """
    LSTM Model for sequential data classification/regression
    """
    
    def __init__(self, input_shape, output_shape, task='classification', config=None):
        """
        Initialize LSTM model
        
        Args:
            input_shape: Shape of input data (timesteps, features)
            output_shape: Number of output units
            task: 'classification' or 'regression'
            config: Model configuration dict
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.task = task
        self.config = config or {
            'lstm_units': [128, 64],
            'dropout_rate': 0.3,
            'bidirectional': False,
            'learning_rate': 0.001
        }
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build LSTM architecture"""
        model = Sequential()
        
        # First LSTM layer
        lstm_layer = LSTM(
            self.config['lstm_units'][0],
            return_sequences=len(self.config['lstm_units']) > 1,
            input_shape=self.input_shape
        )
        
        if self.config.get('bidirectional', False):
            model.add(Bidirectional(lstm_layer))
        else:
            model.add(lstm_layer)
        
        model.add(Dropout(self.config['dropout_rate']))
        
        # Additional LSTM layers
        for i, units in enumerate(self.config['lstm_units'][1:]):
            is_last = (i == len(self.config['lstm_units'][1:]) - 1)
            lstm_layer = LSTM(units, return_sequences=not is_last)
            
            if self.config.get('bidirectional', False):
                model.add(Bidirectional(lstm_layer))
            else:
                model.add(lstm_layer)
            
            model.add(Dropout(self.config['dropout_rate']))
        
        # Output layer
        if self.task == 'classification':
            if self.output_shape == 2:
                model.add(Dense(1, activation='sigmoid'))
            else:
                model.add(Dense(self.output_shape, activation='softmax'))
        else:
            model.add(Dense(self.output_shape, activation='linear'))
        
        # Compile model
        if self.task == 'classification':
            loss = 'binary_crossentropy' if self.output_shape == 2 else 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        else:
            loss = 'mse'
            metrics = ['mae']
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss=loss,
            metrics=metrics
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32):
        """
        Train the LSTM model
        
        Args:
            X_train: Training features (samples, timesteps, features)
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
        
        if self.task == 'classification':
            if self.output_shape == 2:
                return (predictions > 0.5).astype(int).flatten()
            else:
                return np.argmax(predictions, axis=1)
        else:
            return predictions.flatten()
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.predict(X_test)
        
        if self.task == 'classification':
            results = {
                'accuracy': float(np.mean(y_pred == y_test)),
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
        else:
            results = {
                'mse': float(mean_squared_error(y_test, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
                'r2_score': float(r2_score(y_test, y_pred))
            }
        
        return results
    
    def save_model(self, filepath):
        """Save model to file"""
        self.model.save(f"{filepath}.h5")
        
        # Save config
        config_data = {
            'input_shape': self.input_shape,
            'output_shape': self.output_shape,
            'task': self.task,
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
            input_shape=tuple(config_data['input_shape']),
            output_shape=config_data['output_shape'],
            task=config_data['task'],
            config=config_data['config']
        )
        
        # Load model
        instance.model = keras.models.load_model(f"{filepath}.h5")
        
        return instance


if __name__ == "__main__":
    print("LSTM Model for Sequential Data")
    print("Use this model for time series and sequential pattern recognition")

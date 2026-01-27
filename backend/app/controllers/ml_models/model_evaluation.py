"""
Model Evaluation Script
Comprehensive evaluation and comparison of all ML models

This script provides:
- Model training and evaluation
- Performance comparison
- Visual analytics
- Model selection recommendations
- Export results to JSON/CSV
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import json
import time
import warnings
warnings.filterwarnings('ignore')

# Import all model classes
from Decision_Tree import DecisionTreeModel
from K_Means import KMeansModel
from LightGBM import LightGBMModel
from Linear_Regression import LinearRegressionModel
from Logistic_Regression import LogisticRegressionModel
from LSTM import LSTMModel
from Neural_Network import NeuralNetworkModel
from PCA import PCAModel
from Random_Forest import RandomForestModel
from SVM import SVMModel
from XGBoost import XGBoostModel


class ModelEvaluator:
    """
    Comprehensive Model Evaluation and Comparison
    """
    
    def __init__(self, task='classification'):
        """
        Initialize evaluator
        
        Args:
            task: 'classification', 'regression', or 'clustering'
        """
        self.task = task
        self.results = {}
        self.models = {}
        self.best_model = None
        
    def prepare_data(self, X, y=None, test_size=0.2, val_size=0.1, random_state=42):
        """
        Split data into train/validation/test sets
        
        Args:
            X: Features
            y: Labels (None for unsupervised)
            test_size: Test set proportion
            val_size: Validation set proportion
            random_state: Random seed
        
        Returns:
            Dictionary with train/val/test splits
        """
        if y is None:
            # Unsupervised (clustering)
            return {'X_train': X, 'y_train': None}
        
        # Split train and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if self.task == 'classification' else None
        )
        
        # Split train and validation
        if val_size > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val,
                test_size=val_size/(1-test_size),
                random_state=random_state,
                stratify=y_train_val if self.task == 'classification' else None
            )
        else:
            X_train, y_train = X_train_val, y_train_val
            X_val, y_val = None, None
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test
        }
    
    def train_and_evaluate_model(self, model_name, model, data, **train_kwargs):
        """
        Train and evaluate a single model
        
        Args:
            model_name: Name of the model
            model: Model instance
            data: Dictionary with train/val/test splits
            **train_kwargs: Additional training parameters
        
        Returns:
            Dictionary with evaluation results
        """
        print(f"\n{'='*60}")
        print(f"Training {model_name}...")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Train model
            if model_name == 'K-Means':
                # Unsupervised
                model.train(data['X_train'], **train_kwargs)
                eval_results = model.evaluate(data['X_train'])
                training_time = time.time() - start_time
                
                result = {
                    'model_name': model_name,
                    'training_time': training_time,
                    'metrics': eval_results,
                    'status': 'success'
                }
            
            elif model_name == 'PCA':
                # Dimensionality reduction
                model.train(data['X_train'], **train_kwargs)
                eval_results = model.evaluate(data['X_train'])
                training_time = time.time() - start_time
                
                result = {
                    'model_name': model_name,
                    'training_time': training_time,
                    'metrics': eval_results,
                    'status': 'success'
                }
            
            elif model_name in ['LSTM', 'Neural_Network']:
                # Deep learning models
                if data['X_val'] is not None:
                    model.train(
                        data['X_train'], data['y_train'],
                        data['X_val'], data['y_val'],
                        **train_kwargs
                    )
                else:
                    model.train(data['X_train'], data['y_train'], **train_kwargs)
                
                training_time = time.time() - start_time
                
                # Evaluate on test set
                eval_results = model.evaluate(data['X_test'], data['y_test'])
                
                result = {
                    'model_name': model_name,
                    'training_time': training_time,
                    'test_metrics': eval_results,
                    'status': 'success'
                }
            
            else:
                # Standard ML models
                if data['X_val'] is not None and hasattr(model, 'train'):
                    # Check if model supports validation
                    if model_name in ['LightGBM', 'XGBoost']:
                        model.train(
                            data['X_train'], data['y_train'],
                            data['X_val'], data['y_val'],
                            **train_kwargs
                        )
                    else:
                        model.train(data['X_train'], data['y_train'], **train_kwargs)
                else:
                    model.train(data['X_train'], data['y_train'], **train_kwargs)
                
                training_time = time.time() - start_time
                
                # Evaluate on test set
                eval_results = model.evaluate(data['X_test'], data['y_test'])
                
                # Cross-validation score
                cv_score = None
                if self.task == 'classification':
                    try:
                        cv_scores = cross_val_score(
                            model.model, data['X_train'], data['y_train'],
                            cv=5, scoring='accuracy'
                        )
                        cv_score = {
                            'mean': float(cv_scores.mean()),
                            'std': float(cv_scores.std())
                        }
                    except:
                        pass
                
                result = {
                    'model_name': model_name,
                    'training_time': training_time,
                    'test_metrics': eval_results,
                    'cv_score': cv_score,
                    'status': 'success'
                }
            
            print(f"✓ {model_name} trained successfully in {training_time:.2f}s")
            
            # Store model
            self.models[model_name] = model
            
            return result
        
        except Exception as e:
            print(f"✗ Error training {model_name}: {str(e)}")
            return {
                'model_name': model_name,
                'status': 'failed',
                'error': str(e)
            }
    
    def evaluate_all_models(self, X, y=None, models_to_evaluate=None, **kwargs):
        """
        Evaluate all specified models
        
        Args:
            X: Feature matrix
            y: Labels (None for unsupervised)
            models_to_evaluate: List of model names to evaluate (None = all)
            **kwargs: Additional parameters
        
        Returns:
            Dictionary with all results
        """
        # Prepare data
        data = self.prepare_data(X, y, **kwargs.get('data_split', {}))
        
        # Define models to evaluate
        if models_to_evaluate is None:
            if self.task == 'classification':
                models_to_evaluate = [
                    'Decision_Tree', 'Logistic_Regression', 'Random_Forest',
                    'SVM', 'XGBoost', 'LightGBM', 'Neural_Network'
                ]
            elif self.task == 'regression':
                models_to_evaluate = ['Linear_Regression', 'XGBoost', 'LightGBM']
            elif self.task == 'clustering':
                models_to_evaluate = ['K-Means']
        
        # Initialize models
        model_instances = {}
        
        for model_name in models_to_evaluate:
            if model_name == 'Decision_Tree':
                model_instances[model_name] = DecisionTreeModel()
            elif model_name == 'K-Means':
                model_instances[model_name] = KMeansModel()
            elif model_name == 'LightGBM':
                model_instances[model_name] = LightGBMModel()
            elif model_name == 'Linear_Regression':
                model_instances[model_name] = LinearRegressionModel()
            elif model_name == 'Logistic_Regression':
                model_instances[model_name] = LogisticRegressionModel()
            elif model_name == 'Neural_Network':
                num_classes = len(np.unique(y)) if y is not None else 2
                model_instances[model_name] = NeuralNetworkModel(
                    input_dim=X.shape[1],
                    num_classes=num_classes
                )
            elif model_name == 'PCA':
                model_instances[model_name] = PCAModel()
            elif model_name == 'Random_Forest':
                model_instances[model_name] = RandomForestModel()
            elif model_name == 'SVM':
                model_instances[model_name] = SVMModel()
            elif model_name == 'XGBoost':
                model_instances[model_name] = XGBoostModel()
        
        # Train and evaluate each model
        results = []
        for model_name, model in model_instances.items():
            result = self.train_and_evaluate_model(
                model_name, model, data,
                **kwargs.get('train_params', {})
            )
            results.append(result)
            self.results[model_name] = result
        
        return results
    
    def compare_models(self):
        """
        Compare all evaluated models
        
        Returns:
            Pandas DataFrame with comparison
        """
        if not self.results:
            print("No models have been evaluated yet!")
            return None
        
        comparison_data = []
        
        for model_name, result in self.results.items():
            if result['status'] != 'success':
                continue
            
            row = {'Model': model_name}
            
            # Add metrics based on task
            if 'test_metrics' in result:
                metrics = result['test_metrics']
                
                if self.task == 'classification':
                    row['Accuracy'] = metrics.get('accuracy', 0)
                    if 'classification_report' in metrics and '1' in metrics['classification_report']:
                        row['Precision'] = metrics['classification_report']['1']['precision']
                        row['Recall'] = metrics['classification_report']['1']['recall']
                        row['F1-Score'] = metrics['classification_report']['1']['f1-score']
                    if 'roc_auc' in metrics:
                        row['ROC-AUC'] = metrics['roc_auc']
                
                elif self.task == 'regression':
                    row['MSE'] = metrics.get('mse', 0)
                    row['RMSE'] = metrics.get('rmse', 0)
                    row['MAE'] = metrics.get('mae', 0)
                    row['R2'] = metrics.get('r2_score', 0)
            
            elif 'metrics' in result:
                # Clustering or PCA
                metrics = result['metrics']
                if 'silhouette_score' in metrics:
                    row['Silhouette'] = metrics['silhouette_score']
                if 'total_variance_explained' in metrics:
                    row['Variance_Explained'] = metrics['total_variance_explained']
            
            row['Training_Time(s)'] = result.get('training_time', 0)
            
            # CV score
            if 'cv_score' in result and result['cv_score']:
                row['CV_Score'] = result['cv_score']['mean']
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by primary metric
        if self.task == 'classification' and 'Accuracy' in df.columns:
            df = df.sort_values('Accuracy', ascending=False)
        elif self.task == 'regression' and 'R2' in df.columns:
            df = df.sort_values('R2', ascending=False)
        
        return df
    
    def get_best_model(self, metric='accuracy'):
        """
        Get the best performing model
        
        Args:
            metric: Metric to use for comparison
        
        Returns:
            Tuple of (model_name, model_instance, metrics)
        """
        if not self.results:
            return None
        
        best_score = -float('inf')
        best_model_name = None
        
        for model_name, result in self.results.items():
            if result['status'] != 'success':
                continue
            
            if 'test_metrics' in result:
                metrics = result['test_metrics']
                
                if metric == 'accuracy' and 'accuracy' in metrics:
                    score = metrics['accuracy']
                elif metric == 'r2' and 'r2_score' in metrics:
                    score = metrics['r2_score']
                elif metric == 'f1' and 'classification_report' in metrics:
                    score = metrics['classification_report']['weighted avg']['f1-score']
                else:
                    continue
                
                if score > best_score:
                    best_score = score
                    best_model_name = model_name
        
        if best_model_name:
            return (
                best_model_name,
                self.models[best_model_name],
                self.results[best_model_name]
            )
        
        return None
    
    def save_results(self, filepath='evaluation_results'):
        """
        Save evaluation results
        
        Args:
            filepath: Base filepath (without extension)
        """
        # Save as JSON
        with open(f"{filepath}.json", 'w') as f:
            json.dump(self.results, f, indent=4)
        
        # Save comparison as CSV
        df = self.compare_models()
        if df is not None:
            df.to_csv(f"{filepath}_comparison.csv", index=False)
        
        print(f"\nResults saved to:")
        print(f"  - {filepath}.json")
        print(f"  - {filepath}_comparison.csv")
    
    def print_summary(self):
        """Print evaluation summary"""
        print("\n" + "="*80)
        print("MODEL EVALUATION SUMMARY")
        print("="*80)
        
        df = self.compare_models()
        if df is not None:
            print("\n" + df.to_string(index=False))
        
        # Best model
        best = self.get_best_model()
        if best:
            model_name, model, metrics = best
            print(f"\n{'='*80}")
            print(f"🏆 BEST MODEL: {model_name}")
            print(f"{'='*80}")
            
            if 'test_metrics' in metrics:
                test_metrics = metrics['test_metrics']
                if 'accuracy' in test_metrics:
                    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
                if 'r2_score' in test_metrics:
                    print(f"R² Score: {test_metrics['r2_score']:.4f}")


# Example usage
if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║           ML MODEL EVALUATION FRAMEWORK                      ║
╚══════════════════════════════════════════════════════════════╝

Usage Example:
--------------
from model_evaluation import ModelEvaluator
import pandas as pd

# Load your preprocessed data
df = pd.read_csv('preprocessed_data.csv')
X = df.drop('target', axis=1)
y = df['target']

# Initialize evaluator
evaluator = ModelEvaluator(task='classification')

# Evaluate all models
evaluator.evaluate_all_models(X, y)

# Print summary
evaluator.print_summary()

# Save results
evaluator.save_results('my_evaluation')

# Get best model
best_name, best_model, best_metrics = evaluator.get_best_model()
print(f"Best model: {best_name}")

# Save best model
best_model.save_model(f'models/{best_name}_best')
    """)

"""
End-to-End ML Pipeline
Complete workflow: Data Loading → Preprocessing → Model Training → Evaluation → Save Model

This script handles everything automatically:
1. Load raw CSV data
2. Preprocess data (handle missing values, normalize, etc.)
3. Train specified model
4. Evaluate performance
5. Save model and results
6. Generate visualizations

Usage:
    python train_model.py --data data.csv --target target_column --model Linear_Regression
"""
from html import parser
import sys
import pandas as pd
import numpy as np
import argparse
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
from pyparsing import Path
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Import preprocessing
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from ml_models.Linear_Regression import LinearRegressionModel
from ml_models.Logistic_Regression import LogisticRegressionModel
from ml_models.Decision_Tree import DecisionTreeModel
from ml_models.Random_Forest import RandomForestModel
from ml_models.XGBoost import XGBoostModel
from ml_models.LightGBM import LightGBMModel
from ml_models.SVM import SVMModel
from ml_models.Neural_Network import NeuralNetworkModel
from ml_models.K_Means import KMeansModel
from ml_models.PCA import PCAModel

class MLPipeline:
    """
    Complete ML Pipeline from raw data to trained model
    """
    
    def __init__(self, model_name='Linear_Regression', task='regression'):
        """
        Initialize pipeline
        
        Args:
            model_name: Name of model to train
            task: 'classification', 'regression', or 'clustering'
        """
        self.model_name = model_name
        self.task = task
        self.model = None
        self.preprocessor = None
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self, filepath, target_column=None):
        """
        Load data from CSV
        
        Args:
            filepath: Path to CSV file
            target_column: Name of target column (None for clustering)
        
        Returns:
            X, y (features and labels)
        """
        print(f"\n{'='*80}")
        print(f"STEP 1: LOADING DATA")
        print(f"{'='*80}")
        
        df = pd.read_csv(filepath)
        print(f"✓ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Display basic info
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nFirst few rows:")
        print(df.head())
        
        # Separate features and target
        if target_column:
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in data!")
            
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            print(f"\n✓ Features: {X.shape[1]} columns")
            print(f"✓ Target: {target_column}")
            print(f"✓ Target distribution:\n{y.value_counts()}")
            
            return X, y
        else:
            # No target (clustering)
            print(f"\n✓ No target column (unsupervised learning)")
            return df, None
    
    def preprocess_data(self, X, y=None, missing_threshold=0.3, test_size=0.2):
        """
        Preprocess data using our preprocessing pipeline
        
        Args:
            X: Features
            y: Labels
            missing_threshold: Threshold for dropping columns
            test_size: Test set size
        
        Returns:
            Preprocessed train/test splits
        """
        print(f"\n{'='*80}")
        print(f"STEP 2: PREPROCESSING DATA")
        print(f"{'='*80}")
        
        # Import preprocessing components
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.impute import SimpleImputer
        
        # Create a copy
        df = X.copy()
        if y is not None:
            df['__target__'] = y
        
        # 1. Handle missing values
        print("\n1. Handling missing values...")
        missing_before = df.isnull().sum().sum()
        print(f"   Missing values before: {missing_before}")
        
        # Drop columns with too many missing values
        cols_to_drop = []
        for col in df.columns:
            if col == '__target__':
                continue
            missing_pct = df[col].isnull().sum() / len(df)
            if missing_pct > missing_threshold:
                cols_to_drop.append(col)
                print(f"   Dropping {col}: {missing_pct*100:.2f}% missing")
        
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
        
        # Fill remaining missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        if '__target__' in numeric_cols:
            numeric_cols.remove('__target__')
        if '__target__' in categorical_cols:
            categorical_cols.remove('__target__')
        
        # Fill numeric with median
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical with mode
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
        
        missing_after = df.isnull().sum().sum()
        print(f"   Missing values after: {missing_after}")
        print(f"   ✓ Filled {missing_before - missing_after} missing values")
        
        # 2. Remove duplicates
        print("\n2. Removing duplicates...")
        before_dup = len(df)
        df = df.drop_duplicates()
        after_dup = len(df)
        print(f"   ✓ Removed {before_dup - after_dup} duplicate rows")
        
        # 3. Encode categorical variables
        print("\n3. Encoding categorical variables...")
        label_encoders = {}
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le
                print(f"   Encoded {col}: {len(le.classes_)} unique values")
        
        # 4. Normalize numeric features
        print("\n4. Normalizing numeric features...")
        scaler = StandardScaler()
        numeric_cols = [col for col in numeric_cols if col in df.columns]
        if numeric_cols:
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            print(f"   ✓ Normalized {len(numeric_cols)} numeric columns")
        
        # Separate features and target
        if y is not None:
            y_processed = df['__target__']
            X_processed = df.drop(columns=['__target__'])
        else:
            X_processed = df
            y_processed = None
        
        print(f"\n✓ Preprocessing complete!")
        print(f"   Final shape: {X_processed.shape}")
        
        # Split data
        if y_processed is not None:
            print(f"\n5. Splitting data (test_size={test_size})...")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_processed, y_processed,
                test_size=test_size,
                random_state=42,
                stratify=y_processed if self.task == 'classification' else None
            )
            print(f"   Train set: {self.X_train.shape}")
            print(f"   Test set: {self.X_test.shape}")
        else:
            self.X_train = X_processed
            self.y_train = None
            self.X_test = None
            self.y_test = None
        
        # Store preprocessing objects
        self.scaler = scaler
        self.label_encoders = label_encoders
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def build_and_train_model(self, **model_params):
        """
        Build and train the specified model
        
        Args:
            **model_params: Additional model parameters
        """
        print(f"\n{'='*80}")
        print(f"STEP 3: TRAINING {self.model_name.upper()}")
        print(f"{'='*80}")
        
        # Initialize model based on name
        if self.model_name == 'Linear_Regression':
            self.model = LinearRegressionModel()
            print("✓ Linear Regression initialized")
            print("   Best for: Predicting continuous values")
            
        elif self.model_name == 'Logistic_Regression':
            self.model = LogisticRegressionModel()
            print("✓ Logistic Regression initialized")
            print("   Best for: Binary/multiclass classification")
            
        elif self.model_name == 'Decision_Tree':
            self.model = DecisionTreeModel()
            print("✓ Decision Tree initialized")
            print("   Best for: Interpretable classification")
            
        elif self.model_name == 'Random_Forest':
            self.model = RandomForestModel()
            print("✓ Random Forest initialized")
            print("   Best for: Robust classification")
            
        elif self.model_name == 'XGBoost':
            self.model = XGBoostModel()
            print("✓ XGBoost initialized")
            print("   Best for: High-performance classification")
            
        elif self.model_name == 'LightGBM':
            self.model = LightGBMModel()
            print("✓ LightGBM initialized")
            print("   Best for: Fast training on large datasets")
            
        elif self.model_name == 'SVM':
            self.model = SVMModel()
            print("✓ SVM initialized")
            print("   Best for: High-dimensional classification")
            
        elif self.model_name == 'Neural_Network':
            num_classes = len(np.unique(self.y_train)) if self.y_train is not None else 2
            self.model = NeuralNetworkModel(
                input_dim=self.X_train.shape[1],
                num_classes=num_classes
            )
            print("✓ Neural Network initialized")
            print("   Best for: Complex pattern recognition")
            
        elif self.model_name == 'K-Means':
            self.model = KMeansModel()
            print("✓ K-Means initialized")
            print("   Best for: Customer segmentation")
            
        elif self.model_name == 'PCA':
            self.model = PCAModel()
            print("✓ PCA initialized")
            print("   Best for: Dimensionality reduction")
            
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        # Train model
        print(f"\nTraining {self.model_name}...")
        print(f"Training samples: {len(self.X_train)}")
        
        import time
        start_time = time.time()
        
        if self.model_name in ['Neural_Network', 'LSTM']:
            # Deep learning models need validation set
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                self.X_train, self.y_train,
                test_size=0.1,
                random_state=42
            )
            self.model.train(X_train_split, y_train_split, X_val, y_val, epochs=50)
        elif self.model_name in ['XGBoost', 'LightGBM']:
            # Gradient boosting with validation
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                self.X_train, self.y_train,
                test_size=0.1,
                random_state=42
            )
            self.model.train(X_train_split, y_train_split, X_val, y_val)
        elif self.model_name in ['K-Means', 'PCA']:
            # Unsupervised
            self.model.train(self.X_train, **model_params)
        else:
            # Standard ML models
            self.model.train(self.X_train, self.y_train, **model_params)
        
        training_time = time.time() - start_time
        
        print(f"✓ Training complete in {training_time:.2f} seconds")
        
        self.results['training_time'] = training_time
        
        return self.model
    
    def evaluate_model(self):
        """
        Evaluate trained model
        """
        print(f"\n{'='*80}")
        print(f"STEP 4: EVALUATING MODEL")
        print(f"{'='*80}")
        
        if self.model_name in ['K-Means', 'PCA']:
            # Unsupervised evaluation
            eval_results = self.model.evaluate(self.X_train)
        else:
            # Supervised evaluation
            eval_results = self.model.evaluate(self.X_test, self.y_test)
        
        self.results['evaluation'] = eval_results
        
        # Print results
        print(f"\n📊 EVALUATION RESULTS:")
        print(f"{'='*80}")
        
        if self.task == 'regression':
            print(f"Mean Squared Error (MSE):  {eval_results.get('mse', 0):.4f}")
            print(f"Root MSE (RMSE):           {eval_results.get('rmse', 0):.4f}")
            print(f"Mean Absolute Error (MAE): {eval_results.get('mae', 0):.4f}")
            print(f"R² Score:                  {eval_results.get('r2_score', 0):.4f}")
            
            # Interpretation
            r2 = eval_results.get('r2_score', 0)
            print(f"\n💡 Interpretation:")
            if r2 > 0.9:
                print(f"   Excellent! Model explains {r2*100:.1f}% of variance")
            elif r2 > 0.7:
                print(f"   Good! Model explains {r2*100:.1f}% of variance")
            elif r2 > 0.5:
                print(f"   Moderate. Model explains {r2*100:.1f}% of variance")
            else:
                print(f"   Poor. Model only explains {r2*100:.1f}% of variance")
        
        elif self.task == 'classification':
            print(f"Accuracy: {eval_results.get('accuracy', 0):.4f}")
            
            if 'classification_report' in eval_results:
                print(f"\nDetailed Classification Report:")
                report = eval_results['classification_report']
                for label, metrics in report.items():
                    if isinstance(metrics, dict) and label not in ['accuracy', 'macro avg', 'weighted avg']:
                        print(f"\nClass {label}:")
                        print(f"  Precision: {metrics.get('precision', 0):.4f}")
                        print(f"  Recall:    {metrics.get('recall', 0):.4f}")
                        print(f"  F1-Score:  {metrics.get('f1-score', 0):.4f}")
            
            # Interpretation
            acc = eval_results.get('accuracy', 0)
            print(f"\n💡 Interpretation:")
            if acc > 0.95:
                print(f"   Excellent! {acc*100:.1f}% accuracy")
            elif acc > 0.85:
                print(f"   Good! {acc*100:.1f}% accuracy")
            elif acc > 0.70:
                print(f"   Moderate. {acc*100:.1f}% accuracy")
            else:
                print(f"   Poor. Only {acc*100:.1f}% accuracy")
        
        elif self.task == 'clustering':
            print(f"Number of Clusters: {eval_results.get('n_clusters', 0)}")
            print(f"Silhouette Score:   {eval_results.get('silhouette_score', 0):.4f}")
            print(f"Inertia:            {eval_results.get('inertia', 0):.4f}")
            
            if 'cluster_sizes' in eval_results:
                print(f"\nCluster Sizes:")
                for cluster, size in eval_results['cluster_sizes'].items():
                    print(f"  Cluster {cluster}: {size} samples")
        
        return eval_results
    
    def save_model_and_results(self, output_dir='trained_models'):
        """
        Save trained model and results
        
        Args:
            output_dir: Directory to save files
        """
        print(f"\n{'='*80}")
        print(f"STEP 5: SAVING MODEL AND RESULTS")
        print(f"{'='*80}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = os.path.join(output_dir, f"{self.model_name}_{timestamp}")
        
        # Save model
        self.model.save_model(model_path)
        print(f"✓ Model saved to: {model_path}")
        
        # Save results
        results_path = f"{model_path}_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=4)
        print(f"✓ Results saved to: {results_path}")
        
        # Save feature importance if available
        if hasattr(self.model, 'get_feature_importance'):
            importance = self.model.get_feature_importance()
            if importance is not None:
                importance_path = f"{model_path}_feature_importance.json"
                with open(importance_path, 'w') as f:
                    json.dump(importance.tolist(), f, indent=4)
                print(f"✓ Feature importance saved to: {importance_path}")
        
        print(f"\n✅ All files saved successfully!")
        
        return model_path
    
    def plot_results(self, save_path=None):
        """
        Create visualization plots
        """
        print(f"\n{'='*80}")
        print(f"STEP 6: GENERATING VISUALIZATIONS")
        print(f"{'='*80}")
        
        try:
            if self.task == 'regression':
                # Actual vs Predicted plot
                y_pred = self.model.predict(self.X_test)
                
                plt.figure(figsize=(10, 6))
                plt.scatter(self.y_test, y_pred, alpha=0.5)
                plt.plot([self.y_test.min(), self.y_test.max()],
                        [self.y_test.min(), self.y_test.max()],
                        'r--', lw=2)
                plt.xlabel('Actual Values')
                plt.ylabel('Predicted Values')
                plt.title(f'{self.model_name} - Actual vs Predicted')
                plt.grid(True)
                
                if save_path:
                    plt.savefig(f"{save_path}_predictions.png")
                    print(f"✓ Prediction plot saved")
                plt.close()
            
            elif self.task == 'classification':
                # Confusion matrix
                from sklearn.metrics import confusion_matrix
                import seaborn as sns
                
                y_pred = self.model.predict(self.X_test)
                cm = confusion_matrix(self.y_test, y_pred)
                
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.title(f'{self.model_name} - Confusion Matrix')
                
                if save_path:
                    plt.savefig(f"{save_path}_confusion_matrix.png")
                    print(f"✓ Confusion matrix saved")
                plt.close()
            
        except Exception as e:
            print(f"⚠️ Could not generate plots: {str(e)}")
    
    def run_complete_pipeline(self, data_path, target_column=None, **kwargs):
        """
        Run the complete pipeline from start to finish
        
        Args:
            data_path: Path to CSV file
            target_column: Name of target column
            **kwargs: Additional parameters
        """
        print(f"\n{'#'*80}")
        print(f"# COMPLETE ML PIPELINE - {self.model_name.upper()}")
        print(f"# Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*80}")
        
        try:
            # Step 1: Load data
            X, y = self.load_data(data_path, target_column)
            
            # Step 2: Preprocess
            self.preprocess_data(X, y, **kwargs.get('preprocess', {}))
            
            # Step 3: Train model
            self.build_and_train_model(**kwargs.get('train', {}))
            
            # Step 4: Evaluate
            self.evaluate_model()
            
            # Step 5: Save
            model_path = self.save_model_and_results(kwargs.get('output_dir', 'trained_models'))
            
            # Step 6: Visualize
            self.plot_results(model_path)
            
            print(f"\n{'#'*80}")
            print(f"# ✅ PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"# Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'#'*80}")
            
            return self.model, self.results
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None


# Command-line interface
def main():
    parser = argparse.ArgumentParser(description='Train and evaluate ML models')
    parser.add_argument('--data', required=True, help='Path to CSV data file')
    parser.add_argument('--target', required=True, help='Name of target column')
    parser.add_argument('--model', default='Linear_Regression',
                       choices=['Linear_Regression', 'Logistic_Regression', 'Decision_Tree',
                               'Random_Forest', 'XGBoost', 'LightGBM', 'SVM',
                               'Neural_Network', 'K-Means', 'PCA'],
                       help='Model to train')
    parser.add_argument('--task', default='regression',
                       choices=['classification', 'regression', 'clustering'],
                       help='Task type')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (default: 0.2)')
    parser.add_argument('--output-dir', default='trained_models',
                       help='Output directory for models')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = MLPipeline(model_name=args.model, task=args.task)
    
    # Run pipeline
    pipeline.run_complete_pipeline(
        data_path=args.data,
        target_column=args.target,
        preprocess={'test_size': args.test_size},
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    # Example usage in code
    print("""
╔══════════════════════════════════════════════════════════════╗
║           END-TO-END ML TRAINING PIPELINE                    ║
╚══════════════════════════════════════════════════════════════╝

OPTION 1: Command Line
----------------------
python train_model.py --data data.csv --target price --model Linear_Regression --task regression

OPTION 2: Python Code
---------------------
from train_model import MLPipeline

pipeline = MLPipeline(model_name='Linear_Regression', task='regression')
model, results = pipeline.run_complete_pipeline(
    data_path='data.csv',
    target_column='price'
)

SUPPORTED MODELS:
-----------------
• Linear_Regression    - Continuous prediction
• Logistic_Regression  - Binary/multiclass classification
• Decision_Tree        - Interpretable classification
• Random_Forest        - Robust ensemble
• XGBoost             - Competition-grade
• LightGBM            - Fast gradient boosting
• SVM                 - High-dimensional data
• Neural_Network      - Deep learning
• K-Means             - Clustering
• PCA                 - Dimensionality reduction
    """)
    
    # If no arguments, show help
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        parser.print_help()
        sys.exit(1)


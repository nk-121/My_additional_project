# Machine Learning Models Collection

Complete collection of ML models with preprocessing and evaluation framework.

## 📁 Files Overview

### 1. **main.py** - Data Preprocessing API
FastAPI-based preprocessing controller for cleaning and preparing data for ML training.

**Features:**
- Handles missing values (median for numeric, mode for categorical)
- Removes duplicates
- Detects and caps outliers (IQR method)
- Encodes categorical variables
- Normalizes numeric features (StandardScaler)
- Provides detailed preprocessing reports

**Endpoints:**
- `POST /analyze` - Analyze CSV without preprocessing
- `POST /preprocess` - Preprocess CSV and return JSON
- `POST /preprocess/download` - Download preprocessed CSV
- `POST /preprocess/report` - Get preprocessing report

**Run:**
```bash
pip install fastapi uvicorn pandas numpy scikit-learn python-multipart
uvicorn main:app --reload
```

---

## 🤖 ML Models (11 Models)

### Classification Models

#### 1. **Decision_Tree.py**
- **Best for:** Interpretable classification, feature importance
- **Pros:** Easy to understand, handles mixed data types
- **Cons:** Prone to overfitting
- **Use case:** When you need to explain decisions

#### 2. **Logistic_Regression.py**
- **Best for:** Binary classification, probability estimates
- **Pros:** Fast, works well with linear relationships
- **Cons:** Assumes linear decision boundary
- **Use case:** Spam detection, disease diagnosis

#### 3. **Random_Forest.py**
- **Best for:** Robust classification, prevents overfitting
- **Pros:** Very accurate, handles missing values
- **Cons:** Slower than single tree
- **Use case:** Credit risk assessment, customer churn

#### 4. **SVM.py** (Support Vector Machine)
- **Best for:** High-dimensional data, clear margins
- **Pros:** Effective in high dimensions
- **Cons:** Slow on large datasets
- **Use case:** Text classification, image recognition

#### 5. **XGBoost.py**
- **Best for:** Kaggle competitions, production systems
- **Pros:** State-of-the-art accuracy, built-in regularization
- **Cons:** Requires tuning
- **Use case:** Winning competitions, high-stakes predictions

#### 6. **LightGBM.py**
- **Best for:** Large datasets, fast training
- **Pros:** Very fast, memory efficient
- **Cons:** Can overfit on small data
- **Use case:** Large-scale production systems

#### 7. **Neural_Network.py**
- **Best for:** Complex patterns, large datasets
- **Pros:** Learns complex relationships
- **Cons:** Needs lots of data, black box
- **Use case:** Image classification, complex patterns

---

### Regression Models

#### 8. **Linear_Regression.py**
- **Best for:** Continuous value prediction
- **Pros:** Fast, interpretable
- **Cons:** Assumes linear relationship
- **Use case:** Price prediction, sales forecasting

---

### Deep Learning Models

#### 9. **LSTM.py** (Long Short-Term Memory)
- **Best for:** Sequential data, time series
- **Pros:** Handles temporal dependencies
- **Cons:** Computationally expensive
- **Use case:** Stock prediction, weather forecasting, text generation

#### 10. **CNN.py** (Convolutional Neural Network)
- **Best for:** Image/pattern recognition
- **Pros:** Automatic feature learning
- **Cons:** Needs large datasets
- **Use case:** Image classification, computer vision

---

### Unsupervised Learning

#### 11. **K-Means.py**
- **Best for:** Customer segmentation, pattern discovery
- **Pros:** Simple, fast
- **Cons:** Need to specify K
- **Use case:** Customer grouping, market segmentation

#### 12. **PCA.py** (Principal Component Analysis)
- **Best for:** Dimensionality reduction, visualization
- **Pros:** Reduces overfitting, speeds up training
- **Cons:** Loses interpretability
- **Use case:** Feature reduction, data visualization

---

## 🎯 model_evaluation.py - Unified Evaluation Framework

Comprehensive evaluation script that:
- Trains multiple models automatically
- Compares performance metrics
- Provides cross-validation scores
- Generates comparison reports
- Saves results to JSON/CSV
- Identifies best model

**Usage Example:**

```python
from model_evaluation import ModelEvaluator
import pandas as pd

# Load preprocessed data
df = pd.read_csv('preprocessed_data.csv')
X = df.drop('target', axis=1)
y = df['target']

# Initialize evaluator
evaluator = ModelEvaluator(task='classification')

# Evaluate all models
evaluator.evaluate_all_models(X, y)

# Print summary with comparison table
evaluator.print_summary()

# Save results
evaluator.save_results('evaluation_results')

# Get best model
best_name, best_model, best_metrics = evaluator.get_best_model()
print(f"Best model: {best_name}")

# Save best model
best_model.save_model(f'models/{best_name}_best')
```

---

## 📊 Complete Workflow Example

### Step 1: Preprocess Data
```python
import requests

# Upload and preprocess CSV
with open('raw_data.csv', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/preprocess/download',
        files={'file': f},
        params={
            'missing_threshold': 0.3,
            'normalize': True,
            'encode_categorical': True,
            'handle_outliers': True
        }
    )

# Save preprocessed data
with open('preprocessed_data.csv', 'wb') as f:
    f.write(response.content)
```

### Step 2: Train Individual Model
```python
from Random_Forest import RandomForestModel
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('preprocessed_data.csv')
X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model
model = RandomForestModel()
model.train(X_train, y_train, tune_hyperparameters=True)

# Evaluate
results = model.evaluate(X_test, y_test)
print(f"Accuracy: {results['accuracy']}")

# Save model
model.save_model('saved_models/random_forest')

# Load model later
loaded_model = RandomForestModel.load_model('saved_models/random_forest')
```

### Step 3: Evaluate All Models
```python
from model_evaluation import ModelEvaluator

# Initialize
evaluator = ModelEvaluator(task='classification')

# Specify which models to test
models_to_test = [
    'Decision_Tree',
    'Random_Forest',
    'XGBoost',
    'LightGBM',
    'Neural_Network'
]

# Evaluate
evaluator.evaluate_all_models(
    X, y,
    models_to_evaluate=models_to_test
)

# Compare
comparison_df = evaluator.compare_models()
print(comparison_df)

# Save results
evaluator.save_results('comparison_results')
```

---

## 📦 Installation

```bash
# Core dependencies
pip install pandas numpy scikit-learn

# Deep learning
pip install tensorflow keras

# Gradient boosting
pip install xgboost lightgbm

# API
pip install fastapi uvicorn python-multipart

# Visualization
pip install matplotlib seaborn
```

---

## 🎨 Model Selection Guide

**Choose based on your needs:**

| Need | Recommended Model |
|------|------------------|
| Best accuracy | XGBoost, LightGBM |
| Fast training | Logistic Regression, Decision Tree |
| Interpretability | Decision Tree, Linear Regression |
| Large datasets | LightGBM, XGBoost |
| Small datasets | Random Forest, SVM |
| Time series | LSTM |
| Image data | CNN |
| Text classification | SVM, LSTM |
| Customer segmentation | K-Means |
| Feature reduction | PCA |

---

## 📈 Performance Metrics

### Classification
- **Accuracy:** Overall correctness
- **Precision:** True positives / Predicted positives
- **Recall:** True positives / Actual positives
- **F1-Score:** Harmonic mean of precision and recall
- **ROC-AUC:** Area under ROC curve

### Regression
- **MSE:** Mean Squared Error
- **RMSE:** Root Mean Squared Error
- **MAE:** Mean Absolute Error
- **R²:** Coefficient of determination

### Clustering
- **Silhouette Score:** Cluster separation quality
- **Davies-Bouldin:** Cluster compactness

---

## 🔧 Hyperparameter Tuning

Most models support automatic hyperparameter tuning:

```python
# Enable tuning
model.train(X_train, y_train, tune_hyperparameters=True)

# Access best parameters
print(model.best_params)
```

---

## 💾 Model Persistence

All models support save/load:

```python
# Save
model.save_model('path/to/model')

# Load
loaded = ModelClass.load_model('path/to/model')
```

---

## ⚠️ Common Issues

1. **Empty DataFrame Error:** All rows were duplicates
   - Check your data for duplicate entries
   - Ensure data has unique samples

2. **Negative Values After Preprocessing:** Normal!
   - StandardScaler creates negative/positive values
   - Represents "below average" vs "above average"
   - This is expected and helps ML models

3. **Out of Memory:** Dataset too large
   - Use LightGBM (memory efficient)
   - Reduce features with PCA
   - Use data sampling

---

## 📞 Support

Each model file contains detailed docstrings explaining:
- When to use the model
- Pros and cons
- Example use cases
- Parameter explanations

Check the `__doc__` string for help:
```python
print(RandomForestModel.__doc__)
```

---

## 🎓 Learning Path

1. Start with **Logistic Regression** or **Decision Tree** (simple)
2. Try **Random Forest** (more robust)
3. Experiment with **XGBoost/LightGBM** (best performance)
4. Use **Neural Networks** for complex patterns
5. Apply **LSTM** for sequential data
6. Use **PCA** to reduce dimensions

---

**Happy Modeling! 🚀**

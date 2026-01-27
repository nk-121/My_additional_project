"""
FastAPI Data Preprocessing Controller for ML Model Training
Handles missing values, outliers, and data anomalies automatically

Installation:
pip install fastapi uvicorn pandas numpy scikit-learn python-multipart

Run:
uvicorn main:app --reload
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import io
import json
import warnings
warnings.filterwarnings('ignore')


app = FastAPI(title="Data Preprocessing API", version="1.0.0")


class PreprocessingReport(BaseModel):
    """Model for preprocessing report"""
    original_shape: tuple
    final_shape: tuple
    columns_dropped: List[Dict[str, str]]
    columns_kept: List[str]
    missing_values_filled: Dict[str, int]
    outliers_detected: Dict[str, int]
    data_types: Dict[str, str]
    preprocessing_steps: List[str]
    recommendations: List[str]


class DataPreprocessor:
    """
    Comprehensive data preprocessing controller
    """
    
    def __init__(self, missing_threshold: float = 0.3):
        """
        Initialize preprocessor
        
        Args:
            missing_threshold: Threshold for dropping columns (default 0.3 = 30%)
        """
        self.missing_threshold = missing_threshold
        self.report = {
            'original_shape': None,
            'final_shape': None,
            'columns_dropped': [],
            'columns_kept': [],
            'missing_values_filled': {},
            'outliers_detected': {},
            'data_types': {},
            'preprocessing_steps': [],
            'recommendations': []
        }
        self.numeric_cols = []
        self.categorical_cols = []
        self.label_encoders = {}
        
    def analyze_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze dataset and provide initial statistics
        
        This function examines the raw data and provides:
        - Total number of rows and columns
        - Missing value counts and percentages for each column
        - Data types of each column
        - Number of duplicate rows
        
        Args:
            df: Input DataFrame to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        analysis = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_summary': {},
            'data_types': {},
            'duplicates': df.duplicated().sum()
        }
        
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100
            analysis['missing_summary'][col] = {
                'count': int(missing_count),
                'percentage': round(missing_pct, 2)
            }
            analysis['data_types'][col] = str(df[col].dtype)
        
        return analysis
    
    def detect_column_types(self, df: pd.DataFrame):
        """
        Detect and classify columns as numeric or categorical
        
        This function automatically determines which columns contain:
        - Numeric data: integers, floats, or values that can be converted to numbers
        - Categorical data: text, categories, or non-numeric values
        
        The classification is important because:
        - Numeric columns: filled with median, normalized, outliers detected
        - Categorical columns: filled with mode, encoded as numbers
        
        Args:
            df: Input DataFrame to analyze
            
        Side Effects:
            Sets self.numeric_cols and self.categorical_cols lists
        """
        self.numeric_cols = []
        self.categorical_cols = []
        
        for col in df.columns:
            # Check if column is already numeric type
            if pd.api.types.is_numeric_dtype(df[col]):
                self.numeric_cols.append(col)
            else:
                # Try to convert to numeric
                try:
                    pd.to_numeric(df[col].dropna(), errors='raise')
                    self.numeric_cols.append(col)
                except (ValueError, TypeError):
                    # It's categorical
                    self.categorical_cols.append(col)
        
        self.report['data_types'] = {
            'numeric': self.numeric_cols,
            'categorical': self.categorical_cols
        }
        self.report['preprocessing_steps'].append(
            f"Detected {len(self.numeric_cols)} numeric and {len(self.categorical_cols)} categorical columns"
        )
    
    def handle_missing_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop columns with missing values above the threshold
        
        If a column has more than 30% (default) missing values, it's dropped because:
        - Too much missing data makes the column unreliable
        - Filling too many values would introduce bias
        - The column likely won't be useful for model training
        
        Example: If a column has 500 missing values out of 1000 rows (50% missing),
        it will be dropped.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with high-missing columns removed
        """
        cols_to_drop = []
        
        for col in df.columns:
            missing_pct = df[col].isnull().sum() / len(df)
            if missing_pct > self.missing_threshold:
                cols_to_drop.append(col)
                self.report['columns_dropped'].append({
                    'column': col,
                    'reason': f'{round(missing_pct * 100, 2)}% missing values (threshold: {self.missing_threshold * 100}%)'
                })
        
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            self.report['preprocessing_steps'].append(
                f"Dropped {len(cols_to_drop)} columns exceeding {self.missing_threshold * 100}% missing threshold"
            )
        
        return df
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows from the dataset
        
        Duplicate rows are exact copies of other rows. They are removed because:
        - They don't add new information
        - They can bias the model by overrepresenting certain patterns
        - They waste computational resources
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with duplicate rows removed
        """
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed = initial_rows - len(df)
        
        if removed > 0:
            self.report['preprocessing_steps'].append(
                f"Removed {removed} duplicate rows ({removed}/{initial_rows} = {round(removed/initial_rows*100, 2)}%)"
            )
            
            # Warning if too many duplicates removed
            if removed == initial_rows:
                self.report['preprocessing_steps'].append(
                    "⚠️ WARNING: ALL rows were duplicates! Dataset is now empty."
                )
            elif removed / initial_rows > 0.5:
                self.report['preprocessing_steps'].append(
                    f"⚠️ WARNING: Over 50% of rows were duplicates. Consider investigating data collection process."
                )
        
        return df
    
    def detect_outliers_iqr(self, series: pd.Series) -> Tuple[List[int], float, float]:
        """
        Detect outliers using the IQR (Interquartile Range) method
        
        How it works:
        1. Calculate Q1 (25th percentile) and Q3 (75th percentile)
        2. Calculate IQR = Q3 - Q1
        3. Lower bound = Q1 - 1.5 × IQR
        4. Upper bound = Q3 + 1.5 × IQR
        5. Values outside these bounds are outliers
        
        Example: If ages are [20, 22, 23, 25, 200]:
        - Q1=22, Q3=25, IQR=3
        - Lower=17.5, Upper=29.5
        - 200 is an outlier (above 29.5)
        
        Args:
            series: Pandas Series of numeric values
            
        Returns:
            Tuple of (outlier_indices, lower_bound, upper_bound)
        """
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_indices = series[(series < lower_bound) | (series > upper_bound)].index.tolist()
        return outlier_indices, lower_bound, upper_bound
    
    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and cap outliers in numeric columns
        
        Instead of removing outliers (which loses data), this function CAPS them:
        - Values below lower bound → set to lower bound
        - Values above upper bound → set to upper bound
        
        Why cap instead of remove?
        - Preserves data points (no information loss)
        - Maintains dataset size
        - Reduces impact of extreme values on model
        
        Example: If salary column has [30k, 35k, 40k, 45k, 1000k]:
        - Bounds might be [20k, 60k]
        - 1000k gets capped to 60k (still high, but not extreme)
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with outliers capped
        """
        for col in self.numeric_cols:
            if col in df.columns:
                try:
                    # Convert to numeric if needed
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Skip if all NaN after conversion
                    if df[col].notna().sum() == 0:
                        continue
                    
                    outlier_indices, lower_bound, upper_bound = self.detect_outliers_iqr(df[col].dropna())
                    
                    if outlier_indices:
                        # Cap outliers instead of removing
                        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                        self.report['outliers_detected'][col] = len(outlier_indices)
                except Exception as e:
                    print(f"Warning: Could not process outliers for column {col}: {str(e)}")
                    continue
        
        if self.report['outliers_detected']:
            total_outliers = sum(self.report['outliers_detected'].values())
            self.report['preprocessing_steps'].append(
                f"Detected and capped {total_outliers} outliers across {len(self.report['outliers_detected'])} columns"
            )
        
        return df
    
    def fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values with appropriate strategies
        
        Different strategies for different data types:
        
        NUMERIC COLUMNS (age, salary, height, etc.):
        - Use MEDIAN (middle value) instead of mean
        - Why median? It's not affected by outliers
        - Example: [10, 20, 30, NULL, 1000] → Fill NULL with 30 (median)
        - If we used mean (212), it would be too high due to 1000
        
        CATEGORICAL COLUMNS (gender, city, status, etc.):
        - Use MODE (most frequent value)
        - Example: ['Male', 'Female', 'Male', NULL, 'Male'] → Fill NULL with 'Male'
        - Makes sense to use the most common category
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values filled
        """
        # Handle numeric columns - use median
        for col in self.numeric_cols:
            if col in df.columns:
                try:
                    # Convert to numeric
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    missing_count = df[col].isnull().sum()
                    if missing_count > 0:
                        median_value = df[col].median()
                        if pd.notna(median_value):
                            df[col].fillna(median_value, inplace=True)
                            self.report['missing_values_filled'][col] = int(missing_count)
                except Exception as e:
                    print(f"Warning: Could not fill missing values for numeric column {col}: {str(e)}")
        
        # Handle categorical columns - use mode
        for col in self.categorical_cols:
            if col in df.columns:
                try:
                    missing_count = df[col].isnull().sum()
                    if missing_count > 0:
                        mode_value = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                        df[col].fillna(mode_value, inplace=True)
                        self.report['missing_values_filled'][col] = int(missing_count)
                except Exception as e:
                    print(f"Warning: Could not fill missing values for categorical column {col}: {str(e)}")
        
        if self.report['missing_values_filled']:
            total_filled = sum(self.report['missing_values_filled'].values())
            self.report['preprocessing_steps'].append(
                f"Filled {total_filled} missing values (median for numeric, mode for categorical)"
            )
        
        return df
    
    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables as numbers
        
        Machine learning models need numbers, not text. This converts:
        - 'Male' → 0, 'Female' → 1
        - 'Red' → 0, 'Blue' → 1, 'Green' → 2
        - 'Low' → 0, 'Medium' → 1, 'High' → 2
        
        Uses Label Encoding:
        - Each unique category gets a unique number
        - Numbers start from 0
        - Consistent across the dataset
        
        Example: Column 'City' with ['Mumbai', 'Delhi', 'Mumbai', 'Bangalore']
        - Mumbai → 0
        - Delhi → 1  
        - Bangalore → 2
        - Result: [0, 1, 0, 2]
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with categorical columns encoded as numbers
        """
        for col in self.categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        if self.categorical_cols:
            self.report['preprocessing_steps'].append(
                f"Encoded {len(self.categorical_cols)} categorical columns using Label Encoding"
            )
        
        return df
    
    def normalize_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize numeric columns using StandardScaler (Z-score normalization)
        
        ⚠️ THIS IS WHY VALUES BECOME NEGATIVE AND POSITIVE! ⚠️
        
        What StandardScaler does:
        1. Calculates mean (average) and standard deviation for each column
        2. Transforms each value: new_value = (old_value - mean) / std_deviation
        
        Result:
        - Mean becomes 0
        - Standard deviation becomes 1
        - Values ABOVE average → POSITIVE numbers
        - Values BELOW average → NEGATIVE numbers
        
        EXAMPLE:
        Original ages: [20, 30, 40, 50, 60]
        - Mean = 40, Std = 14.14
        - Age 20: (20-40)/14.14 = -1.41 (NEGATIVE - below average)
        - Age 30: (30-40)/14.14 = -0.71 (NEGATIVE - below average)
        - Age 40: (40-40)/14.14 =  0.00 (ZERO - exactly average)
        - Age 50: (50-40)/14.14 =  0.71 (POSITIVE - above average)
        - Age 60: (60-40)/14.14 =  1.41 (POSITIVE - above average)
        
        Why normalize?
        - Puts all features on same scale (age in years vs salary in thousands)
        - Prevents features with large values from dominating the model
        - Many ML algorithms work better with normalized data
        - Speeds up training
        
        The negative/positive values are NORMAL and EXPECTED!
        - They represent "how many standard deviations from the mean"
        - Negative = below average
        - Positive = above average
        - The actual prediction will still make sense!
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with normalized numeric columns (values typically between -3 and +3)
        """
        numeric_to_scale = [col for col in self.numeric_cols if col in df.columns]
        
        if numeric_to_scale:
            scaler = StandardScaler()
            df[numeric_to_scale] = scaler.fit_transform(df[numeric_to_scale])
            self.report['preprocessing_steps'].append(
                f"Normalized {len(numeric_to_scale)} numeric columns using StandardScaler"
            )
        
        return df
    
    def generate_recommendations(self, df: pd.DataFrame):
        """
        Generate recommendations based on data analysis
        
        Analyzes the dataset and suggests improvements:
        - Identifies low-variance columns (almost all same value)
        - Detects high-cardinality categorical columns (too many unique values)
        - Checks if dataset is too small
        - Warns about feature-to-sample ratio
        
        These recommendations help improve model performance
        
        Args:
            df: Processed DataFrame
            
        Side Effects:
            Sets self.report['recommendations'] list
        """
        recommendations = []
        
        # Check for low variance columns
        for col in self.numeric_cols:
            if col in df.columns and df[col].std() < 0.01:
                recommendations.append(
                    f"Column '{col}' has very low variance - consider removing"
                )
        
        # Check for high cardinality categorical
        for col in self.categorical_cols:
            if col in df.columns and df[col].nunique() > 50:
                recommendations.append(
                    f"Column '{col}' has high cardinality ({df[col].nunique()} unique values) - consider grouping rare categories"
                )
        
        # Check dataset size
        if len(df) < 100:
            recommendations.append(
                "Dataset is small (<100 rows) - consider gathering more data for better model performance"
            )
        
        # Check feature-to-sample ratio
        if len(df.columns) > len(df) * 0.1:
            recommendations.append(
                f"High feature-to-sample ratio ({len(df.columns)} features, {len(df)} samples) - consider feature selection"
            )
        
        self.report['recommendations'] = recommendations
    
    def preprocess(self, df: pd.DataFrame, 
                   normalize: bool = True,
                   encode_categorical: bool = True,
                   handle_outliers: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """
        Main preprocessing pipeline - processes data step by step
        
        COMPLETE WORKFLOW:
        
        Step 1: Detect column types (numeric vs categorical)
        Step 2: Remove duplicate rows
        Step 3: Drop columns with >30% missing values
        Step 4: Detect and cap outliers (if enabled)
        Step 5: Fill remaining missing values (median/mode)
        Step 6: Encode categorical variables to numbers (if enabled)
        Step 7: Normalize numeric features with StandardScaler (if enabled)
                ⚠️ THIS CREATES NEGATIVE/POSITIVE VALUES ⚠️
        Step 8: Generate recommendations for further improvements
        
        WHY SOME VALUES BECOME NEGATIVE:
        - After normalization (Step 7), values are transformed to show
          how far they are from the average
        - Negative = below average
        - Positive = above average
        - Zero = exactly average
        - This is NORMAL and helps the ML model learn better!
        
        Args:
            df: Input DataFrame (raw data)
            normalize: If True, normalizes numeric features (creates -/+ values)
            encode_categorical: If True, converts text to numbers
            handle_outliers: If True, caps extreme values
        
        Returns:
            Tuple of:
            - processed_df: Clean DataFrame ready for ML training
            - report: Dictionary with detailed preprocessing report
        """
        self.report['original_shape'] = df.shape
        
        # Step 1: Detect column types
        self.detect_column_types(df)
        
        # Step 2: Remove duplicates
        df = self.remove_duplicates(df)
        
        # Step 3: Drop columns with too many missing values
        df = self.handle_missing_columns(df)
        
        # Update column types after dropping
        self.detect_column_types(df)
        
        # Step 4: Handle outliers
        if handle_outliers:
            df = self.handle_outliers(df)
        
        # Check if DataFrame is empty after preprocessing
        if len(df) == 0:
            raise ValueError(
                "No data remains after preprocessing. Possible reasons:\n"
                "1. All rows were duplicates and were removed\n"
                "2. All columns exceeded the missing value threshold\n"
                "3. Dataset was empty to begin with\n"
                f"Original shape: {self.report['original_shape']}\n"
                "Please check your input data."
            )
        
        # Step 5: Fill remaining missing values
        df = self.fill_missing_values(df)
        
        # Step 6: Encode categorical variables
        if encode_categorical and self.categorical_cols:
            df = self.encode_categorical(df)
        
        # Step 7: Normalize numeric features
        if normalize:
            df = self.normalize_numeric(df)
        
        # Step 8: Generate recommendations
        self.generate_recommendations(df)
        
        self.report['final_shape'] = df.shape
        self.report['columns_kept'] = df.columns.tolist()
        
        return df, self.report


# FastAPI Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Data Preprocessing API",
        "version": "1.0.0",
        "endpoints": {
            "/preprocess": "POST - Upload CSV and get preprocessed data",
            "/analyze": "POST - Analyze CSV without preprocessing"
        }
    }


def convert_to_serializable(obj):
    """
    Convert numpy/pandas types to Python native types for JSON serialization
    
    FastAPI cannot directly convert numpy types (like np.int64) to JSON.
    This function recursively converts all numpy types to Python types:
    - np.int64 → int
    - np.float64 → float
    - np.ndarray → list
    - Handles nested dictionaries and lists
    
    Args:
        obj: Object to convert (can be dict, list, numpy type, etc.)
        
    Returns:
        Same object with all numpy types converted to Python types
    """
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return obj


@app.post("/analyze")
async def analyze_csv(file: UploadFile = File(...)):
    """
    Analyze CSV file without preprocessing
    
    Returns initial statistics and data quality report
    """
    try:
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor()
        
        # Analyze data
        analysis = preprocessor.analyze_data(df)
        
        # Convert to serializable format
        analysis = convert_to_serializable(analysis)
        
        return {
            "status": "success",
            "filename": file.filename,
            "analysis": analysis
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error analyzing file: {str(e)}")


@app.post("/preprocess")
async def preprocess_csv(
    file: UploadFile = File(...),
    missing_threshold: float = 0.3,
    normalize: bool = True,
    encode_categorical: bool = True,
    handle_outliers: bool = True,
    return_csv: bool = False
):
    """
    Preprocess CSV file for ML model training
    
    Parameters:
    - file: CSV file to preprocess
    - missing_threshold: Threshold for dropping columns (0.3 = 30% missing)
    - normalize: Whether to normalize numeric features
    - encode_categorical: Whether to encode categorical features
    - handle_outliers: Whether to detect and cap outliers
    - return_csv: If True, returns CSV file; if False, returns JSON (default: False)
    """
    try:
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor(missing_threshold=missing_threshold)
        
        # Preprocess data
        processed_df, report = preprocessor.preprocess(
            df,
            normalize=normalize,
            encode_categorical=encode_categorical,
            handle_outliers=handle_outliers
        )
        
        # Convert report to serializable format
        report = convert_to_serializable(report)
        
        # Return as CSV file or JSON
        if return_csv:
            # Convert to CSV
            output = io.StringIO()
            processed_df.to_csv(output, index=False)
            output.seek(0)
            
            return StreamingResponse(
                io.BytesIO(output.getvalue().encode()),
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename=preprocessed_{file.filename}",
                    "X-Preprocessing-Report": json.dumps(report)
                }
            )
        else:
            # Return as JSON
            return {
                "status": "success",
                "filename": file.filename,
                "report": report,
                "processed_data": processed_df.head(10).to_dict(orient='records'),  # First 10 rows
                "message": "Set return_csv=true to download the full preprocessed CSV"
            }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error preprocessing file: {str(e)}")


@app.post("/preprocess/download")
async def download_preprocessed_csv(
    file: UploadFile = File(...),
    missing_threshold: float = 0.3,
    normalize: bool = True,
    encode_categorical: bool = True,
    handle_outliers: bool = True
):
    """
    Preprocess CSV and download the preprocessed file
    
    This endpoint returns the preprocessed CSV file for download
    """
    try:
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor(missing_threshold=missing_threshold)
        
        # Preprocess data
        processed_df, report = preprocessor.preprocess(
            df,
            normalize=normalize,
            encode_categorical=encode_categorical,
            handle_outliers=handle_outliers
        )
        
        # Convert to CSV
        output = io.StringIO()
        processed_df.to_csv(output, index=False)
        output.seek(0)
        
        # Convert report to serializable format
        report = convert_to_serializable(report)
        
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=preprocessed_{file.filename}",
                "X-Preprocessing-Report": json.dumps(report)
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error preprocessing file: {str(e)}")


@app.post("/preprocess/report")
async def get_preprocessing_report(
    file: UploadFile = File(...),
    missing_threshold: float = 0.3
):
    """
    Get detailed preprocessing report without returning the data
    """
    try:
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor(missing_threshold=missing_threshold)
        
        # Preprocess data
        _, report = preprocessor.preprocess(df)
        
        # Convert to serializable format
        report = convert_to_serializable(report)
        
        return {
            "status": "success",
            "filename": file.filename,
            "report": report
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error generating report: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

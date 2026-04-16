import openml
import pandas as pd
import numpy as np
import os
from filelock import FileLock
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def get_data(problem, index, verbose=True):
    """Downloads and preprocesses a dataset from OpenML suites.

    Args:
        problem (str): Type of machine learning task, either 'classification' 
            or 'regression'.
        index (int): The 1-based index of the task within the OpenML suite.

    Returns:
        tuple: A tuple containing:
            - X_processed (np.ndarray): Processed feature matrix.
            - y_processed (np.ndarray): Processed target vector.
            - dataset_name (str): Name of the OpenML dataset.
            - task_id (int): Unique identifier of the OpenML task.

    Raises:
        ValueError: If `problem` is not 'classification' or 'regression', 
            or if `index` is out of the suite's bounds.
    """
    # Define suite IDs based on the problem type
    if problem == 'classification':
        suit_id = 99
        max_index = 72
    elif problem == 'regression':
        suit_id = 353
        max_index = 35
    else:
        raise ValueError("Problem type must be 'classification' or 'regression'.")
    
    # Validate the task index
    if not (1 <= index <= max_index):
        raise ValueError(f"Index for {problem} must be between 1 and {max_index}.")

    # Prevent race conditions during concurrent downloads using a file lock
    lock_file = os.path.expanduser("~/.openml_cache.lock")

    with FileLock(lock_file):
        try:
            suite = openml.study.get_suite(suite_id=suit_id)
        except Exception as e:
            print("Error connecting to OpenML.")
            raise e
            
        task_ids = list(suite.tasks)
        task_id = task_ids[index - 1]
        if verbose:
            print(f"🔄 Loading dataset {index}/{max_index} (Task ID: {task_id})...")

        # Fetch task and dataset metadata
        task = openml.tasks.get_task(task_id, download_data=True)
        dataset = task.get_dataset()
        dataset_name = dataset.name
        
        # Load raw data as a pandas DataFrame
        X_raw, y_raw, categorical_indicator, _ = dataset.get_data(
            target=dataset.default_target_attribute,
            dataset_format='dataframe'
        )
    
    # --- Feature Preprocessing (X) ---
    # Identify categorical and numerical columns
    cat_cols = [col for col, is_cat in zip(X_raw.columns, categorical_indicator) if is_cat]
    num_cols = [col for col, is_cat in zip(X_raw.columns, categorical_indicator) if not is_cat]
    
    # Transformation pipeline for numerical data
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
    ])
    
    # Transformation pipeline for categorical data
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine transformers into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ],
        verbose_feature_names_out=False
    )
    
    # Fit and transform the feature matrix
    X_processed = preprocessor.fit_transform(X_raw)
    
    # --- Label Preprocessing (y) ---
    if problem == 'classification':
        unique_classes, counts = np.unique(y_raw, return_counts=True)
        
        if len(unique_classes) == 2:
            # Binary Classification: Map minority class to 1 and convert to NumPy array
            minority_class = unique_classes[np.argmin(counts)]
            y_processed = (y_raw == minority_class).values.astype(int)
            
            ratio = counts.min() / counts.sum()
            if verbose:
                print(f"✅ {dataset_name}: Binary. Class '1' assigned to '{minority_class}' (Ratio: {ratio:.2%})")
        else:
            # Multiclass: LabelEncoder returns a NumPy array by default
            le = LabelEncoder()
            y_processed = le.fit_transform(y_raw)
            if verbose:
                print(f"⚠️ {dataset_name}: Multiclass ({len(unique_classes)} classes).")
    else:
        # Regression: Convert pandas Series to NumPy array and handle potential NaNs
        y_processed = y_raw.values.astype(float)
        if np.isnan(y_processed).any():
            y_processed = SimpleImputer(strategy='median').fit_transform(y_processed.reshape(-1, 1)).flatten()
        # si hay varias columnas en y, se asume que es regresión multivariante y se deja como matriz 2D, de lo contrario se convierte a vector 1D
        #evita que pase esto Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
        y_processed = y_processed.astype(float).reshape(-1, 1)
        if verbose:
            print(f"📈 {dataset_name}: Regression.")

    # Ensure X_processed is a NumPy array (ColumnTransformer might return a DataFrame)
    if hasattr(X_processed, 'to_numpy'):
        X_processed = X_processed.to_numpy()

    return X_processed, y_processed, dataset_name, task_id
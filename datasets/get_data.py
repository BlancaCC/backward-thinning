import openml
import pandas as pd
import numpy as np
import os
from filelock import FileLock  # Required: pip install filelock
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def get_data(problem, index):
    """Downloads and preprocesses a dataset from OpenML suites.

    Args:
        problem: String, either 'classification' or 'regression'.
        index: Integer, the 1-based index of the task within the suite.

    Returns:
        X_processed: A NumPy array or DataFrame of processed features.
        y_processed: A NumPy array of processed labels.
        dataset_name: String, the name of the dataset.
        task_id: Integer, the OpenML task ID.

    Raises:
        ValueError: If the problem type is unknown or index is out of bounds.
    """
    if problem == 'classification':
        suit_id = 99
        max_index = 72
    elif problem == 'regression':
        suit_id = 100
        max_index = 30
    else:
        raise ValueError("Problem type must be 'classification' or 'regression'.")
    
    # Validate the task index against the suite size.
    if not (1 <= index <= max_index):
        raise ValueError(f"Index for {problem} must be between 1 and {max_index}.")

    # Use a file lock to prevent race conditions during concurrent downloads
    # in cluster environments.
    lock_file = os.path.expanduser("~/.openml_cache.lock")

    with FileLock(lock_file):
        try:
            suite = openml.study.get_suite(suite_id=suit_id)
        except Exception as e:
            print("Error connecting to OpenML.")
            raise e
            
        task_ids = list(suite.tasks)
        task_id = task_ids[index - 1]
        
        print(f"🔄 Loading dataset {index}/{max_index} (Task ID: {task_id})...")

        # Download the task and associated dataset metadata.
        task = openml.tasks.get_task(task_id, download_data=True)
        dataset = task.get_dataset()
        dataset_name = dataset.name
        
        X_raw, y_raw, categorical_indicator, _ = dataset.get_data(
            target=dataset.default_target_attribute,
            dataset_format='dataframe'
        )
    
    # --- Feature Preprocessing (X) ---
    # Separate columns by type using the OpenML categorical indicator.
    cat_cols = [col for col, is_cat in zip(X_raw.columns, categorical_indicator) if is_cat]
    num_cols = [col for col, is_cat in zip(X_raw.columns, categorical_indicator) if not is_cat]
    
    # Define pipelines for numerical (impute) and categorical (impute + OHE) data.
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
    ])
    
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ],
        verbose_feature_names_out=False
    )
    
    X_processed = preprocessor.fit_transform(X_raw)
    
    # --- Label Preprocessing (y) ---
    unique_classes, counts = np.unique(y_raw, return_counts=True)
    
    if len(unique_classes) == 2:
        # Binary Classification: Explicitly map the minority class to 1.
        # This ensures consistency for metrics like PR-AUC or ROC-AUC.
        minority_class = unique_classes[np.argmin(counts)]
        y_processed = (y_raw == minority_class).astype(int)
        
        ratio = counts.min() / counts.sum()
        print(f"✅ {dataset_name}: Binary. Class '1' assigned to '{minority_class}' "
              f"(Ratio: {ratio:.2%})")
    else:
        # Multiclass: Use standard label encoding.
        le = LabelEncoder()
        y_processed = le.fit_transform(y_raw)
        print(f"⚠️ {dataset_name}: Multiclass ({len(unique_classes)} classes). "
              "Standard encoding.")

    return X_processed, y_processed, dataset_name, task_id

# import openml
# import pandas as pd
# import numpy as np
# import os
# from filelock import FileLock # <--- NECESARIO: pip install filelock
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# def get_data(problem, index):

#     if problem == 'classification':
#         suit_id = 99
#         max_index = 72
        
#     elif problem == 'regression':
#         suit_id = 100
#         max_index = 30
#         if not (1 <= index <= 30):
#             raise ValueError("Index for regression must be between 1 and 30.")
#     else:
#         raise ValueError("Problem type must be 'classification' or 'regression'.")
    
#     # 1. Validate index
#     if not (1 <= index <= max_index):
#         raise ValueError(f"Index for {problem} must be between 1 and {max_index}.")

#     # LOCK PATH: Evita colisiones en cluster
#     lock_file = os.path.expanduser("~/.openml_cache.lock")

#     with FileLock(lock_file):
#         # 2. Get suite (Safe inside lock)
#         try:
#             suite = openml.study.get_suite(suite_id=suit_id)
#         except Exception as e:
#             print("Error connecting to OpenML.")
#             raise e
            
#         task_ids = list(suite.tasks)
#         task_id = task_ids[index - 1]
        
#         print(f"🔄 Loading dataset {index}/{max_index} (Task ID: {task_id})...")

#         # 3. Download Data (CRITICAL SECTION FOR RACE CONDITIONS)
#         task = openml.tasks.get_task(task_id, download_data=True)
#         dataset = task.get_dataset()
#         dataset_name = dataset.name
        
#         X_raw, y_raw, categorical_indicator, _ = dataset.get_data(
#             target=dataset.default_target_attribute,
#             dataset_format='dataframe'
#         )
    
#     # 4. FEATURE PREPROCESSING (X)
#     cat_cols = [col for col, is_cat in zip(X_raw.columns, categorical_indicator) if is_cat]
#     num_cols = [col for col, is_cat in zip(X_raw.columns, categorical_indicator) if not is_cat]
    
#     num_transformer = Pipeline(steps=[
#         ('imputer', SimpleImputer(strategy='median')),
#     ])
    
#     cat_transformer = Pipeline(steps=[
#         ('imputer', SimpleImputer(strategy='most_frequent')),
#         ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
#     ])
    
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', num_transformer, num_cols),
#             ('cat', cat_transformer, cat_cols)
#         ],
#         verbose_feature_names_out=False
#     )
    
#     X_processed = preprocessor.fit_transform(X_raw)
    
#     # 5. LABEL PREPROCESSING (y)
#     unique_classes, counts = np.unique(y_raw, return_counts=True)
    
#     if len(unique_classes) == 2:
#         # --- BINARY CASE: MINORITY CLASS = 1 ---
#         minority_class = unique_classes[np.argmin(counts)]
        
#         # Mapeamos Minoritaria -> 1. 
#         # IMPORTANTE: En el script principal debes leer la probabilidad de la columna 1.
#         y_processed = (y_raw == minority_class).astype(int)
        
#         ratio = counts.min() / counts.sum()
#         print(f"✅ {dataset_name}: Binary. Class '1' assigned to '{minority_class}' (Ratio: {ratio:.2%})")
        
#     else:
#         le = LabelEncoder()
#         y_processed = le.fit_transform(y_raw)
#         print(f"⚠️ {dataset_name}: Multiclass ({len(unique_classes)} classes). Standard encoding.")

#     return X_processed, y_processed, dataset_name, task_id
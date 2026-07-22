# automate_muhammad_aiyub.py
"""
Automated preprocessing script for BPJS Fraud Detection dataset.
This script replicates the preprocessing steps from the experimental notebook.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import warnings
warnings.filterwarnings('ignore')


def run_preprocessing(train_path, val_path, output_dir='namadataset_preprocessing'):
    """
    Load raw CSV files, apply preprocessing pipeline, and save processed data.

    Parameters:
    -----------
    train_path : str
        Path to training CSV file.
    val_path : str
        Path to validation CSV file (may or may not contain the target column).
    output_dir : str, default='namadataset_preprocessing'
        Directory where processed numpy arrays and preprocessor will be saved.

    Returns:
    --------
    None
    """
    # 1. Load data
    print(f"Memuat data dari {train_path} dan {val_path}")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    # 2. Separate target column ('label')
    target_col = 'label'
    if target_col not in df_train.columns:
        raise ValueError(f"Kolom target '{target_col}' tidak ditemukan di data training.")

    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]

    # Handle possible missing values in target (just in case)
    if y_train.isnull().any():
        y_train = y_train.fillna(y_train.mode()[0])

    # For validation set: check if target exists
    if target_col in df_val.columns:
        X_val = df_val.drop(columns=[target_col])
        y_val = df_val[target_col]
        has_y_val = True
    else:
        print(f"Peringatan: Kolom '{target_col}' tidak ditemukan di df_val. Menganggap df_val hanya berisi fitur.")
        X_val = df_val.copy()
        y_val = None
        has_y_val = False

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    if has_y_val:
        print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    else:
        print(f"X_val shape: {X_val.shape} (Tanpa target)")

    # 3. Identify numeric and categorical columns
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    print(f"Jumlah kolom numerik: {len(num_cols)}")
    print(f"Jumlah kolom kategorikal: {len(cat_cols)}")

    # 4. Build preprocessing pipelines
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])

    # 5. Fit and transform
    print("Melakukan preprocessing...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    print(f"Shape data train setelah preprocessing: {X_train_processed.shape}")
    print(f"Shape data val setelah preprocessing:   {X_val_processed.shape}")

    # 6. Save results
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'X_train_processed.npy'), X_train_processed)
    np.save(os.path.join(output_dir, 'X_val_processed.npy'), X_val_processed)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train.values)
    if has_y_val and y_val is not None:
        np.save(os.path.join(output_dir, 'y_val.npy'), y_val.values)
    joblib.dump(preprocessor, os.path.join(output_dir, 'preprocessor.pkl'))

    print(f"✅ Preprocessing selesai. Semua file tersimpan di folder '{output_dir}'.")


if __name__ == "__main__":
    # Determine default paths based on environment
    # If running in Colab, files are often in /content/
    # If running in GitHub Actions, we expect them in namadataset_raw/
    default_train = 'namadataset_raw/fraud_detection_train.csv'
    default_val = 'namadataset_raw/fraud_detection_val.csv'
    
    # Check if files exist in the default location; if not, try /content/
    if not os.path.exists(default_train) and os.path.exists('/content/fraud_detection_train.csv'):
        default_train = '/content/fraud_detection_train.csv'
        default_val = '/content/fraud_detection_val.csv'
    
    # Allow overriding via command-line arguments: python automate.py <train_path> <val_path> [output_dir]
    if len(sys.argv) >= 3:
        train_path = sys.argv[1]
        val_path = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) >= 4 else 'namadataset_preprocessing'
    else:
        train_path = default_train
        val_path = default_val
        output_dir = 'namadataset_preprocessing'

    print(f"Using train_path: {train_path}")
    print(f"Using val_path: {val_path}")
    print(f"Output directory: {output_dir}")

    run_preprocessing(train_path, val_path, output_dir)

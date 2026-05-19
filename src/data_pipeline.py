# File path: GiantVirus_Classification/src/data_pipeline.py

import pandas as pd
from sklearn.model_selection import train_test_split
import os
import re

def clean_taxonomy_string(val):
    """
    Cleans complex taxonomic strings like:
    'NCLDV__Asfuvirales:8(88.89%),ARC__Micrarchaeia:1(11.11%)'
    and extracts the primary label 'NCLDV__Asfuvirales'.
    """
    if pd.isna(val):
        return val
    
    val_str = str(val).strip()
    
    # If it contains commas or colons (like prediction outputs), take the first major component
    if ',' in val_str or ':' in val_str:
        # Split by comma first to get the first candidate
        first_candidate = val_str.split(',')[0]
        # Split by colon to drop the count/percentage part
        clean_label = first_candidate.split(':')[0].strip()
        return clean_label
        
    return val_str

def load_metadata(metadata_path: str) -> pd.DataFrame:
    """
    Load the official label metadata table, dynamically map columns,
    and clean complex mixture labels in taxonomic cells.
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
    
    # Read Excel or CSV based on file extension
    if metadata_path.endswith('.xlsx') or metadata_path.endswith('.xls'):
        df = pd.read_excel(metadata_path, engine='openpyxl')
    else:
        df = pd.read_csv(metadata_path)
    
    # Initialize an empty mapping dictionary
    column_mapping = {}
    
    # Dynamically scan columns for keywords to protect against hidden characters
    for col in df.columns:
        col_str = str(col).strip()
        
        if 'Isolate ID' in col_str or 'Genome_ID' in col_str:
            column_mapping[col] = 'Genome_ID'
        elif 'Class' in col_str:
            column_mapping[col] = 'Class'
        elif 'Order' in col_str:
            column_mapping[col] = 'Order'
        elif 'Family' in col_str:
            column_mapping[col] = 'Family'
            
    print(f"[INFO] Detected and mapped columns: {column_mapping}")
    
    # Rename the columns safely
    df = df.rename(columns=column_mapping)
    
    # Filter out rows missing key taxonomic labels using standardized names
    df = df.dropna(subset=['Class', 'Order', 'Family'])
    
    # --- CRITICAL CLEANING STEP ---
    # Apply cleaning to taxonomic columns to ensure they are clean, solid strings
    print("[INFO] Cleaning messy taxonomy label strings...")
    df['Class'] = df['Class'].apply(clean_taxonomy_string)
    df['Order'] = df['Order'].apply(clean_taxonomy_string)
    df['Family'] = df['Family'].apply(clean_taxonomy_string)
    
    # Convert entire column type explicitly to string to satisfy scikit-learn validation
    df['Class'] = df['Class'].astype(str)
    df['Order'] = df['Order'].astype(str)
    df['Family'] = df['Family'].astype(str)
    
    return df

def stratified_split(df: pd.DataFrame, test_size=0.15, val_size=0.15, random_state=42):
    """
    Perform stratified splitting of the dataset into train, validation, and test sets.
    Filters out rare Family categories that contain fewer than 2 samples to enable stratification.
    """
    # Count the number of samples per Family
    family_counts = df['Family'].value_counts()
    
    # Find families that have too few members (less than 2)
    rare_families = family_counts[family_counts < 2].index.tolist()
    
    if rare_families:
        print(f"[INFO] Filtering out rare families with less than 2 samples: {rare_families}")
        df = df[~df['Family'].isin(rare_families)]
    
    # 1. Split out the independent test set first
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        stratify=df['Family'], 
        random_state=random_state
    )
    
    # 2. Split out the validation set from the remaining training data
    relative_val_size = val_size / (1.0 - test_size)
    
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=relative_val_size, 
        stratify=train_val_df['Family'], 
        random_state=random_state
    )
    
    print("Data Split Complete:")
    print(f"Train: {len(train_df)} samples")
    print(f"Validation: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    pass
# File path: GiantVirus_Classification/src/data_pipeline.py

import pandas as pd
from sklearn.model_selection import train_test_split
import os

def clean_taxonomy_string(val):
    """
    Cleans complex taxonomic strings like:
    'NCLDV__Asfuvirales:8(88.89%),ARC__Micrarchaeia:1(11.11%)'
    and extracts the primary label 'NCLDV__Asfuvirales'.
    """
    if pd.isna(val):
        return "Unknown"
    
    val_str = str(val).strip()
    
    # If it contains commas or colons, take the first major component
    if ',' in val_str or ':' in val_str:
        first_candidate = val_str.split(',')[0]
        clean_label = first_candidate.split(':')[0].strip()
        return clean_label
        
    return val_str

def load_metadata(metadata_path: str) -> pd.DataFrame:
    """
    Load the official label metadata table, dynamically map columns with strict rules
    to avoid multi-column duplicate collision, and clean complex mixture labels.
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
    
    if metadata_path.endswith('.xlsx') or metadata_path.endswith('.xls'):
        df = pd.read_excel(metadata_path, engine='openpyxl')
    else:
        df = pd.read_csv(metadata_path)
    
    column_mapping = {}
    
    # Strict priority scanning to avoid naming collision
    for col in df.columns:
        col_str = str(col).strip()
        
        # 1. Isolate ID has highest priority
        if 'Isolate ID' in col_str or 'Genome_ID' in col_str:
            column_mapping[col] = 'Genome_ID'
        # 2. If it contains 'Family', it must be Family
        elif 'Family' in col_str:
            column_mapping[col] = 'Family'
        # 3. If it contains 'Prediction' and 'Class', it's the true target Class column
        elif 'Class' in col_str and ('Prediction' in col_str or 'Predictio' in col_str):
            column_mapping[col] = 'Class'
        # 4. If it contains 'Order' (even if it has 'Class' in string like 'GVClass Order'), it's Order
        elif 'Order' in col_str:
            column_mapping[col] = 'Order'
        # 5. Fallback for Class if no explicit prediction column was captured yet
        elif 'Class' in col_str and 'Class' not in column_mapping.values():
            column_mapping[col] = 'Class'
            
    print(f"[INFO] Strict Column Mapping Applied: {column_mapping}")
    
    # Rename and keep only the successfully mapped columns to drop duplicates
    df = df.rename(columns=column_mapping)
    
    # Keep only one column if duplicates still exist by chance
    df = df.loc[:, ~df.columns.duplicated()].copy()
    
    # Ensure all required columns exist
    required = ['Genome_ID', 'Class', 'Order', 'Family']
    for req in required:
        if req not in df.columns:
            raise KeyError(f"Missing required column: '{req}'. Current columns: {list(df.columns)}")
            
    # Filter out rows missing key taxonomic labels
    df = df.dropna(subset=['Class', 'Order', 'Family'])
    
    # --- DATA CLEANING STEP ---
    print("[INFO] Cleaning messy taxonomy label strings...")
    df['Class'] = df['Class'].apply(clean_taxonomy_string).astype(str)
    df['Order'] = df['Order'].apply(clean_taxonomy_string).astype(str)
    df['Family'] = df['Family'].apply(clean_taxonomy_string).astype(str)
    
    return df

def stratified_split(df: pd.DataFrame, test_size=0.15, val_size=0.15, random_state=42):
    """
    Perform stratified splitting of the dataset into train, validation, and test sets.
    Filters out rare Family categories that contain fewer than 2 samples to enable stratification.
    """
    family_counts = df['Family'].value_counts()
    rare_families = family_counts[family_counts < 2].index.tolist()
    
    if rare_families:
        print(f"[INFO] Filtering out rare families with less than 2 samples: {rare_families}")
        df = df[~df['Family'].isin(rare_families)]
    
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        stratify=df['Family'], 
        random_state=random_state
    )
    
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
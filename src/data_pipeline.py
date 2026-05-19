# File path: GiantVirus_Classification/src/data_pipeline.py

import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_metadata(metadata_path: str) -> pd.DataFrame:
    """
    Load the official label metadata table and dynamically map columns
    by checking for keywords (Class, Order, Family, Isolate ID) to handle 
    hidden spaces, trailing characters, or unexpected typos in original sheets.
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
        col_str = str(col).strip() # Remove any leading/trailing whitespace
        
        if 'Isolate ID' in col_str or 'Genome_ID' in col_str:
            column_mapping[col] = 'Genome_ID'
        elif 'Class' in col_str:
            column_mapping[col] = 'Class'
        elif 'Order' in col_str:
            column_mapping[col] = 'Order'
        elif 'Family' in col_str:
            column_mapping[col] = 'Family'
            
    print(f"[INFO] Detected and mapped columns: {column_mapping}")
    
    # Verify if all 4 required internal columns were successfully captured
    required = ['Genome_ID', 'Class', 'Order', 'Family']
    found_mapped = list(column_mapping.values())
    missing = [req for req in required if req not in found_mapped]
    
    if missing:
        raise KeyError(
            f"Could not automatically detect columns for: {missing}. "
            f"Actual columns present in the file are: {list(df.columns)}"
        )
    
    # Rename the columns safely to unified internal variable names
    df = df.rename(columns=column_mapping)
    
    # Filter out rows missing key taxonomic labels using standardized names
    df = df.dropna(subset=['Class', 'Order', 'Family'])
    return df

def stratified_split(df: pd.DataFrame, test_size=0.15, val_size=0.15, random_state=42):
    """
    Perform stratified splitting of the dataset into train, validation, and test sets.
    Stratification is strictly based on the finest granularity 'Family' to avoid 
    homologous information leakage across splits.
    """
    # 1. Split out the independent test set first
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        stratify=df['Family'], 
        random_state=random_state
    )
    
    # 2. Split out the validation set from the remaining training data
    # Calculate the relative proportion of val within the train_val subset
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
    # Self-contained placeholder for standalone pipeline testing
    pass
# File path: GiantVirus_Classification/src/data_pipeline.py

import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_metadata(metadata_path: str) -> pd.DataFrame:
    """
    Load the official label metadata table and map custom column names 
    to standard names: Genome_ID, Class, Order, Family.
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
    
    # Read Excel or CSV
    if metadata_path.endswith('.xlsx') or metadata_path.endswith('.xls'):
        df = pd.read_excel(metadata_path, engine='openpyxl')
    else:
        df = pd.read_csv(metadata_path)
    
    # Define the mapping from your Excel columns to standard project columns
    column_mapping = {
        'Isolate ID and Taxonomy': 'Genome_ID',
        'GVClass Taxomomy Predictio': 'Class',
        'Isolate Order Taxonomy': 'Order',
        'Isolate Family Taxonomy': 'Family'
    }
    
    # Rename the columns to prevent KeyError
    df = df.rename(columns=column_mapping)
    
    # Filter out rows missing key taxonomic labels using standard names
    df = df.dropna(subset=['Class', 'Order', 'Family'])
    return df

def stratified_split(df: pd.DataFrame, test_size=0.15, val_size=0.15, random_state=42):
    """
    Perform stratified splitting of the dataset into train, validation, and test sets.
    """
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
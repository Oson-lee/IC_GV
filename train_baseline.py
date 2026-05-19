# File path: GiantVirus_Classification/train_baseline.py

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Import our custom modules
from src.data_pipeline import load_metadata, stratified_split
from src.sequence_processor import process_fasta_file
from src.feature_engineering import KmerFeatureExtractor
from src.model_baseline import get_baseline_model
from src.evaluation import aggregate_slice_probabilities

def prepare_slice_data(df: pd.DataFrame, drep_dir: str, window_size: int = 10000, stride: int = 5000):
    """
    Reads FASTA files based on the dataframe, slices them, and aligns them with labels.
    Returns lists of slice sequences, corresponding labels, and their parent genome IDs.
    """
    all_slices = []
    slice_labels = []
    genome_ids = []
    
    for _, row in df.iterrows():
        genome_id = row['Genome_ID']
        family_label = row['Family']
        
        # Construct file path (assuming files are named like 'Genome_ID.fna')
        fasta_path = os.path.join(drep_dir, f"{genome_id}.fna")
        
        if not os.path.exists(fasta_path):
            print(f"[WARNING] File not found: {fasta_path}. Skipping.")
            continue
            
        # Process and slice the sequence
        slices = process_fasta_file(fasta_path, window_size=window_size, stride=stride)
        
        # Append to our dataset
        all_slices.extend(slices)
        slice_labels.extend([family_label] * len(slices))
        genome_ids.extend([genome_id] * len(slices))
        
    return all_slices, np.array(slice_labels), np.array(genome_ids)

def main():
    print("=== Giant Virus Classification: Baseline Training Pipeline ===")
    
    # --- Configuration ---
    DATA_DIR = "./data"
    METADATA_PATH = os.path.join(DATA_DIR, "metadata.csv")
    DREP_DIR = os.path.join(DATA_DIR, "drep_genomes")
    MODEL_SAVE_DIR = "./saved_models"
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    # Model choice ('rf', 'xgboost', or 'svm')
    MODEL_TYPE = 'rf' 
    KMER_SIZE = 4
    
    # --- Step 1: Load and Split Data ---
    print("\n[Step 1] Loading and splitting metadata...")
    df = load_metadata(METADATA_PATH)
    train_df, val_df, test_df = stratified_split(df)
    
    # Encode string labels (Family names) into integers
    label_encoder = LabelEncoder()
    # Fit only on the entire dataframe to know all possible classes
    label_encoder.fit(df['Family']) 
    
    # --- Step 2: Prepare Slices ---
    print("\n[Step 2] Slicing genomes into manageable windows...")
    print("Processing Training Set...")
    train_seqs, train_labels_str, train_gids = prepare_slice_data(train_df, DREP_DIR)
    print("Processing Test Set...")
    test_seqs, test_labels_str, test_gids = prepare_slice_data(test_df, DREP_DIR)
    
    y_train = label_encoder.transform(train_labels_str)
    y_test = label_encoder.transform(test_labels_str)
    
    # --- Step 3: Feature Engineering (K-mers) ---
    print(f"\n[Step 3] Extracting {KMER_SIZE}-mer TF-IDF features...")
    extractor = KmerFeatureExtractor(k=KMER_SIZE)
    X_train = extractor.fit_transform_tfidf(train_seqs)
    X_test = extractor.transform_tfidf(test_seqs)
    
    print(f"Feature matrix shape: Train {X_train.shape}, Test {X_test.shape}")
    
    # --- Step 4: Model Training ---
    print(f"\n[Step 4] Training {MODEL_TYPE.upper()} baseline model...")
    model = get_baseline_model(MODEL_TYPE, n_estimators=100) # n_estimators is for RF
    model.fit(X_train, y_train)
    print("Training completed.")
    
    # --- Step 5: Evaluation & Aggregation ---
    print("\n[Step 5] Evaluating on Test Set with Multi-slice Voting...")
    # Get probability predictions for all slices
    test_slice_probs = model.predict_proba(X_test)
    
    # Group predictions by original Genome ID
    unique_gids = np.unique(test_gids)
    final_y_true = []
    final_y_pred = []
    
    for gid in unique_gids:
        # Find indices of all slices belonging to this genome
        slice_indices = np.where(test_gids == gid)[0]
        
        # Get the true label for this genome (all slices have the same label)
        true_label = y_test[slice_indices[0]]
        final_y_true.append(true_label)
        
        # Aggregate probabilities using soft-voting (average)
        genome_slice_probs = test_slice_probs[slice_indices]
        agg_probs = aggregate_slice_probabilities(genome_slice_probs, method='average')
        
        # The final prediction is the class with the highest aggregated probability
        pred_label = np.argmax(agg_probs)
        final_y_pred.append(pred_label)
        
    final_y_true = np.array(final_y_true)
    final_y_pred = np.array(final_y_pred)
    
    # Calculate genome-level accuracy
    acc = accuracy_score(final_y_true, final_y_pred)
    print(f"\n>>> Genome-Level Accuracy (Voting): {acc * 100:.2f}% <<<")
    
    print("\nClassification Report (Family Level):")
    # Decode integers back to string names for the report
    target_names = label_encoder.classes_
    print(classification_report(final_y_true, final_y_pred, target_names=target_names, zero_division=0))
    
    # --- Step 6: Save Artifacts ---
    print("\n[Step 6] Saving model and feature extractor...")
    joblib.dump(model, os.path.join(MODEL_SAVE_DIR, f"{MODEL_TYPE}_baseline.pkl"))
    joblib.dump(extractor, os.path.join(MODEL_SAVE_DIR, "kmer_extractor.pkl"))
    joblib.dump(label_encoder, os.path.join(MODEL_SAVE_DIR, "label_encoder.pkl"))
    print("Pipeline finished successfully! Artifacts saved to 'saved_models/'.")

if __name__ == "__main__":
    main()
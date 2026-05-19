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
    Reads FASTA files based on the dataframe, handles loose filename matching,
    slices them, and aligns them with labels.
    """
    all_slices = []
    slice_labels = []
    genome_ids = []
    
    # Cache all available files in the directory to speed up scanning
    available_files = os.listdir(drep_dir)
    print(f"[INFO] Found {len(available_files)} total files in drep_genomes directory.")
    
    for _, row in df.iterrows():
        genome_id = str(row['Genome_ID']).strip()
        family_label = row['Family']
        
        # --- Robust Filename Matching Logic ---
        target_file = None
        
        # 1. Try exact match first
        possible_exact = f"{genome_id}.fna"
        if possible_exact in available_files:
            target_file = possible_exact
        else:
            # 2. Split by space to get the true short ID prefix (e.g., "GCA_000906035-1_ViralProj195482")
            short_id = genome_id.split()[0]
            
            # Check standard extensions
            for ext in ['.fna', '.fasta', '.fa']:
                possible_match = f"{short_id}{ext}"
                if possible_match in available_files:
                    target_file = possible_match
                    break
            
            # 3. Ultimate fallback: if still not found, scan files for containment
            if target_file is None:
                for f in available_files:
                    if f.endswith(('.fna', '.fasta', '.fa')):
                        f_base = os.path.splitext(f)[0]
                        if short_id in f_base or f_base in short_id:
                            target_file = f
                            break
                        
        if target_file is None:
            print(f"[WARNING] File not found for ID: {genome_id}. Skipping.")
            continue
            
        fasta_path = os.path.join(drep_dir, target_file)
        
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
    METADATA_PATH = os.path.join(DATA_DIR, "metadata.xlsx") # Make sure this is correct
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
    label_encoder.fit(df['Family']) 
    
    # --- Step 2: Prepare Slices ---
    print("\n[Step 2] Slicing genomes into manageable windows...")
    print("Processing Training Set...")
    train_seqs, train_labels_str, train_gids = prepare_slice_data(train_df, DREP_DIR)
    print(f"-> Extracted {len(train_seqs)} training slices.")
    
    print("Processing Test Set...")
    test_seqs, test_labels_str, test_gids = prepare_slice_data(test_df, DREP_DIR)
    print(f"-> Extracted {len(test_seqs)} test slices.")
    
    if len(train_seqs) == 0 or len(test_seqs) == 0:
        print("\n[ERROR] No sequences found! Please check your file paths and names.")
        sys.exit(1)
        
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
    model = get_baseline_model(MODEL_TYPE, n_estimators=100)
    model.fit(X_train, y_train)
    print("Training completed.")
    
    # --- Step 5: Evaluation & Aggregation ---
    print("\n[Step 5] Evaluating on Test Set with Multi-slice Voting...")
    test_slice_probs = model.predict_proba(X_test)
    
    unique_gids = np.unique(test_gids)
    final_y_true = []
    final_y_pred = []
    
    for gid in unique_gids:
        slice_indices = np.where(test_gids == gid)[0]
        
        true_label = y_test[slice_indices[0]]
        final_y_true.append(true_label)
        
        genome_slice_probs = test_slice_probs[slice_indices]
        agg_probs = aggregate_slice_probabilities(genome_slice_probs, method='average')
        
        pred_label = np.argmax(agg_probs)
        final_y_pred.append(pred_label)
        
    final_y_true = np.array(final_y_true)
    final_y_pred = np.array(final_y_pred)
    
    acc = accuracy_score(final_y_true, final_y_pred)
    print(f"\n>>> Genome-Level Accuracy (Voting): {acc * 100:.2f}% <<<")
    
    print("\nClassification Report (Family Level):")
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
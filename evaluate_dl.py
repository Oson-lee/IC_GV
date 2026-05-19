# File path: GiantVirus_Classification/evaluate_dl.py

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Import custom modules
from src.data_pipeline import load_metadata, stratified_split
from src.sequence_processor import process_fasta_file
from src.model_deep_learning import GiantVirus1DCNN, GiantVirusTransformer, GiantVirusHyperbolicNet

# --- Step 1: Replicate Dataset Structure ---
class GiantVirusEvalDataset(Dataset):
    """
    Dataset optimized for evaluation, returning only sequences.
    We handle labels separately for genome-level grouping.
    """
    def __init__(self, sequences, max_len=10000):
        self.sequences = sequences
        self.max_len = max_len
        self.vocab = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx][:self.max_len]
        token_ids = [self.vocab.get(base, 4) for base in seq]
        if len(token_ids) < self.max_len:
            token_ids += [4] * (self.max_len - len(token_ids))
        return torch.tensor(token_ids, dtype=torch.long)

# --- Step 2: Extraction with Genome ID Tracking ---
def prepare_eval_data(df, drep_dir, window_size=10000, stride=5000):
    """
    Extracts sequences and strictly tracks their parent Genome ID for voting.
    """
    available_files = os.listdir(drep_dir)
    all_seqs, class_list, order_list, family_list, genome_ids = [], [], [], [], []
    
    for _, row in df.iterrows():
        genome_id = str(row['Genome_ID']).strip()
        short_id = genome_id.split()[0]
        
        target_file = None
        for ext in ['.fna', '.fasta', '.fa']:
            if f"{short_id}{ext}" in available_files:
                target_file = f"{short_id}{ext}"
                break
                
        if target_file is None:
            continue
            
        fasta_path = os.path.join(drep_dir, target_file)
        slices = process_fasta_file(fasta_path, window_size=window_size, stride=stride)
        
        all_seqs.extend(slices)
        class_list.extend([row['Class']] * len(slices))
        order_list.extend([row['Order']] * len(slices))
        family_list.extend([row['Family']] * len(slices))
        # Crucial addition for evaluation: track the parent ID!
        genome_ids.extend([genome_id] * len(slices))
        
    return all_seqs, np.array(class_list), np.array(order_list), np.array(family_list), np.array(genome_ids)

# --- Step 3: Main Evaluation Pipeline ---
def main():
    print("=== Giant Virus Classification: DL Multi-Task Evaluation ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Evaluating on device: {device}")
    
    DATA_DIR = "./data"
    METADATA_PATH = os.path.join(DATA_DIR, "metadata.xlsx")
    DREP_DIR = os.path.join(DATA_DIR, "drep_genomes")
    MODEL_SAVE_DIR = "./saved_models"
    
    # Toggle this to match what you trained
    MODEL_CHOICE = 'hyperbolic' 
    MODEL_PATH = os.path.join(MODEL_SAVE_DIR, f"giant_virus_{MODEL_CHOICE}.pth")
    
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Trained model weights not found at {MODEL_PATH}")
        sys.exit(1)
        
    # 1. Re-initialize Encoders and Splitting
    df = load_metadata(METADATA_PATH)
    _, _, test_df = stratified_split(df)
    
    le_class = LabelEncoder().fit(df['Class'])
    le_order = LabelEncoder().fit(df['Order'])
    le_family = LabelEncoder().fit(df['Family'])
    
    # 2. Extract Test Slices
    print("\n[Processing] Extracting genomic windows from Test Set...")
    test_seqs, test_c_str, test_o_str, test_f_str, test_gids = prepare_eval_data(test_df, DREP_DIR)
    print(f"-> Extracted {len(test_seqs)} test slices for voting.")
    
    if len(test_seqs) == 0:
        print("[ERROR] No test sequences extracted. Check file paths.")
        sys.exit(1)
        
    # Transform true string labels to integers
    y_test_c = le_class.transform(test_c_str)
    y_test_o = le_order.transform(test_o_str)
    y_test_f = le_family.transform(test_f_str)
    
    # 3. Initialize Dataloader & Model
    test_dataset = GiantVirusEvalDataset(test_seqs)
    # shuffle=False is strictly required to keep alignment with test_gids!
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"\n[Loading] Initializing {MODEL_CHOICE.upper()} architecture...")
    if MODEL_CHOICE == 'cnn':
        model = GiantVirus1DCNN(vocab_size=5, num_classes=len(le_class.classes_), num_orders=len(le_order.classes_), num_families=len(le_family.classes_))
    elif MODEL_CHOICE == 'transformer':
        model = GiantVirusTransformer(vocab_size=5, num_classes=len(le_class.classes_), num_orders=len(le_order.classes_), num_families=len(le_family.classes_))
    elif MODEL_CHOICE == 'hyperbolic':
        model = GiantVirusHyperbolicNet(vocab_size=5, num_classes=len(le_class.classes_), num_orders=len(le_order.classes_), num_families=len(le_family.classes_))
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    
    # 4. Run Inference
    print("[Inference] Running forward pass on all test slices...")
    all_probs_c, all_probs_o, all_probs_f = [], [], []
    
    with torch.no_grad():
        for bx in test_loader:
            bx = bx.to(device)
            out_c, out_o, out_f = model(bx)
            
            # Apply softmax to convert raw logits to probabilities
            probs_c = F.softmax(out_c, dim=-1).cpu().numpy()
            probs_o = F.softmax(out_o, dim=-1).cpu().numpy()
            probs_f = F.softmax(out_f, dim=-1).cpu().numpy()
            
            all_probs_c.append(probs_c)
            all_probs_o.append(probs_o)
            all_probs_f.append(probs_f)
            
    all_probs_c = np.vstack(all_probs_c)
    all_probs_o = np.vstack(all_probs_o)
    all_probs_f = np.vstack(all_probs_f)
    
    # 5. Soft-Voting Aggregation per Genome
    print("\n[Evaluation] Aggregating slice probabilities (Soft-Voting)...")
    unique_gids = np.unique(test_gids)
    
    final_true_c, final_true_o, final_true_f = [], [], []
    final_pred_c, final_pred_o, final_pred_f = [], [], []
    
    for gid in unique_gids:
        # Find indices of all slices belonging to this specific genome
        slice_indices = np.where(test_gids == gid)[0]
        
        # Get true labels for this genome
        final_true_c.append(y_test_c[slice_indices[0]])
        final_true_o.append(y_test_o[slice_indices[0]])
        final_true_f.append(y_test_f[slice_indices[0]])
        
        # Average the probabilities across all slices
        avg_prob_c = np.mean(all_probs_c[slice_indices], axis=0)
        avg_prob_o = np.mean(all_probs_o[slice_indices], axis=0)
        avg_prob_f = np.mean(all_probs_f[slice_indices], axis=0)
        
        # Final prediction is the argmax of the averaged probability
        final_pred_c.append(np.argmax(avg_prob_c))
        final_pred_o.append(np.argmax(avg_prob_o))
        final_pred_f.append(np.argmax(avg_prob_f))
        
    # 6. Generate Reports
    def print_report(y_true, y_pred, label_encoder, level_name):
        acc = accuracy_score(y_true, y_pred)
        print(f"\n{'='*50}")
        print(f">>> {level_name.upper()} LEVEL ACCURACY: {acc * 100:.2f}% <<<")
        print(f"{'='*50}")
        
        present_classes = np.unique(np.concatenate([y_true, y_pred]))
        target_names = [label_encoder.classes_[i] for i in present_classes]
        
        print(classification_report(
            y_true, 
            y_pred, 
            labels=present_classes, 
            target_names=target_names, 
            zero_division=0
        ))

    print_report(final_true_c, final_pred_c, le_class, "Class")
    print_report(final_true_o, final_pred_o, le_order, "Order")
    print_report(final_true_f, final_pred_f, le_family, "Family")

if __name__ == "__main__":
    main()
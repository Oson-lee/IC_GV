# File path: evaluate_dl.py

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Import custom modules from the src directory
from src.data_pipeline import load_metadata, stratified_split
from src.sequence_processor import process_fasta_file
from src.model_deep_learning import GiantVirus1DCNN, GiantVirusTransformer, GiantVirusHyperbolicNet

def clean_state_dict_keys(state_dict):
    """
    Removes the '_orig_mod.' prefix from state dict keys caused by torch.compile 
    to guarantee flawless loading back into standard model architectures.
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('_orig_mod.', '')
        new_state_dict[new_key] = value
    return new_state_dict

def evaluate_genome_level_soft_voting(model, model_path, test_df, drep_dir, le_c, le_o, le_f, device):
    """
    Executes the multi-slice soft-voting inference mechanism on complete, full-length genomes.
    Aggregates slice-level probability distributions via a global mean before final argmax selection.
    """
    # Load and patch weight tensors safely
    raw_weights = torch.load(model_path, map_location=device)
    cleaned_weights = clean_state_dict_keys(raw_weights)
    model.load_state_dict(cleaned_weights)
    model.to(device)
    model.eval()
    
    available_files = os.listdir(drep_dir)
    vocab = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    max_len = 10000
    
    true_classes, pred_classes = [], []
    true_orders, pred_orders = [], []
    true_families, pred_families = [], []
    
    print(f"[Inference] Running soft-voting pipeline for model checkpoint: {os.path.basename(model_path)}")
    
    with torch.no_grad():
        for _, row in test_df.iterrows():
            genome_id = str(row['Genome_ID']).strip()
            short_id = genome_id.split()[0]
            
            # Match genomic FASTA files under raw dataset naming anomalies
            target_file = None
            for ext in ['.fna', '.fasta', '.fa']:
                if f"{short_id}{ext}" in available_files:
                    target_file = f"{short_id}{ext}"
                    break
            if target_file is None:
                for f in available_files:
                    if f.endswith(('.fna', '.fasta', '.fa')):
                        f_base = os.path.splitext(f)[0]
                        if short_id in f_base or f_base in short_id:
                            target_file = f
                            break
            if target_file is None:
                continue
                
            fasta_path = os.path.join(drep_dir, target_file)
            
            # Execute slicing (Standard test evaluation window and stride settings)
            slices = process_fasta_file(fasta_path, window_size=max_len, stride=5000)
            if len(slices) == 0:
                continue
                
            # Pack all biological slices from the single genome into a unified batch tensor
            tensor_list = []
            for chunk in slices:
                token_ids = [vocab.get(base, 4) for base in chunk[:max_len]]
                if len(token_ids) < max_len:
                    token_ids += [4] * (max_len - len(token_ids))
                tensor_list.append(token_ids)
                
            genome_batch = torch.tensor(tensor_list, dtype=torch.long).to(device)
            
            # Forward prediction with automatic mixed precision compatibility
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16):
                out_c, out_o, out_f = model(genome_batch)
                
                # Apply Softmax to translate logits into standard probability vectors
                prob_c = torch.softmax(out_c, dim=-1).mean(dim=0)
                prob_o = torch.softmax(out_o, dim=-1).mean(dim=0)
                prob_f = torch.softmax(out_f, dim=-1).mean(dim=0)
            
            # Pull predicted indices back to host CPU
            pred_c_idx = torch.argmax(prob_c).item()
            pred_o_idx = torch.argmax(prob_o).item()
            pred_f_idx = torch.argmax(prob_f).item()
            
            # Map structural taxonomic labels back to string categories
            true_classes.append(le_class.transform([row['Class']])[0])
            true_orders.append(le_order.transform([row['Order']])[0])
            true_families.append(le_family.transform([row['Family']])[0])
            
            pred_classes.append(pred_c_idx)
            pred_orders.append(pred_o_idx)
            pred_families.append(pred_f_idx)
            
    # Calculate global structural genome-level accuracy scores
    acc_c = accuracy_score(true_classes, pred_classes) * 100
    acc_o = accuracy_score(true_orders, pred_orders) * 100
    acc_f = accuracy_score(true_families, pred_families) * 100
    
    return acc_c, acc_o, acc_f

if __name__ == "__main__":
    print("=== Giant Virus Classification: Unified Genome-Level Evaluation ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    DATA_DIR = "./data"
    METADATA_PATH = os.path.join(DATA_DIR, "metadata.xlsx")
    DREP_DIR = os.path.join(DATA_DIR, "drep_genomes")
    MODEL_SAVE_DIR = "./saved_models"
    
    df = load_metadata(METADATA_PATH)
    _, _, test_df = stratified_split(df)
    
    le_class = LabelEncoder().fit(df['Class'])
    le_order = LabelEncoder().fit(df['Order'])
    le_family = LabelEncoder().fit(df['Family'])
    
    architectures = ['1d_cnn', 'transformer', 'hyperbolic']
    ablation_records = []
    
    for m_type in architectures:
        model_path = os.path.join(MODEL_SAVE_DIR, f"giant_virus_{m_type}.pth")
        if not os.path.exists(model_path):
            print(f"[WARNING] Model checkpoint missing for {m_type.upper()}. Skipping.")
            continue
            
        # Dynamically spawn architecture structures
        if m_type == '1d_cnn':
            model = GiantVirus1DCNN(vocab_size=5, num_classes=len(le_class.classes_), num_orders=len(le_order.classes_), num_families=len(le_family.classes_))
        elif m_type == 'transformer':
            model = GiantVirusTransformer(vocab_size=5, num_classes=len(le_class.classes_), num_orders=len(le_order.classes_), num_families=len(le_family.classes_))
        elif m_type == 'hyperbolic':
            model = GiantVirusHyperbolicNet(vocab_size=5, num_classes=len(le_class.classes_), num_orders=len(le_order.classes_), num_families=len(le_family.classes_))
            
        acc_class, acc_order, acc_family = evaluate_genome_level_soft_voting(
            model, model_path, test_df, DREP_DIR, le_class, le_order, le_family, device
        )
        
        ablation_records.append({
            "Architecture Backbone": m_type.upper(),
            "Class Accuracy": f"{acc_class:.2f}%",
            "Order Accuracy": f"{acc_order:.2f}%",
            "Family Accuracy": f"{acc_family:.2f}%"
        })
        
    # Build and print the comprehensive final ablation comparison matrix
    summary_df = pd.DataFrame(ablation_records)
    print("\n" + "="*20 + " FINAL ABLATION STUDY RESULTS (GENOME-LEVEL) " + "="*20)
    print(summary_df.to_markdown(index=False))
    print("="*65)
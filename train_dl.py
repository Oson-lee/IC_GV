# File path: train_dl.py

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

# Import custom modules from the src directory
from src.data_pipeline import load_metadata, stratified_split
from src.sequence_processor import process_fasta_file
from src.model_deep_learning import GiantVirus1DCNN, GiantVirusTransformer, GiantVirusHyperbolicNet

# --- Step 1: PyTorch Dataset Definition ---
class GiantVirusDataset(Dataset):
    """
    Converts raw nucleotide character strings into numeric sequence tokens
    and bundles hierarchical labels (Class, Order, Family) into a unified tensor batch.
    """
    def __init__(self, sequences, classes, orders, families, max_len=10000):
        self.sequences = sequences
        self.classes = torch.tensor(classes, dtype=torch.long)
        self.orders = torch.tensor(orders, dtype=torch.long)
        self.families = torch.tensor(families, dtype=torch.long)
        self.max_len = max_len
        
        # Standard DNA nucleotide token mapping (Vocabulary Size = 5)
        self.vocab = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx][:self.max_len]
        # Map tokens, default to 'N' (token 4) if any ambiguous or unexpected IUPAC characters appear
        token_ids = [self.vocab.get(base, 4) for base in seq]
        
        # Explicit padding with token 4 ('N') to preserve fixed-size structural dimensions
        if len(token_ids) < self.max_len:
            token_ids += [4] * (self.max_len - len(token_ids))
            
        return torch.tensor(token_ids, dtype=torch.long), self.classes[idx], self.orders[idx], self.families[idx]


# --- Step 2: Advanced Sequence Slicing Alignment with Data Augmentation ---
def prepare_dl_data(df, drep_dir, window_size=10000, stride=5000):
    """
    Scans the data repository, aligns short prefixes and full genome IDs,
    slices sequences via sliding windows, and replicates parent/child labels.
    Using a low stride (e.g., 1000) acts as massive data augmentation for training.
    """
    available_files = os.listdir(drep_dir)
    all_seqs = []
    class_list = []
    order_list = []
    family_list = []
    
    for _, row in df.iterrows():
        genome_id = str(row['Genome_ID']).strip()
        short_id = genome_id.split()[0]
        
        target_file = None
        # Phase 1: Direct structural matching
        for ext in ['.fna', '.fasta', '.fa']:
            if f"{short_id}{ext}" in available_files:
                target_file = f"{short_id}{ext}"
                break
                
        # Phase 2: Ultimate fallback substring scan
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
        # Slicing execution
        slices = process_fasta_file(fasta_path, window_size=window_size, stride=stride)
        
        all_seqs.extend(slices)
        class_list.extend([row['Class']] * len(slices))
        order_list.extend([row['Order']] * len(slices))
        family_list.extend([row['Family']] * len(slices))
        
    return all_seqs, class_list, order_list, family_list


# --- Step 3: Multi-Backbone Ablation Training Loop ---
def main():
    print("=== Giant Virus Classification: Deep Learning Multi-Task Pipeline ===")
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using execution device: {device}")
    
    DATA_DIR = "./data"
    METADATA_PATH = os.path.join(DATA_DIR, "metadata.xlsx")
    DREP_DIR = os.path.join(DATA_DIR, "drep_genomes")
    MODEL_SAVE_DIR = "./saved_models"
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    # 1. Load and Partition Metadata
    df = load_metadata(METADATA_PATH)
    train_df, val_df, test_df = stratified_split(df)
    
    # Unify and fit categorical Label Encoders across all valid ranks
    le_class = LabelEncoder().fit(df['Class'])
    le_order = LabelEncoder().fit(df['Order'])
    le_family = LabelEncoder().fit(df['Family'])
    
    # 2. Extract and Align Windows with Targeted Strides (Data Augmentation Strategy)
    print("\n[Processing] Extracting training windows (Aggressive 1,000 bp Stride for Augmentation)...")
    train_seqs, train_c_str, train_o_str, train_f_str = prepare_dl_data(train_df, DREP_DIR, window_size=10000, stride=1000)
    
    print("[Processing] Extracting validation windows (Clean 5,000 bp Stride for Pure Validation)...")
    val_seqs, val_c_str, val_o_str, val_f_str = prepare_dl_data(val_df, DREP_DIR, window_size=10000, stride=5000)
    
    # Map raw strings into transformed integer classes
    train_c = le_class.transform(train_c_str)
    train_o = le_order.transform(train_o_str)
    train_f = le_family.transform(train_f_str)
    
    val_c = le_class.transform(val_c_str)
    val_o = le_order.transform(val_o_str)
    val_f = le_family.transform(val_f_str)
    
    print(f"[INFO] Augmentation Result -> Train slices: {len(train_seqs)}, Val slices: {len(val_seqs)}")
    
    # Instantiate clean PyTorch DataLoaders
    train_dataset = GiantVirusDataset(train_seqs, train_c, train_o, train_f, max_len=10000)
    val_dataset = GiantVirusDataset(val_seqs, val_c, val_o, val_f, max_len=10000)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, drop_last=False)
    
    # Compute inverse class frequencies to mitigate severe long-tailed distribution biases
    def calculate_weights(labels, num_classes):
        classes_present = np.unique(labels)
        weights = compute_class_weight(class_weight='balanced', classes=classes_present, y=labels)
        full_weights = np.ones(num_classes, dtype=np.float32)
        for idx, cls_id in enumerate(classes_present):
            full_weights[cls_id] = weights[idx]
        return torch.tensor(full_weights, dtype=torch.float).to(device)

    class_w = calculate_weights(train_c, len(le_class.classes_))
    order_w = calculate_weights(train_o, len(le_order.classes_))
    family_w = calculate_weights(train_f, len(le_family.classes_))
    
    criterion_class = nn.CrossEntropyLoss(weight=class_w)
    criterion_order = nn.CrossEntropyLoss(weight=order_w)
    criterion_family = nn.CrossEntropyLoss(weight=family_w)
    
    # 3. Automated Ablation Control Stream Loop
    model_architectures = ['1d_cnn', 'transformer', 'hyperbolic']
    num_epochs = 10
    
    for model_type in model_architectures:
        print(f"\n" + "="*20 + f" Starting Training Backbone: {model_type.upper()} " + "="*20)
        
        # Dynamic object instantiation matching the core research parameters
        if model_type == '1d_cnn':
            model = GiantVirus1DCNN(
                vocab_size=5, 
                num_classes=len(le_class.classes_), 
                num_orders=len(le_order.classes_), 
                num_families=len(le_family.classes_)
            )
        elif model_type == 'transformer':
            model = GiantVirusTransformer(
                vocab_size=5, 
                num_classes=len(le_class.classes_), 
                num_orders=len(le_order.classes_), 
                num_families=len(le_family.classes_)
            )
        elif model_type == 'hyperbolic':
            model = GiantVirusHyperbolicNet(
                vocab_size=5, 
                num_classes=len(le_class.classes_), 
                num_orders=len(le_order.classes_), 
                num_families=len(le_family.classes_)
            )
            
        model = model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct_f = 0
            total_slices = 0
            
            for batch_tokens, batch_c, batch_o, batch_f in train_loader:
                batch_tokens = batch_tokens.to(device)
                batch_c = batch_c.to(device)
                batch_o = batch_o.to(device)
                batch_f = batch_f.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass across shared backbone heads
                out_c, out_o, out_f = model(batch_tokens)
                
                # Multi-task loss summation forcing structural hierarchical consistency
                loss_c = criterion_class(out_c, batch_c)
                loss_o = criterion_order(out_o, batch_o)
                loss_f = criterion_family(out_f, batch_f)
                total_loss = loss_c + loss_o + loss_f
                
                total_loss.backward()
                optimizer.step()
                
                running_loss += total_loss.item() * batch_tokens.size(0)
                preds_f = torch.argmax(out_f, dim=-1)
                correct_f += (preds_f == batch_f).sum().item()
                total_slices += batch_tokens.size(0)
                
            epoch_loss = running_loss / total_slices
            epoch_acc_f = (correct_f / total_slices) * 100
            
            # Validation Step
            model.eval()
            val_loss = 0.0
            val_correct_c = 0
            val_correct_o = 0
            val_correct_f = 0
            val_total = 0
            
            with torch.no_grad():
                for v_tokens, v_c, v_o, v_f in val_loader:
                    v_tokens = v_tokens.to(device)
                    v_c = v_c.to(device)
                    v_o = v_o.to(device)
                    v_f = v_f.to(device)
                    
                    vo_c, vo_o, vo_f = model(v_tokens)
                    
                    vl_c = criterion_class(vo_c, v_c)
                    vl_o = criterion_order(vo_o, v_o)
                    vl_f = criterion_family(vo_f, v_f)
                    
                    val_loss += (vl_c + vl_o + vl_f).item() * v_tokens.size(0)
                    
                    val_correct_c += (torch.argmax(vo_c, dim=-1) == v_c).sum().item()
                    val_correct_o += (torch.argmax(vo_o, dim=-1) == v_o).sum().item()
                    val_correct_f += (torch.argmax(vo_f, dim=-1) == v_f).sum().item()
                    val_total += v_tokens.size(0)
                    
            epoch_val_loss = val_loss / val_total
            val_acc_c = (val_correct_c / val_total) * 100
            val_acc_o = (val_correct_o / val_total) * 100
            val_acc_f = (val_correct_f / val_total) * 100
            
            print(f"Epoch [{epoch+1}/{num_epochs}] -> Train Loss: {epoch_loss:.4f} | Train Family Acc: {epoch_acc_f:.2f}% || Val Loss: {epoch_val_loss:.4f} | Val Family Acc: {val_acc_f:.2f}%")
            
            # Save the highest-performing checkpoint model states
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                checkpoint_path = os.path.join(MODEL_SAVE_DIR, f"giant_virus_{model_type}.pth")
                torch.save(model.state_dict(), checkpoint_path)
                
        print(f"[SUCCESS] Finished training {model_type.upper()}. Optimal weights saved to {checkpoint_path}")

if __name__ == "__main__":
    main()
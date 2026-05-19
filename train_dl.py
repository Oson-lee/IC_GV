# File path: GiantVirus_Classification/train_dl.py

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

# Import custom modules
from src.data_pipeline import load_metadata, stratified_split
from src.sequence_processor import process_fasta_file
from src.model_deep_learning import GiantVirus1DCNN, GiantVirusTransformer, GiantVirusHyperbolicNet

# --- Step 1: PyTorch Dataset Definition ---
class GiantVirusDataset(Dataset):
    """
    Converts nucleotide characters directly into sequence index tokens
    and bundles hierarchical labels (Class, Order, Family) together.
    """
    def __init__(self, sequences, classes, orders, families, max_len=10000):
        self.sequences = sequences
        self.classes = torch.tensor(classes, dtype=torch.long)
        self.orders = torch.tensor(orders, dtype=torch.long)
        self.families = torch.tensor(families, dtype=torch.long)
        self.max_len = max_len
        
        # Standard DNA nucleotide token mapping (Vocab size = 5)
        self.vocab = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx][:self.max_len]
        # Map tokens, default to 'N' (4) if an unexpected character appears
        token_ids = [self.vocab.get(base, 4) for base in seq]
        
        # Pad with 4 ('N') if short
        if len(token_ids) < self.max_len:
            token_ids += [4] * (self.max_len - len(token_ids))
            
        return torch.tensor(token_ids, dtype=torch.long), self.classes[idx], self.orders[idx], self.families[idx]

# --- Step 2: Sequence Slicing Alignment ---
def prepare_dl_data(df, drep_dir, window_size=10000, stride=5000):
    available_files = os.listdir(drep_dir)
    all_seqs, class_list, order_list, family_list = [], [], [], []
    
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
        
    return all_seqs, class_list, order_list, family_list

# --- Step 3: Main Training Loop ---
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
    
    # 1. Load Data
    df = load_metadata(METADATA_PATH)
    train_df, val_df, test_df = stratified_split(df)
    
    # Fit Label Encoders across all tasks
    le_class = LabelEncoder().fit(df['Class'])
    le_order = LabelEncoder().fit(df['Order'])
    le_family = LabelEncoder().fit(df['Family'])
    
    # 2. Extract and Align Windows
    print("\n[Processing] Extracting genomic windows for Deep Learning...")
    train_seqs, train_c, train_o, train_f = prepare_dl_data(train_df, DREP_DIR)
    val_seqs, val_c, val_o, val_f = prepare_dl_data(val_df, DREP_DIR)
    
    print(f"-> Train slices: {len(train_seqs)}, Validation slices: {len(val_seqs)}")
    
    # Transform labels to integers
    tr_c = le_class.transform(train_c)
    tr_o = le_order.transform(train_o)
    tr_f = le_family.transform(train_f)
    
    v_c = le_class.transform(val_c)
    v_o = le_order.transform(val_o)
    v_f = le_family.transform(val_f)
    
    # 3. Create PyTorch Dataloaders
    train_dataset = GiantVirusDataset(train_seqs, tr_c, tr_o, tr_f)
    val_dataset = GiantVirusDataset(val_seqs, v_c, v_o, v_f)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 4. Model Selection (Toggle between 'cnn', 'transformer', 'hyperbolic')
    MODEL_CHOICE = 'hyperbolic' 
    
    # Standardizing vocab_size=5 since we process raw sequence tokens (A, C, G, T, N)
    if MODEL_CHOICE == 'cnn':
        model = GiantVirus1DCNN(vocab_size=5, num_classes=len(le_class.classes_), num_orders=len(le_order.classes_), num_families=len(le_family.classes_))
    elif MODEL_CHOICE == 'transformer':
        model = GiantVirusTransformer(vocab_size=5, num_classes=len(le_class.classes_), num_orders=len(le_order.classes_), num_families=len(le_family.classes_))
    elif MODEL_CHOICE == 'hyperbolic':
        model = GiantVirusHyperbolicNet(vocab_size=5, num_classes=len(le_class.classes_), num_orders=len(le_order.classes_), num_families=len(le_family.classes_))
        
    model = model.to(device)
    
    # 5. Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # 6. Execution Loop
    EPOCHS = 10
    print(f"\n[Training] Commencing training for {MODEL_CHOICE.upper()} over {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_x, batch_c, batch_o, batch_f in train_loader:
            batch_x = batch_x.to(device)
            batch_c = batch_c.to(device)
            batch_o = batch_o.to(device)
            batch_f = batch_f.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass yields multi-head logits
            out_c, out_o, out_f = model(batch_x)
            
            # Aggregate joint multi-task losses
            loss_c = criterion(out_c, batch_c)
            loss_o = criterion(out_o, batch_o)
            loss_f = criterion(out_f, batch_f)
            
            # Joint optimization target
            joint_loss = loss_c + loss_o + loss_f
            
            joint_loss.backward()
            optimizer.step()
            
            total_loss += joint_loss.item()
            
        # Validation evaluation step (Comprehensive multi-level tracking)
        model.eval()
        val_correct_c = 0
        val_correct_o = 0
        val_correct_f = 0
        
        with torch.no_grad():
            for bx, bc, bo, bf in val_loader:
                bx = bx.to(device)
                bc, bo, bf = bc.to(device), bo.to(device), bf.to(device)
                
                out_c, out_o, out_f = model(bx)
                
                val_correct_c += (out_c.argmax(dim=-1) == bc).sum().item()
                val_correct_o += (out_o.argmax(dim=-1) == bo).sum().item()
                val_correct_f += (out_f.argmax(dim=-1) == bf).sum().item()
                
        val_acc_c = val_correct_c / len(val_dataset)
        val_acc_o = val_correct_o / len(val_dataset)
        val_acc_f = val_correct_f / len(val_dataset)
        
        print(f"Epoch {epoch+1:02d}/{EPOCHS:02d} | Train Loss: {total_loss/len(train_loader):.4f} | "
              f"Val Acc -> Class: {val_acc_c*100:.1f}% | Order: {val_acc_o*100:.1f}% | Family: {val_acc_f*100:.1f}%")

    # Save trained model matrix weights
    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, f"giant_virus_{MODEL_CHOICE}.pth"))
    print(f"\n[SUCCESS] Model weights exported to '{MODEL_SAVE_DIR}/giant_virus_{MODEL_CHOICE}.pth'!")

if __name__ == "__main__":
    main()
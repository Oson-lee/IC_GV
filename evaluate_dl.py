# File path: evaluate_dl.py

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from math import pi

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
            slices = process_fasta_file(fasta_path, window_size=max_len, stride=5000)
            if len(slices) == 0:
                continue
                
            tensor_list = []
            for chunk in slices:
                token_ids = [vocab.get(base, 4) for base in chunk[:max_len]]
                if len(token_ids) < max_len:
                    token_ids += [4] * (max_len - len(token_ids))
                tensor_list.append(token_ids)
                
            genome_batch = torch.tensor(tensor_list, dtype=torch.long).to(device)
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16):
                out_c, out_o, out_f = model(genome_batch)
                prob_c = torch.softmax(out_c, dim=-1).mean(dim=0)
                prob_o = torch.softmax(out_o, dim=-1).mean(dim=0)
                prob_f = torch.softmax(out_f, dim=-1).mean(dim=0)
            
            pred_c_idx = torch.argmax(prob_c).item()
            pred_o_idx = torch.argmax(prob_o).item()
            pred_f_idx = torch.argmax(prob_f).item()
            
            true_classes.append(le_class.transform([row['Class']])[0])
            true_orders.append(le_order.transform([row['Order']])[0])
            true_families.append(le_family.transform([row['Family']])[0])
            
            pred_classes.append(pred_c_idx)
            pred_orders.append(pred_o_idx)
            pred_families.append(pred_f_idx)
            
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
    PICTURES_DIR = "./pictures" # Directory pre-created by the user
    
    df = load_metadata(METADATA_PATH)
    _, _, test_df = stratified_split(df)
    
    le_class = LabelEncoder().fit(df['Class'])
    le_order = LabelEncoder().fit(df['Order'])
    le_family = LabelEncoder().fit(df['Family'])
    
    architectures = ['1d_cnn', 'transformer', 'hyperbolic']
    raw_metrics = {}
    ablation_records = []
    
    for m_type in architectures:
        model_path = os.path.join(MODEL_SAVE_DIR, f"giant_virus_{m_type}.pth")
        if not os.path.exists(model_path):
            print(f"[WARNING] Model checkpoint missing for {m_type.upper()}. Skipping.")
            continue
            
        if m_type == '1d_cnn':
            model = GiantVirus1DCNN(vocab_size=5, num_classes=len(le_class.classes_), num_orders=len(le_order.classes_), num_families=len(le_family.classes_))
        elif m_type == 'transformer':
            model = GiantVirusTransformer(vocab_size=5, num_classes=len(le_class.classes_), num_orders=len(le_order.classes_), num_families=len(le_family.classes_))
        elif m_type == 'hyperbolic':
            model = GiantVirusHyperbolicNet(vocab_size=5, num_classes=len(le_class.classes_), num_orders=len(le_order.classes_), num_families=len(le_family.classes_))
            
        acc_class, acc_order, acc_family = evaluate_genome_level_soft_voting(
            model, model_path, test_df, DREP_DIR, le_class, le_order, le_family, device
        )
        
        raw_metrics[m_type.upper()] = [acc_class, acc_order, acc_family]
        
        ablation_records.append({
            "Architecture Backbone": m_type.upper(),
            "Class Accuracy": f"{acc_class:.2f}%",
            "Order Accuracy": f"{acc_order:.2f}%",
            "Family Accuracy": f"{acc_family:.2f}%"
        })
        
    # --- 1. Print Standard Terminal Report Table ---
    summary_df = pd.DataFrame(ablation_records)
    print("\n" + "="*20 + " FINAL ABLATION STUDY RESULTS (GENOME-LEVEL) " + "="*20)
    print(summary_df.to_markdown(index=False))
    print("="*65)
    
    # Global visual constants setup
    taxonomic_levels = ['Class', 'Order', 'Family']
    colors = {'1D_CNN': '#1f77b4', 'TRANSFORMER': '#ff7f0e', 'HYPERBOLIC': '#d62728'}
    
    # --- 2. Plot Type A: Grouped Bar Chart ---
    print("\n[Visualization 1/3] Generating Grouped Bar Chart...")
    x = np.arange(len(taxonomic_levels))
    bar_width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects_cnn = ax.bar(x - bar_width, raw_metrics.get('1D_CNN', [0,0,0]), bar_width, label='1D_CNN (Local Motifs)', color=colors['1D_CNN'], edgecolor='black', alpha=0.9)
    rects_trans = ax.bar(x, raw_metrics.get('TRANSFORMER', [0,0,0]), bar_width, label='TRANSFORMER (Global Attention)', color=colors['TRANSFORMER'], edgecolor='black', alpha=0.9)
    rects_hype = ax.bar(x + bar_width, raw_metrics.get('HYPERBOLIC', [0,0,0]), bar_width, label='HYPERBOLIC (Hierarchical Ball)', color=colors['HYPERBOLIC'], edgecolor='black', alpha=0.9)
    
    ax.set_ylabel('Genome-Level Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Ablation Analysis: Quantitative Performance Breakdown', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(taxonomic_levels, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 115)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', fontsize=10)
    
    def attach_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
    attach_labels(rects_cnn)
    attach_labels(rects_trans)
    attach_labels(rects_hype)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PICTURES_DIR, "ablation_bar_chart.png"), dpi=300)
    plt.close()

    # --- 3. Plot Type B: Hierarchical Decay Path (Line Plot) ---
    print("[Visualization 2/3] Generating Taxonomic Decay Trend Plot...")
    fig, ax = plt.subplots(figsize=(9, 6))
    
    for m_type in ['1D_CNN', 'TRANSFORMER', 'HYPERBOLIC']:
        if m_type in raw_metrics:
            ax.plot(taxonomic_levels, raw_metrics[m_type], marker='o', markersize=8, linewidth=2.5, 
                    label=f"{m_type} Resolution Path", color=colors[m_type])
            for i, val in enumerate(raw_metrics[m_type]):
                ax.annotate(f'{val:.1f}%', (taxonomic_levels[i], val), textcoords="offset points", 
                            xytext=(0,10), ha='center', fontsize=9, fontweight='bold', color=colors[m_type])
                            
    ax.set_ylabel('Genome-Level Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Taxonomic Hierarchy Depth', fontsize=12, fontweight='bold')
    ax.set_title('Hierarchical Resolution Decay Analysis across Ranks', fontsize=13, fontweight='bold', pad=15)
    ax.set_ylim(-5, 115)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PICTURES_DIR, "taxonomic_decay_trend.png"), dpi=300)
    plt.close()

    # --- 4. Plot Type C: Architectural Performance Radar Chart ---
    print("[Visualization 3/3] Generating Performance Radar Profile Graph...")
    categories = ['Class Accuracy', 'Order Accuracy', 'Family Accuracy']
    num_vars = len(categories)
    
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1] # Close the circular geometry loop
    
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    
    plt.xticks(angles[:-1], categories, color='black', size=11, fontweight='bold')
    ax.set_rlabel_position(30)
    plt.yticks([25, 50, 75, 100], ["25%", "50%", "75%", "100%"], color="grey", size=9)
    plt.ylim(0, 105)
    
    for m_type in ['1D_CNN', 'TRANSFORMER', 'HYPERBOLIC']:
        if m_type in raw_metrics:
            values = raw_metrics[m_type]
            values_closed = values + values[:1] # Close the circular geometry loop
            ax.plot(angles, values_closed, linewidth=2, linestyle='solid', label=m_type, color=colors[m_type])
            ax.fill(angles, values_closed, color=colors[m_type], alpha=0.1)
            
    plt.title('Global Architectural Performance Profile Matrix', fontsize=13, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.15), fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PICTURES_DIR, "model_performance_radar.png"), dpi=300)
    plt.close()
    
    print(f"\n[SUCCESS] Execution finished! Three distinct scientific plots are saved in your './pictures' folder.")
# Internal Taxonomic Classification of Giant Viruses

This project is the undergraduate course design for "Introduction to Statistical Learning and Data Science". The goal is to design and implement an efficient machine learning and deep learning system capable of classifying given genomic FASTA (`.fna`) files of giant viruses into their internal taxonomic levels (Class, Order, Family).

## Core Objectives

- **Multi-level Classification Task**: Given a giant virus genomic sequence of arbitrary length, the model must simultaneously predict its corresponding Class, Order, and Family, ensuring hierarchical consistency (Consistency Checking).
- **Handling Variable Length Inputs**: The system must be capable of processing inputs ranging from short contigs (< 1kbp) to complete genomes (> 1Mbp).
- **Advanced Geometric Deep Learning Exploration**: Explore and compare the performance differences between traditional Euclidean space and Hyperbolic space (e.g., Poincaré Embeddings) in capturing the tree-like taxonomic hierarchical structure.

## Project Architecture

The project adopts a flat and modular structure to ensure that data flows, feature engineering, and model codes do not interfere with each other, facilitating ablation studies:

```
GiantVirus_Classification/
├── data/                            # Data Directory
│   ├── origin_genomes/              # Original downloaded .fna genome sequence files
│   ├── drep_genomes/                # High-quality representative sequences de-replicated via dRep (95% ANI)
│   └── metadata.csv                 # Official label metadata table (Genome ID -> Class/Order/Family mapping)
│
├── notebooks/                       # Exploratory Data Analysis (EDA) and Drafts
│   └── 01_EDA_and_Splitting.ipynb   # Scripts for sequence length distribution, label imbalance, and data splitting
│
├── src/                             # Core Source Code Directory
│   ├── __init__.py                  # Makes src an importable Python package
│   ├── data_pipeline.py             # Reads metadata and performs Stratified Split for datasets
│   ├── sequence_processor.py        # Handles sliding window slicing for long sequences and padding for short ones
│   ├── feature_engineering.py       # Extracts genomic features (e.g., K-mer frequencies, TF-IDF matrices)
│   ├── model_baseline.py            # Traditional ML baselines (Random Forest, XGBoost, Linear SVM)
│   ├── model_deep_learning.py       # Deep learning models (1D-CNN, Transformer, Hyperbolic Space Embeddings)
│   └── evaluation.py                # Evaluation module (Top-1/3 accuracy, Macro/Micro F1, voting aggregation)
│
├── saved_models/                    # Stores trained model weights and serialized objects (.pkl, .pth)
├── results/                         # Stores generated confusion matrices, ablation study reports, etc.
├── requirements.txt                 # Project environment dependencies list
└── README.md                        # This project documentation file
```

## Core Execution Pipeline

### Stage 1: Data Preprocessing & De-replication

1. Use the **dRep** tool to de-replicate original sequences in `data/origin_genomes/` at a **95% ANI (Average Nucleotide Identity)** threshold, filtering out highly homologous strains. Save results to `data/drep_genomes/`.
2. Extract the representative genomes of the de-replicated sequences and match their IDs with `data/metadata.csv`.

### Stage 2: Data Splitting & Sequence Slicing

1. **Preventing Data Leakage**: Before slicing sequences, perform Stratified Sampling based on species/sample origin in `src/data_pipeline.py` to strictly divide the dataset into Train / Val / Test sets.
2. **Length Alignment**: Implement a sliding window strategy in `src/sequence_processor.py` (e.g., window size 10 kbp, stride 5 kbp). Sequences larger than the window are sliced and inherit parent labels; sequences smaller than the window are padded with `N` bases.

### Stage 3: Multi-method Feature & Model Exploration

The project will iteratively conduct comparative experiments in the following order:

1. **Baseline Models**: K-mer counts (k=3~6) + Term Frequency-Inverse Document Frequency (TF-IDF) ➜ Random Forest / XGBoost / SVM.
2. **Deep Learning**: Nucleotide 1-hot encoding or trainable embeddings ➜ Lightweight 1D-CNN.
3. **Sequence Language Models**: Treat K-mers as tokens ➜ Fine-tune a lightweight Transformer classifier.
4. **Hyperbolic Geometric Embeddings (Core Research Direction)**: Use the Poincaré model to map genomic sequence features into hyperbolic space to fit the strong tree-like biological topology (Class-Order-Family), using a hyperbolic classifier for prediction.

### Stage 4: Hierarchical Alignment & Multi-slice Voting

1. **Multi-slice Prediction Aggregation**: During Inference, a complete `.fna` genome is sliced into multiple fragments. The model predicts all fragments, and `src/evaluation.py` reconstructs the final classification of the genome via Voting or Averaging probabilities.
2. **Consistency Checking**: Verify whether the model's outputs across the Class, Order, and Family levels comply with true biological taxonomic inclusion relationships, eliminating invalid classification paths.

## Evaluation & Experimental Design

The system will output the following metrics for a comprehensive performance evaluation:

- **Basic Classification Metrics**: Top-1 / Top-3 Accuracy, Macro F1-Score, Micro F1-Score.
- **Hierarchy-specific Metrics**: Hierarchical Precision, Hierarchical Recall.
- **Ablation Studies**: Compare the impact of different features (K-mer vs. Embedding) and different geometric space mappings (Euclidean vs. Hyperbolic) on capturing multi-level structures.

## Quick Start / Environment Setup

Please ensure Python 3.8+ is installed locally, and run the following command to install basic dependencies:

```
pip install -r requirements.txt
```

*Main dependencies include: biopython, scikit-learn, xgboost, pandas, numpy, torch (if using deep learning)*

## Project Lead & Contact Information

- **Xufeng Kong** ([kxf.scut@outlook.com](mailto:kxf.scut@outlook.com))
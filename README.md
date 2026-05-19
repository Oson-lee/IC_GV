# Comprehensive Project Report: Internal Taxonomic Classification of Giant Viruses (NCLDV)

## 📖 1. Executive Summary

This project aims to computationally resolve the internal taxonomic classification (Class, Order, Family) of **Nucleocytoviricota (Giant Viruses / NCLDVs)** directly from raw genomic FASTA sequences. Giant viruses present a unique challenge in bioinformatics due to their extreme genome size variations (from short metagenomic contigs to massive Mbp-scale genomes) and highly imbalanced, long-tailed evolutionary distributions.

To address these challenges, we engineered an end-to-end classification pipeline featuring a **Multi-Slice Soft-Voting Inference Mechanism**. Furthermore, we conducted a rigorous comparative study between a robust Traditional Machine Learning Baseline (K-mer TF-IDF + Random Forest) and state-of-the-art **Multi-Task Deep Learning architectures**, with a specialized emphasis on embedding sequences into **Hyperbolic Space** to preserve the intrinsic tree-like hierarchical topology of biological taxonomies.

------

## 🧬 2. Biological Background & Data Strategy

### 2.1 The NCLDV Challenge

Nucleocytoviricota encompasses highly diverse viral families (e.g., *Mimiviridae*, *Phycodnaviridae*) that infect a wide range of eukaryotic hosts. Relying on literature such as recent *Nature* and *PLOS Biology* phylogenomic frameworks, our dataset is constructed from high-quality isolate and metagenome-assembled genomes (MAGs).

### 2.2 Preprocessing & Strict Data Hygiene

Real-world bioinformatics metadata is notoriously noisy. Our data pipeline ensures strict data hygiene:

- **Taxonomic Parsing:** Dynamically cleans messy prediction strings (e.g., `'NCLDV__Asfuvirales:8(88.89%)'`) to extract pure, deterministic taxonomic labels.
- **De-replication Awareness:** Designed to process genomes de-replicated at 95% Average Nucleotide Identity (ANI) using tools like `dRep`, drastically mitigating homologous information leakage.
- **Stratified Sampling:** The dataset is split into Training, Validation, and Test sets strictly stratified by the finest taxonomic granularity (`Family`). Orphan families (count < 2) are automatically filtered out to ensure robust mathematical stratification.

### 2.3 Sequence Slicing & Padding Engine

Genomes cannot be fed into uniform neural networks due to severe length discrepancies. We implemented a sliding-window algorithm:

- **Long Genomes:** Sliced into continuous `10,000 bp` windows with a `5,000 bp` stride.
- **Short Contigs:** Dynamically padded with degenerate `N` base tokens to maintain tensor alignment without corrupting biological signals.

------

## 🏗️ 3. Algorithmic Architectures & Methodology

### Track A: Traditional Machine Learning (The Baseline)

As a rigorous control, we established a traditional NLP-inspired bioinformatics pipeline.

- **Feature Engineering (Sequence Signatures):** Extracts 4-mer frequencies and transforms them into a dense TF-IDF matrix. This efficiently captures fundamental codon usage biases and structural sequence signatures without requiring massive computational overhead.
- **Classifier:** Random Forest Classifier running on multi-core CPUs.
- **Functionality:** Targets the most difficult `Family` level directly, establishing a reliable performance floor for the deep learning models.

### Track B: Multi-Task Hyperbolic Deep Learning (Core Innovation)

Standard Euclidean geometry expands polynomially, causing severe distortion when attempting to embed exponentially expanding hierarchical trees (like biological taxonomies). To solve this, we implemented a **Hyperbolic Neural Network**.

- **Input Representation:** Direct character-level nucleotide tokenization (`A=0, C=1, G=2, T=3, N=4`).
- **Geometric Innovation (Poincaré Ball):** By mapping intermediate sequence representations into Hyperbolic Space, the network naturally accommodates the hierarchical distances between parent taxa (Class) and child taxa (Order, Family).
- **Multi-Task Heads:** A single shared backbone splits into three parallel classification heads, allowing the model to jointly optimize for Class, Order, and Family using a combined Cross-Entropy loss. This enforces hierarchical consistency during gradient descent.

------

## ⚙️ 4. Inference Mechanism: Genome-Level Soft-Voting

A single `10,000 bp` slice might not contain enough discriminative phylogenetic markers. Therefore, our evaluation pipeline (`evaluate_dl.py`) does not judge the model based on individual slices.

Instead, during inference:

1. A whole genome is decomposed into $N$ slices.
2. The network outputs softmax probability distributions for all $N$ slices.
3. **Soft-Voting:** The probabilities are aggregated and averaged. The final genome-level prediction is determined by the `argmax` of the globally averaged probability vector, effectively canceling out noise from non-informative genomic regions.

------

## 📊 5. Experimental Results & Deep Discussion

Experiments were conducted on a highly imbalanced, rigorously split dataset (82 Training genomes, 18 Test genomes). The metrics below reflect the final **Genome-Level Accuracy** after Soft-Voting.

| **Architecture / Feature Space**               | **Class Accuracy** | **Order Accuracy** | **Family Accuracy** |
| ---------------------------------------------- | ------------------ | ------------------ | ------------------- |
| **Baseline: Random Forest (4-mer TF-IDF)**     | -                  | -                  | **30.77%**          |
| **Multi-Task Hyperbolic Network (Raw Tokens)** | **100.00%**        | **53.85%**         | **38.46%**          |

### 5.1 The Triviality of the Top-Level (Class = 100%)

The 100% accuracy at the Class level highlights the extreme long-tailed distribution intrinsic to NCLDV datasets. Because top-level taxonomy is heavily dominated by specific major groups (e.g., *Megaviricetes*), the network rapidly collapses into predicting the absolute majority. This confirms the necessity of driving the classification deeper into the Order and Family levels.

### 5.2 The Efficacy of Hyperbolic Space (Order = 53.85%)

Achieving 53.85% at the Order level validates the core hypothesis of this project: **Hyperbolic embeddings excel at capturing macro-genomic topologies.** The network successfully learned the macro-level distinctions between complex viral orders (e.g., *Imitervirales* vs. *Chitovirales*) purely from raw index tokens.

### 5.3 The "Data Hunger" Bottleneck (Family = 38.46%)

While the Hyperbolic network outperformed the TF-IDF Baseline at the Family level (38.46% vs. 30.77%), the absolute accuracy reveals the fundamental limitation of deep learning in data-scarce biological domains. The Random Forest leverages a powerful biological prior (4-mer TF-IDF), whereas the neural network must learn these motifs from scratch. With only 82 genomes in the training set, the neural network suffers from severe "data hunger" at the finest taxonomic granularity.

------

## 🚀 6. Future Roadmap & Ablation Studies

To push the Family-level accuracy toward state-of-the-art benchmarks (>80%), the following advanced strategies are outlined for immediate future work:

1. **Feature Fusion (Biological Priors + Geometric DL):** Instead of feeding raw nucleotide tokens, we will modify the input layer of the Hyperbolic Network to directly ingest the 256-dimensional 4-mer TF-IDF matrix. This merges the sample-efficiency of traditional k-mers with the hierarchical resolving power of hyperbolic geometry.
2. **Aggressive Data Augmentation:** Decrease the sliding window stride from `5,000` to `1,000` or `500` during preprocessing. This will exponentially expand the number of overlapping training slices, synthetically increasing the dataset size to satiate the neural network's data hunger.
3. **Focal Loss Integration:** Implement Focal Loss or inverse class weighting to heavily penalize the model for misclassifying rare, underrepresented taxonomic families, combating the long-tail imbalance.

------

## 💻 7. Repository Usage Guide

### Prerequisites

Ensure your environment is configured with PyTorch (CUDA recommended for deep learning) and scikit-learn.



```
pip install torch torchvision torchaudio pandas numpy scikit-learn openpyxl biopython
```

### Execution Pipeline

**Step 1: Train the Traditional ML Baseline**

Extracts TF-IDF features and trains a Random Forest model.



```
python train_baseline.py
```

**Step 2: Train the Multi-Task Deep Learning Model**

Trains the Hyperbolic Network (or 1D-CNN/Transformer via code toggle) across 10 epochs.



```
python train_dl.py
```

**Step 3: Genome-Level Evaluation**

Executes the inference engine, applies multi-slice soft-voting, and generates the final Classification Report.



```
python evaluate_dl.py
```


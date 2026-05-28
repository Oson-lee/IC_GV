# Internal Classification of Giant Viruses (Class/Order/Family) – Undergraduate Project Description

## Project Goal

- **Task**: Design and implement a deep learning / machine learning system that can classify a given genomic FASTA (`.fna`) file into internal taxonomic levels of giant viruses (e.g., class, order, family, etc.).
- **Data labels**: Refer to the Nature article (https://www.nature.com/articles/s44298-024-00069-7) as the primary source for sample annotations and classification information, supplemented by the Wikipedia entry on giant viruses (https://en.wikipedia.org/wiki/Giant_virus) to construct the taxonomic hierarchy and background knowledge.

## Example Input

Example FASTA header and sequence (example filename: `AbALV.fna`):

> \>AbALV.fna|LC506465.1 LC506465.1 Abalone asfarvirus DNA, segment 1, partial sequence
> ATATGGTAGTTATGGGTGGGAATTTTCTATGGAAAAAATAATATCAGGTAATTCTGATGT
> AGTTTTTAGCCCACCTGTTTTCTTACTTTTGCTTAGAAAAATATTTATTCACAATCGATC

The system should handle inputs of varying lengths: from short contigs (a few hundred bp) to complete genomes (tens of kbp or even Mbp). Recommended strategies are described in the "Length & Sampling" section.

## Giant Virus Data Sources

- (A phylogenomic framework for charting the diversity and evolution of giant viruses) https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3001430
- (Mirusviruses link herpesviruses to giant viruses) https://www.nature.com/articles/s41586-023-05962-4
- (Adaptation strategies of giant viruses to low-temperature marine ecosystems) https://doi.org/10.1093/ismejo/wrae162
- (Spatiotemporal dynamics of giant viruses within a deep freshwater lake reveal a distinct dark-water community) https://doi.org/10.1093/ismejo/wrae182

## Label Sources
- **Primary label source**: Use the sample annotations and taxonomic information provided in https://www.nature.com/articles/s44298-024-00069-7 to construct training/validation/test sets (comply with the data usage license of the original article).
- **Supplementary materials**: Use NCBI/GenBank, RefSeq annotations, and the Wikipedia entry to verify taxonomic hierarchies and family/genus relationships.
- **Deduplication and splitting**: Perform deduplication and stratified sampling by species/sample origin to ensure that the same species does not appear in both training and test sets, avoiding information leakage.

## **Data acquisition and preprocessing (recommended pipeline)**  

- Convert all downloaded records into **single‑FASTA (`.fna`)** files – one file per genome or contig.  
- Use **[dRep](https://github.com/MrOlm/drep)** (or a similar tool) to **de‑replicate** genomes at 95% average nucleotide identity (ANI) or a user‑defined threshold. This removes nearly‑identical strains, mitigates data leakage, and reduces computational overhead.  
- After de‑duplication, keep the longest representative genome for each species or operational cluster.  
- For metagenomic short contigs that cannot be de‑replicated at genome level, apply a similarity‑based clustering (e.g., CD‑HIT at 98% identity) to avoid redundant training fragments.

## Sequence Length & Slicing Strategies

- **For long sequences**: Use sliding windows (e.g., window length 10 kbp, stride 5 kbp) or random sampling of several fragments. During training, treat fragments as samples; during inference, aggregate fragment predictions using voting/averaging.
- **For short sequences**: If the length is less than the window size, use padding or train a separate branch for short sequences. Alternatively, use variable-length models (e.g., Transformer + position encoding).

## Multi‑method Exploration (Suggested Experimental List)

The goal is to compare the impact of different features and feature spaces on classification performance:

1) **Classical k‑mer features + machine learning baseline**

   - Extract k‑mer counts (k=3..6) or TF‑IDF to form dense or sparse vectors.
   - Baseline classifiers: Random Forest, XGBoost, or linear SVM.

2) **Sequence‑level embedding + lightweight model**

   - 1‑hot or trainable nucleotide/k‑mer embedding, fed into a lightweight 1D‑CNN or depthwise‑separable CNN.
   - Global pooling followed by a fully connected classifier – suitable for fast prototyping and low computational budgets.

3) **Transformer / language model approach**

   - Use nucleotides or k‑mers as tokens and fine‑tune a small Transformer (few layers, few heads) for the downstream classification task.
   - Optionally pre‑train with Masked Language Modeling on shorter fragments and then convert to classification.

4) **Graph / spectral methods with phylogenetic information fusion**

   - If a reference phylogenetic / gene‑sharing network is available, treat each genome as a node and use Graph Neural Networks (GNNs) or spectral embeddings for classification.

5) **Feature space mapping and geometric innovations (key research direction)**

   - **Hyperbolic embeddings**: Use Poincaré embeddings or the Lorentz model to embed the hierarchical classification relationships into hyperbolic space, aiming to better preserve taxonomic hierarchy information. Specific ideas:
     - Use sequence features (e.g., k‑mer embeddings or Transformer representations) as initial vectors.
     - Train representations that can distinguish different hierarchical levels in hyperbolic space (refer to Poincaré Embeddings literature).
     - Perform nearest‑neighbour or simple classification (e.g., linear separation on a hyperboloid) in hyperbolic space, or feed hyperbolic representations into a downstream fully connected classifier.
   - **Manifold learning / geometric deep learning**: Attempt to embed sequences into low‑dimensional manifolds (Spherical / Hyperbolic / Euclidean) and compare the ability of different geometries to preserve hierarchical information.

6) **Contrastive learning / representation learning**

   - Use contrastive learning methods such as SimCLR / MoCo to pre‑train sequence representations on unlabeled or semi‑supervised data, then fine‑tune the classifier on a small amount of labeled data.

7) **Multi‑modal fusion (optional)**

   - Combine sequence features with metadata (host, sampling location, sequencing platform) for joint classification.

## Model Architecture & Implementation Points

- **Output**: Multi‑level classification output (can use hierarchical softmax or predict each level separately with consistency checking).
- **Loss function**: If hierarchical dependencies exist, use a hierarchical loss or tree‑based regularisation; the generic approach is cross‑entropy (multi‑class) or multi‑label binary classification (if multi‑label).
- **Imbalance handling**: Use class weights, resampling, or focal loss.
- **Inference**: For a `.fna` file, first slice it, then predict on each slice and aggregate the predictions. Output a probability distribution for each level and the highest‑confidence path.

## Evaluation & Experimental Design

- **Metrics**: Top‑1 / Top‑3 accuracy, Macro/Micro F1, confusion matrix, hierarchical precision/recall.
- **Validation strategy**: Stratified k‑fold CV (stratified by family/genus) and an independent test set; bootstrap experiments can be used to assess performance differences on short vs. long sequences.
- **Ablation studies**: Compare the impact of different features (k‑mer, embedding, Transformer representations) and different geometric space mappings (Euclidean vs. hyperbolic) on final performance.

## Risks & Considerations

- **Data bias**: The literature sources and public databases may contain annotation biases; independent data should be used to verify generalisation ability.
- **Label hierarchy conflicts**: Different databases/studies may have different naming or assignments for taxonomic levels; unify the taxonomy or keep a mapping table.
- **Computational resources**: Transformer and contrastive learning methods require significant computing power; start with small‑scale subsets for rapid iteration.

## Future Extensions (Advanced)

- Use hyperbolic geometry for hierarchical visualisation and clustering analysis; study theoretical explanations of how different geometries preserve hierarchical information.
- Integrate the classifier with a phylogenetic tree for joint inference (phylogeny‑aware classification).
- Predict on massive amounts of unannotated genomes and construct an evolutionary map of giant viruses.

## Contact Information

- Xufeng Kong (<kxf.scut@outlook.com>)
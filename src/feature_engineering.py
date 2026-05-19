import numpy as np
from itertools import product
from sklearn.feature_extraction.text import TfidfTransformer
from typing import List

class KmerFeatureExtractor:
    """
    K-mer feature extractor. Converts DNA sequences into K-mer frequency vectors.
    """
    def __init__(self, k: int = 4):
        self.k = k
        self.alphabet = 'ACGT'
        # Generate all possible K-mer combinations as feature dimensions
        self.kmers = [''.join(p) for p in product(self.alphabet, repeat=self.k)]
        self.kmer_to_idx = {kmer: idx for idx, kmer in enumerate(self.kmers)}
        self.vocab_size = len(self.kmers)

    def extract_counts(self, sequence: str) -> np.ndarray:
        """
        Extract the K-mer frequency vector for a single sequence.
        """
        # Remove non-standard bases (N, etc.)
        clean_seq = ''.join([base for base in sequence if base in self.alphabet])
        
        counts = np.zeros(self.vocab_size, dtype=np.float32)
        if len(clean_seq) < self.k:
            return counts
            
        # Count K-mer occurrences
        for i in range(len(clean_seq) - self.k + 1):
            kmer = clean_seq[i:i+self.k]
            if kmer in self.kmer_to_idx:
                counts[self.kmer_to_idx[kmer]] += 1
                
        return counts

    def fit_transform_tfidf(self, sequences: List[str]) -> np.ndarray:
        """
        Extract K-mer features for a list of sequences and apply TF-IDF transformation.
        """
        # 1. Extract the frequency matrix for all sequences
        count_matrix = np.array([self.extract_counts(seq) for seq in sequences])
        
        # 2. Fit and apply TF-IDF
        self.tfidf = TfidfTransformer()
        tfidf_matrix = self.tfidf.fit_transform(count_matrix).toarray()
        
        return tfidf_matrix
        
    def transform_tfidf(self, sequences: List[str]) -> np.ndarray:
        """
        Apply the fitted TF-IDF transformation to new data (e.g., validation/test sets).
        """
        if not hasattr(self, 'tfidf'):
            raise ValueError("TF-IDF has not been fitted. Call fit_transform_tfidf first.")
            
        count_matrix = np.array([self.extract_counts(seq) for seq in sequences])
        tfidf_matrix = self.tfidf.transform(count_matrix).toarray()
        
        return tfidf_matrix
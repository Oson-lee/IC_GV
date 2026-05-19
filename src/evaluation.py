# File path: GiantVirus_Classification/src/evaluation.py

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from typing import List, Dict, Tuple

def aggregate_slice_probabilities(slice_probs: np.ndarray, method: str = 'average') -> np.ndarray:
    """
    Aggregate prediction probabilities from multiple slices of a single genome.
    
    Args:
        slice_probs (np.ndarray): Shape (num_slices, num_classes), probabilities for each slice.
        method (str): 'average' (soft voting) or 'max' (taking the maximum confidence).
        
    Returns:
        np.ndarray: Aggregated probability vector of shape (num_classes,).
    """
    if method == 'average':
        return np.mean(slice_probs, axis=0)
    elif method == 'max':
        return np.max(slice_probs, axis=0)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")

class HierarchicalConsistencyChecker:
    """
    Validates if the predicted (Class, Order, Family) path exists in the true biological taxonomy.
    """
    def __init__(self, taxonomy_tree: Dict[int, Dict[int, List[int]]]):
        """
        Args:
            taxonomy_tree: A nested dictionary representing valid paths.
                           Format: {class_id: {order_id: [family_id_1, family_id_2, ...]}}
        """
        self.taxonomy_tree = taxonomy_tree

    def is_valid_path(self, pred_class: int, pred_order: int, pred_family: int) -> bool:
        """
        Check if a specific taxonomic hierarchy prediction is valid.
        """
        if pred_class not in self.taxonomy_tree:
            return False
        if pred_order not in self.taxonomy_tree[pred_class]:
            return False
        if pred_family not in self.taxonomy_tree[pred_class][pred_order]:
            return False
        return True

    def force_consistent_prediction(self, class_probs: np.ndarray, 
                                    order_probs: np.ndarray, 
                                    family_probs: np.ndarray) -> Tuple[int, int, int]:
        """
        Finds the combination of (Class, Order, Family) that maximizes the joint probability 
        while strictly adhering to the biological taxonomy tree.
        """
        best_prob = -1.0
        best_path = (-1, -1, -1)
        
        # Iterate through all valid paths in the taxonomy tree
        for c_id, orders in self.taxonomy_tree.items():
            for o_id, families in orders.items():
                for f_id in families:
                    # Calculate joint probability (assuming independence for simplicity in soft-voting)
                    joint_prob = class_probs[c_id] * order_probs[o_id] * family_probs[f_id]
                    
                    if joint_prob > best_prob:
                        best_prob = joint_prob
                        best_path = (c_id, o_id, f_id)
                        
        return best_path

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate standard classification metrics for a specific taxonomic level.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average='macro', zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average='micro', zero_division=0)
    }
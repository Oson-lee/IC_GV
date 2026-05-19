# File path: GiantVirus_Classification/src/model_baseline.py

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from typing import Any

def build_random_forest(n_estimators: int = 100, random_state: int = 42) -> RandomForestClassifier:
    """
    Initialize and return a Random Forest classifier.
    Class weight is set to 'balanced' to handle potential label imbalance 
    in giant virus taxonomic levels.
    """
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight='balanced',
        n_jobs=-1  # Use all available CPU cores for faster training
    )
    return rf_model

def build_xgboost(random_state: int = 42) -> xgb.XGBClassifier:
    """
    Initialize and return an XGBoost classifier.
    Configured for multi-class classification using softmax probabilities.
    """
    xgb_model = xgb.XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        random_state=random_state,
        n_jobs=-1
    )
    return xgb_model

def build_svm(kernel: str = 'linear', random_state: int = 42) -> SVC:
    """
    Initialize and return a Support Vector Machine (SVM) classifier.
    Probability is set to True to allow for soft voting/averaging 
    across multiple sequence slices later during inference.
    """
    svm_model = SVC(
        kernel=kernel,
        probability=True,  # Crucial for returning probabilities instead of just hard labels
        random_state=random_state,
        class_weight='balanced'
    )
    return svm_model

def get_baseline_model(model_name: str, **kwargs) -> Any:
    """
    A factory function to retrieve the specified baseline model.
    
    Args:
        model_name (str): The name of the model ('rf', 'xgboost', or 'svm').
        **kwargs: Additional hyperparameters for the specific model.
        
    Returns:
        The initialized scikit-learn compatible model instance.
    """
    model_name = model_name.lower()
    if model_name in ['rf', 'random_forest']:
        return build_random_forest(**kwargs)
    elif model_name in ['xgboost', 'xgb']:
        return build_xgboost(**kwargs)
    elif model_name == 'svm':
        return build_svm(**kwargs)
    else:
        raise ValueError(f"Unsupported model name: {model_name}. Choose from 'rf', 'xgboost', or 'svm'.")
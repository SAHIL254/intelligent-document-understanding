"""
Utility Functions and Constants for NLP IDU Project
====================================================
Provides helper functions and project-wide constants.
"""

import os
from pathlib import Path
from typing import List, Dict, Any
import json
import pickle


# ============================================================================
# PATHS & DIRECTORIES
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, ARTIFACTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

MODEL_CONFIGS = {
    "text_classification": {
        "vectorizer_model": "tfidf",
        "classifier_model": "logistic_regression",
        "max_features": 5000,
        "ngram_range": (1, 2),
        "min_df": 5,
        "max_df": 0.7,
        "test_size": 0.2,
        "random_state": 42,
    },
    "ner": {
        "model_name": "en_core_web_sm",
        "entity_types": ["PERSON", "ORG", "GPE", "DATE", "MONEY", "PRODUCT"],
    },
    "summarization": {
        "transformer_model": "t5-small",
        "max_input_length": 512,
        "max_summary_length": 130,
        "min_summary_length": 40,
        "num_beams": 4,
        "device": "cpu",
    },
}


# ============================================================================
# TEXT PREPROCESSING CONSTANTS
# ============================================================================

STOPWORDS_EN = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once"
}

# Special characters to remove during preprocessing
SPECIAL_CHARS = r'[^a-zA-Z0-9\s]'


# ============================================================================
# FILE I/O UTILITIES
# ============================================================================

def save_json(data: Dict[str, Any], filepath: str) -> None:
    """
    Save dictionary as JSON file.
    
    Args:
        data (dict): Data to save
        filepath (str): Output file path
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load JSON file as dictionary.
    
    Args:
        filepath (str): Input file path
        
    Returns:
        dict: Loaded data
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def save_pickle(obj: Any, filepath: str) -> None:
    """
    Save Python object as pickle file.
    
    Args:
        obj: Object to save
        filepath (str): Output file path
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath: str) -> Any:
    """
    Load Python object from pickle file.
    
    Args:
        filepath (str): Input file path
        
    Returns:
        Object loaded from pickle
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


# ============================================================================
# TEXT UTILITIES
# ============================================================================

def validate_text_input(text: str, min_length: int = 20) -> bool:
    """
    Validate if input text meets minimum requirements.
    
    Args:
        text (str): Input text to validate
        min_length (int): Minimum text length required
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not text or len(text.strip()) < min_length:
        return False
    return True


def get_text_statistics(text: str) -> Dict[str, int]:
    """
    Calculate statistics about input text.
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Dictionary with word count, char count, sentence count
    """
    words = text.split()
    sentences = text.split('.')
    
    return {
        "char_count": len(text),
        "word_count": len(words),
        "sentence_count": len([s for s in sentences if s.strip()]),
        "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0,
    }


# ============================================================================
# DATA UTILITIES
# ============================================================================

def ensure_dataframe(data):
    """
    Ensure data is in pandas DataFrame format.
    
    Args:
        data: Input data (list, dict, or DataFrame)
        
    Returns:
        DataFrame: Data in DataFrame format
    """
    import pandas as pd
    
    if isinstance(data, pd.DataFrame):
        return data
    elif isinstance(data, list):
        return pd.DataFrame(data)
    elif isinstance(data, dict):
        return pd.DataFrame([data])
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")


# ============================================================================
# MODEL UTILITIES
# ============================================================================

def get_model_path(model_name: str) -> str:
    """
    Get full path to saved model file.
    
    Args:
        model_name (str): Name of the model
        
    Returns:
        str: Full path to model file
    """
    return str(MODELS_DIR / f"{model_name}.pkl")


def get_artifact_path(artifact_name: str, extension: str = "json") -> str:
    """
    Get full path to artifact file.
    
    Args:
        artifact_name (str): Name of the artifact
        extension (str): File extension
        
    Returns:
        str: Full path to artifact file
    """
    return str(ARTIFACTS_DIR / f"{artifact_name}.{extension}")
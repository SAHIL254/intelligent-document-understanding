"""
Utility Functions
=================

Helper functions for the NLP pipeline.
"""

import json
from typing import Any, Dict, List
from pathlib import Path
import logging


logger = logging.getLogger(__name__)


def setup_logging(level=logging.INFO):
    """
    Configure logging for the application.
    
    Args:
        level: Logging level
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def save_results_to_json(results: Dict[str, Any], filepath: str) -> None:
    """
    Save pipeline results to JSON file.
    
    Args:
        results: Results dictionary
        filepath: Path to save file
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")


def load_results_from_json(filepath: str) -> Dict[str, Any]:
    """
    Load pipeline results from JSON file.
    
    Args:
        filepath: Path to results file
        
    Returns:
        Results dictionary
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            results = json.load(f)
        logger.info(f"Results loaded from {filepath}")
        return results
    except Exception as e:
        logger.error(f"Failed to load results: {e}")
        return {}


def read_text_file(filepath: str) -> str:
    """
    Read text from file.
    
    Args:
        filepath: Path to text file
        
    Returns:
        File content as string
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except Exception as e:
        logger.error(f"Failed to read file: {e}")
        return ""


def batch_read_text_files(directory: str, pattern: str = "*.txt") -> Dict[str, str]:
    """
    Read multiple text files from directory.
    
    Args:
        directory: Directory path
        pattern: File pattern (default: *.txt)
        
    Returns:
        Dictionary with filenames as keys and contents as values
    """
    results = {}
    try:
        path = Path(directory)
        for filepath in path.glob(pattern):
            content = read_text_file(str(filepath))
            results[filepath.name] = content
        logger.info(f"Loaded {len(results)} files from {directory}")
    except Exception as e:
        logger.error(f"Failed to read files: {e}")
    
    return results


def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix for truncated text
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def format_entities_table(entities: List[Dict[str, Any]]) -> str:
    """
    Format entities as table string.
    
    Args:
        entities: List of entity dictionaries
        
    Returns:
        Formatted table string
    """
    if not entities:
        return "No entities found."
    
    from tabulate import tabulate
    
    headers = ["Text", "Label"]
    rows = [[e.get("text"), e.get("label")] for e in entities]
    
    return tabulate(rows, headers=headers, tablefmt="grid")


def calculate_processing_time(start_time: float, end_time: float) -> float:
    """
    Calculate processing time in seconds.
    
    Args:
        start_time: Start timestamp
        end_time: End timestamp
        
    Returns:
        Processing time in seconds
    """
    return round(end_time - start_time, 4)


def get_text_statistics(text: str) -> Dict[str, Any]:
    """
    Calculate text statistics.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary with statistics
    """
    words = text.split()
    sentences = text.split('.')
    
    return {
        "character_count": len(text),
        "word_count": len(words),
        "sentence_count": len([s for s in sentences if s.strip()]),
        "average_word_length": round(
            len(text) / len(words) if words else 0,
            2
        ),
        "average_sentence_length": round(
            len(words) / len([s for s in sentences if s.strip()]) 
            if sentences else 0,
            2
        )
    }


class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str = "Operation"):
        """
        Initialize timer.
        
        Args:
            name: Name of operation
        """
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        """Start timer."""
        import time
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        """Stop timer and log result."""
        import time
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        logger.info(f"{self.name} completed in {elapsed:.4f}s")


def validate_model_paths(model_path: str, vectorizer_path: str) -> bool:
    """
    Validate that model files exist.
    
    Args:
        model_path: Path to model file
        vectorizer_path: Path to vectorizer file
        
    Returns:
        True if both files exist
    """
    model_exists = Path(model_path).exists()
    vectorizer_exists = Path(vectorizer_path).exists()
    
    if not model_exists:
        logger.warning(f"Model file not found: {model_path}")
    if not vectorizer_exists:
        logger.warning(f"Vectorizer file not found: {vectorizer_path}")
    
    return model_exists and vectorizer_exists
"""
Text Classification Component
=============================

Handles text classification using scikit-learn models.
"""

import joblib
from typing import Optional
from pathlib import Path

from src.logger import LoggerMixin
from src.exceptions import (
    ModelLoadingError,
    ValidationError,
    ClassificationError
)


class TextClassifier(LoggerMixin):
    """
    Text classification component using TF-IDF vectorization and sklearn classifier.
    
    Attributes:
        model: Trained classifier model
        vectorizer: TF-IDF vectorizer
        model_path: Path to saved model
        vectorizer_path: Path to saved vectorizer
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        vectorizer_path: Optional[str] = None
    ):
        """
        Initialize TextClassifier.
        
        Args:
            model_path: Path to saved classifier model
            vectorizer_path: Path to saved TF-IDF vectorizer
        """
        self.setup_logger()
        
        self.model = None
        self.vectorizer = None
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        
        if model_path and vectorizer_path:
            self.load_models()
    
    def load_models(self) -> None:
        """
        Load pre-trained classifier and vectorizer from disk.
        
        Raises:
            ModelLoadingError: If model files don't exist or fail to load
        """
        if not self.model_path or not self.vectorizer_path:
            raise ModelLoadingError(
                "Model paths not specified",
                model_name="TextClassifier"
            )
        
        try:
            self.log_info(f"Loading classifier model from {self.model_path}")
            self.model = joblib.load(self.model_path)
            
            self.log_info(f"Loading vectorizer from {self.vectorizer_path}")
            self.vectorizer = joblib.load(self.vectorizer_path)
            
            self.log_info("Classification models loaded successfully")
        except FileNotFoundError as e:
            self.log_error(f"Model file not found: {e}")
            raise ModelLoadingError(
                f"Model file not found",
                model_name="TextClassifier",
                model_path=str(e),
                original_exception=e
            )
        except Exception as e:
            self.log_error(f"Failed to load models: {e}")
            raise ModelLoadingError(
                f"Failed to load classification models: {e}",
                model_name="TextClassifier",
                original_exception=e
            )
    
    def classify(self, text: str) -> str:
        """
        Classify input text into a category.
        
        Args:
            text: Input text to classify
            
        Returns:
            Predicted category label
            
        Raises:
            ValidationError: If text is invalid
            ClassificationError: If classification fails
        """
        if self.model is None or self.vectorizer is None:
            self.log_error("Models not loaded")
            raise ModelLoadingError(
                "Models not loaded. Call load_models() first.",
                model_name="TextClassifier"
            )
        
        if not text or len(text.strip()) < 5:
            self.log_warning(f"Text too short for classification: {len(text)} chars")
            raise ValidationError(
                "Text is too short (minimum 5 characters).",
                field="text",
                value=text[:50] if text else ""
            )
        
        try:
            self.log_debug(f"Classifying text of length {len(text)}")
            X = self.vectorizer.transform([text])
            prediction = self.model.predict(X)[0]
            
            self.log_info(f"Classification completed: {prediction}")
            return prediction
        except Exception as e:
            self.log_error(f"Classification failed: {e}")
            raise ClassificationError(
                f"Classification failed: {e}",
                original_exception=e
            )
    
    def classify_batch(self, texts: list) -> list:
        """
        Classify multiple texts.
        
        Args:
            texts: List of texts to classify
            
        Returns:
            List of predicted categories
            
        Raises:
            ValidationError: If texts list is invalid
            ClassificationError: If classification fails
        """
        if self.model is None or self.vectorizer is None:
            self.log_error("Models not loaded for batch classification")
            raise ModelLoadingError(
                "Models not loaded. Call load_models() first.",
                model_name="TextClassifier"
            )
        
        if not texts or len(texts) == 0:
            raise ValidationError(
                "Texts list is empty",
                field="texts"
            )
        
        try:
            self.log_debug(f"Batch classifying {len(texts)} texts")
            X = self.vectorizer.transform(texts)
            predictions = self.model.predict(X)
            
            self.log_info(f"Batch classification completed for {len(texts)} texts")
            return predictions.tolist()
        except Exception as e:
            self.log_error(f"Batch classification failed: {e}")
            raise ClassificationError(
                f"Batch classification failed: {e}",
                original_exception=e
            )
    
    def save_models(self, model_path: str, vectorizer_path: str) -> None:
        """
        Save trained classifier and vectorizer.
        
        Args:
            model_path: Path to save classifier
            vectorizer_path: Path to save vectorizer
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("No models to save. Train models first.")
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        print(f"Models saved to {model_path} and {vectorizer_path}")
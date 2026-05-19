"""
Named Entity Recognition Component
===================================

Handles NER using spaCy models.
"""

import spacy
from typing import List, Dict, Optional

from src.logger import LoggerMixin
from src.exceptions import (
    ModelLoadingError,
    ValidationError,
    NERError
)


class NamedEntityRecognizer(LoggerMixin):
    """
    Named Entity Recognition component using spaCy.
    
    Attributes:
        nlp: spaCy language model
        model_name: Name of the spaCy model
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize NamedEntityRecognizer.
        
        Args:
            model_name: Name of spaCy model to load
            
        Raises:
            ModelLoadingError: If model is not installed
        """
        self.setup_logger()
        
        self.model_name = model_name
        self.nlp = None
        self.load_model()
    
    def load_model(self) -> None:
        """
        Load spaCy model.
        
        Raises:
            ModelLoadingError: If model cannot be loaded
        """
        try:
            self.log_info(f"Loading spaCy model '{self.model_name}'")
            self.nlp = spacy.load(self.model_name)
            self.log_info(f"spaCy model '{self.model_name}' loaded successfully")
        except OSError as e:
            self.log_error(f"Failed to load spaCy model: {e}")
            raise ModelLoadingError(
                f"Failed to load spaCy model '{self.model_name}'. "
                f"Install it with: python -m spacy download {self.model_name}",
                model_name=self.model_name,
                original_exception=e
            )
    
    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text
            
        Returns:
            List of dictionaries with 'text' and 'label' keys
            
        Raises:
            ValidationError: If text is invalid
            NERError: If extraction fails
        """
        if not text or len(text.strip()) < 5:
            self.log_warning(f"Text too short for NER: {len(text)} chars")
            raise ValidationError(
                "Input text is too short (minimum 5 characters).",
                field="text",
                value=text[:50] if text else ""
            )
        
        if self.nlp is None:
            self.log_error("spaCy model not loaded")
            raise ModelLoadingError(
                "Model not loaded. Call load_model() first.",
                model_name=self.model_name
            )
        
        try:
            self.log_debug(f"Extracting entities from text of length {len(text)}")
            doc = self.nlp(text)
            
            entities = [
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                }
                for ent in doc.ents
            ]
            
            self.log_info(f"Extracted {len(entities)} entities")
            return entities
        except Exception as e:
            self.log_error(f"Entity extraction failed: {e}")
            raise NERError(
                f"Entity extraction failed: {e}",
                original_exception=e
            )
    
    def extract_entities_by_label(
        self,
        text: str,
        label: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """
        Extract entities filtered by label.
        
        Args:
            text: Input text
            label: Optional label filter (e.g., 'PERSON', 'ORG')
            
        Returns:
            Dictionary with labels as keys and lists of entities as values
        """
        entities = self.extract_entities(text)
        
        if label:
            filtered = [e["text"] for e in entities if e["label"] == label]
            return {label: filtered}
        
        result = {}
        for entity in entities:
            entity_label = entity["label"]
            if entity_label not in result:
                result[entity_label] = []
            result[entity_label].append(entity["text"])
        
        return result
    
    def get_available_labels(self) -> List[str]:
        """
        Get list of entity labels the model can recognize.
        
        Returns:
            List of entity labels
        """
        if self.nlp is None:
            raise ValueError("Model not loaded.")
        
        return self.nlp.pipe_labels.get("ner", [])
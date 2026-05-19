"""
Inference Pipeline
==================

Production-ready inference pipeline for serving predictions.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from src.pipeline.nlp_pipeline import NLPPipeline
from src.logger import LoggerMixin
from src.exceptions import (
    ValidationError,
    TextTooShortError,
    TextTooLongError,
    PipelineError,
    ComponentNotAvailableError,
    handle_exception
)


class TaskType(Enum):
    """Supported NLP task types."""
    CLASSIFICATION = "classification"
    NER = "named_entity_recognition"
    SUMMARIZATION = "summarization"
    FULL = "full_pipeline"


@dataclass
class PipelineConfig:
    """Configuration for inference pipeline."""
    classifier_model_path: Optional[str] = None
    classifier_vectorizer_path: Optional[str] = None
    ner_model_name: str = "en_core_web_sm"
    summarizer_model_name: str = "t5-small"
    max_text_length: int = 10000
    min_text_length: int = 20


class InferencePipeline(LoggerMixin):
    """
    Production inference pipeline with task-specific routing.
    
    Features:
    - Task-specific processing
    - Input validation
    - Error handling
    - Performance optimization
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize InferencePipeline.
        
        Args:
            config: PipelineConfig instance
        """
        self.setup_logger()
        
        self.config = config or PipelineConfig()
        
        self.log_info("Initializing InferencePipeline")
        
        try:
            self.pipeline = NLPPipeline(
                classifier_model_path=self.config.classifier_model_path,
                classifier_vectorizer_path=self.config.classifier_vectorizer_path,
                ner_model_name=self.config.ner_model_name,
                summarizer_model_name=self.config.summarizer_model_name
            )
            self.log_info("InferencePipeline initialization completed")
        except Exception as e:
            self.log_error(f"Pipeline initialization failed: {e}")
            raise PipelineError(
                f"Failed to initialize pipeline: {e}",
                original_exception=e
            )
    
    def _validate_input(self, text: str) -> bool:
        """
        Validate input text.
        
        Args:
            text: Input text to validate
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If text is invalid
        """
        if not text or not isinstance(text, str):
            self.log_warning("Invalid input: not a string or empty")
            raise ValidationError(
                "Input must be a non-empty string.",
                field="text"
            )
        
        text_length = len(text.strip())
        
        if text_length < self.config.min_text_length:
            self.log_warning(f"Text too short: {text_length} < {self.config.min_text_length}")
            raise TextTooShortError(
                self.config.min_text_length,
                text_length
            )
        
        if text_length > self.config.max_text_length:
            self.log_warning(f"Text too long: {text_length} > {self.config.max_text_length}")
            raise TextTooLongError(
                self.config.max_text_length,
                text_length
            )
        
        return True
    
    def classify(self, text: str) -> Dict[str, Any]:
        """
        Perform text classification.
        
        Args:
            text: Input text
            
        Returns:
            Classification result
        """
        self._validate_input(text)
        
        return self.pipeline.process_document(
            text,
            include_classification=True,
            include_ner=False,
            include_summary=False
        )
    
    def extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text
            
        Returns:
            NER result with entities
        """
        self._validate_input(text)
        
        return self.pipeline.process_document(
            text,
            include_classification=False,
            include_ner=True,
            include_summary=False
        )
    
    def summarize(self, text: str, max_length: int = 130) -> Dict[str, Any]:
        """
        Generate text summary.
        
        Args:
            text: Input text
            max_length: Maximum summary length
            
        Returns:
            Summarization result
        """
        self._validate_input(text)
        
        return self.pipeline.process_document(
            text,
            include_classification=False,
            include_ner=False,
            include_summary=True,
            summary_length=max_length
        )
    
    def full_analysis(self, text: str) -> Dict[str, Any]:
        """
        Perform complete document analysis.
        
        Args:
            text: Input text
            
        Returns:
            Results from all components
        """
        self._validate_input(text)
        
        return self.pipeline.process_document(
            text,
            include_classification=True,
            include_ner=True,
            include_summary=True
        )
    
    def process(
        self,
        text: str,
        task: TaskType = TaskType.FULL
    ) -> Dict[str, Any]:
        """
        Process text with task routing.
        
        Args:
            text: Input text
            task: Type of task to perform
            
        Returns:
            Task-specific results
            
        Raises:
            ValueError: If text is invalid or task unknown
        """
        self._validate_input(text)
        
        if task == TaskType.CLASSIFICATION:
            return self.classify(text)
        elif task == TaskType.NER:
            return self.extract_entities(text)
        elif task == TaskType.SUMMARIZATION:
            return self.summarize(text)
        elif task == TaskType.FULL:
            return self.full_analysis(text)
        else:
            raise ValueError(f"Unknown task type: {task}")
    
    def batch_process(
        self,
        texts: List[str],
        task: TaskType = TaskType.FULL
    ) -> List[Dict[str, Any]]:
        """
        Process multiple texts.
        
        Args:
            texts: List of input texts
            task: Type of task to perform
            
        Returns:
            List of results
        """
        results = []
        for text in texts:
            try:
                result = self.process(text, task)
                result["status"] = "success"
            except Exception as e:
                result = {
                    "status": "error",
                    "error": str(e)
                }
            results.append(result)
        
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get pipeline system status.
        
        Returns:
            System status information
        """
        status = self.pipeline.get_pipeline_status()
        status["config"] = {
            "min_text_length": self.config.min_text_length,
            "max_text_length": self.config.max_text_length,
            "ner_model": self.config.ner_model_name,
            "summarizer_model": self.config.summarizer_model_name
        }
        
        return status
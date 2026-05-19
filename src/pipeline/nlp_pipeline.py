"""
NLP Pipeline Orchestrator
=========================

Combines classification, NER, and summarization into a single workflow.
"""

from typing import Dict, List, Any, Optional

from src.components import (
    TextClassifier,
    NamedEntityRecognizer,
    TextSummarizer
)
from src.logger import LoggerMixin
from src.exceptions import (
    PipelineError,
    ComponentNotAvailableError,
    handle_exception
)


class NLPPipeline(LoggerMixin):
    """
    Complete NLP pipeline combining multiple components.
    
    Workflow:
    1. Text Classification - Categorize document
    2. Named Entity Recognition - Extract entities
    3. Text Summarization - Generate summary
    
    Attributes:
        classifier: TextClassifier instance
        ner: NamedEntityRecognizer instance
        summarizer: TextSummarizer instance
    """
    
    def __init__(
        self,
        classifier_model_path: Optional[str] = None,
        classifier_vectorizer_path: Optional[str] = None,
        ner_model_name: str = "en_core_web_sm",
        summarizer_model_name: str = "t5-small"
    ):
        """
        Initialize NLPPipeline with all components.
        
        Args:
            classifier_model_path: Path to saved classifier model
            classifier_vectorizer_path: Path to saved vectorizer
            ner_model_name: spaCy model name for NER
            summarizer_model_name: Hugging Face model for summarization
        """
        self.setup_logger()
        
        self.log_info("Initializing NLPPipeline")
        
        try:
            self.classifier = TextClassifier(
                model_path=classifier_model_path,
                vectorizer_path=classifier_vectorizer_path
            )
        except Exception as e:
            self.log_warning(f"Classifier initialization error: {e}")
            self.classifier = None
        
        try:
            self.ner = NamedEntityRecognizer(model_name=ner_model_name)
        except Exception as e:
            self.log_warning(f"NER initialization error: {e}")
            self.ner = None
        
        try:
            self.summarizer = TextSummarizer(model_name=summarizer_model_name)
        except Exception as e:
            self.log_warning(f"Summarizer initialization error: {e}")
            self.summarizer = None
        
        self.log_info("NLPPipeline initialization completed")
    
    def process_document(
        self,
        text: str,
        include_classification: bool = True,
        include_ner: bool = True,
        include_summary: bool = True,
        summary_length: int = 130
    ) -> Dict[str, Any]:
        """
        Process a document through the complete pipeline.
        
        Args:
            text: Input document text
            include_classification: Whether to perform classification
            include_ner: Whether to perform NER
            include_summary: Whether to perform summarization
            summary_length: Maximum length of generated summary
            
        Returns:
            Dictionary with results from enabled components
            
        Raises:
            ValueError: If text is invalid or too short
        """
        if not text or len(text.strip()) < 20:
            raise ValueError("Input text is too short (minimum 20 characters).")
        
        result = {
            "input_text": text[:200] + "..." if len(text) > 200 else text,
            "text_length": len(text)
        }
        
        # 1. Classification
        if include_classification:
            try:
                category = self.classifier.classify(text)
                result["category"] = category
            except Exception as e:
                result["classification_error"] = str(e)
        
        # 2. Named Entity Recognition
        if include_ner:
            try:
                entities = self.ner.extract_entities(text)
                result["entities"] = entities
                result["entity_summary"] = self.ner.extract_entities_by_label(text)
            except Exception as e:
                result["ner_error"] = str(e)
        
        # 3. Text Summarization
        if include_summary:
            try:
                summary = self.summarizer.summarize(
                    text,
                    max_summary_length=summary_length
                )
                result["summary"] = summary
            except Exception as e:
                result["summarization_error"] = str(e)
        
        return result
    
    def process_batch(
        self,
        texts: List[str],
        include_classification: bool = True,
        include_ner: bool = True,
        include_summary: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process multiple documents.
        
        Args:
            texts: List of document texts
            include_classification: Whether to perform classification
            include_ner: Whether to perform NER
            include_summary: Whether to perform summarization
            
        Returns:
            List of result dictionaries
        """
        results = []
        for text in texts:
            try:
                result = self.process_document(
                    text,
                    include_classification=include_classification,
                    include_ner=include_ner,
                    include_summary=include_summary
                )
                results.append(result)
            except Exception as e:
                results.append({"error": str(e)})
        
        return results
    
    def get_pipeline_status(self) -> Dict[str, bool]:
        """
        Check status of all pipeline components.
        
        Returns:
            Dictionary indicating if each component is ready
        """
        return {
            "classifier_ready": (
                self.classifier.model is not None 
                and self.classifier.vectorizer is not None
            ),
            "ner_ready": self.ner.nlp is not None,
            "summarizer_ready": (
                self.summarizer.model is not None 
                and self.summarizer.tokenizer is not None
            )
        }
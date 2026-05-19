"""
Text Summarization Component
============================

Handles abstractive summarization using transformer models.
"""

import torch
from typing import Optional

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

from src.logger import LoggerMixin
from src.exceptions import (
    ModelLoadingError,
    ValidationError,
    SummarizationError
)


class TextSummarizer(LoggerMixin):
    """
    Text summarization component using T5 or similar seq2seq models.
    
    Attributes:
        model_name: Name of the transformer model
        tokenizer: Hugging Face tokenizer
        model: Sequence-to-sequence model
        device: CPU or GPU device
    """
    
    def __init__(
        self,
        model_name: str = "t5-small",
        device: Optional[str] = None
    ):
        """
        Initialize TextSummarizer.
        
        Args:
            model_name: Name of the pre-trained model from Hugging Face
            device: Device to use ('cuda' or 'cpu'). Auto-detected if None.
        """
        self.setup_logger()
        
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.log_info(f"Using device: {self.device}")
        self.load_model()
    
    def load_model(self) -> None:
        """
        Load tokenizer and model from Hugging Face.
        
        Raises:
            ModelLoadingError: If model loading fails
        """
        try:
            self.log_info(f"Loading tokenizer for '{self.model_name}'")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            self.log_info(f"Loading model '{self.model_name}'")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            
            self.model.to(self.device)
            self.model.eval()
            
            self.log_info(f"Summarization model '{self.model_name}' loaded on {self.device}")
        except Exception as e:
            self.log_error(f"Failed to load model: {e}")
            raise ModelLoadingError(
                f"Failed to load summarization model '{self.model_name}': {e}",
                model_name=self.model_name,
                original_exception=e
            )
    
    def summarize(
        self,
        text: str,
        max_input_length: int = 512,
        max_summary_length: int = 130,
        min_summary_length: int = 40,
        num_beams: int = 4
    ) -> str:
        """
        Generate a summary for the input text.
        
        Args:
            text: Input text to summarize
            max_input_length: Maximum input token length
            max_summary_length: Maximum summary token length
            min_summary_length: Minimum summary token length
            num_beams: Number of beams for beam search
            
        Returns:
            Generated summary text
            
        Raises:
            ValidationError: If text is invalid or too short
            SummarizationError: If summarization fails
        """
        if not text or len(text.strip()) < 20:
            self.log_warning(f"Text too short for summarization: {len(text)} chars")
            raise ValidationError(
                "Input text is too short (minimum 20 characters).",
                field="text",
                value=text[:50] if text else ""
            )
        
        if self.model is None or self.tokenizer is None:
            self.log_error("Model not loaded")
            raise ModelLoadingError(
                "Model not loaded. Call load_model() first.",
                model_name=self.model_name
            )
        
        try:
            self.log_debug(f"Summarizing text of length {len(text)}")
            
            # Truncate text to avoid memory issues
            truncated_text = text[:1000]
            
            # Tokenize input
            inputs = self.tokenizer(
                f"summarize: {truncated_text}",
                return_tensors="pt",
                max_length=max_input_length,
                truncation=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate summary
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs["input_ids"],
                    max_length=max_summary_length,
                    min_length=min_summary_length,
                    num_beams=num_beams,
                    early_stopping=True
                )
            
            # Decode summary
            summary = self.tokenizer.decode(
                summary_ids[0],
                skip_special_tokens=True
            )
            
            self.log_info(f"Summarization completed. Summary length: {len(summary)}")
            return summary
        except Exception as e:
            self.log_error(f"Summarization failed: {e}")
            raise SummarizationError(
                f"Summarization failed: {e}",
                original_exception=e
            )
    
    def summarize_batch(
        self,
        texts: list,
        max_input_length: int = 512,
        max_summary_length: int = 130,
        min_summary_length: int = 40
    ) -> list:
        """
        Summarize multiple texts.
        
        Args:
            texts: List of texts to summarize
            max_input_length: Maximum input token length
            max_summary_length: Maximum summary token length
            min_summary_length: Minimum summary token length
            
        Returns:
            List of summaries
        """
        summaries = []
        for text in texts:
            summary = self.summarize(
                text,
                max_input_length=max_input_length,
                max_summary_length=max_summary_length,
                min_summary_length=min_summary_length
            )
            summaries.append(summary)
        
        return summaries
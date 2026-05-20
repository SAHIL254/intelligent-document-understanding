"""
Text Summarization Component
=============================
Handles transformer-based text summarization.
Extracted from: 05_text_summarization.ipynb
"""

import torch
from typing import Dict, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.logger import get_logger
from src.exceptions import ModelTrainingError
from src.utils import MODEL_CONFIGS


logger = get_logger(__name__)


class TextSummarizer:
    """
    Handles text summarization using pre-trained T5 transformer models.
    """
    
    def __init__(self, model_name: str = "t5-small", device: str = "cpu"):
        """
        Initialize text summarizer with T5 model.
        
        Args:
            model_name (str): Name of T5 model to load
            device (str): Device to run model on ("cpu" or "cuda")
            
        Raises:
            ModelTrainingError: If model loading fails
        """
        self.model_name = model_name
        self.device = torch.device(device)
        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        
        try:
            logger.info(f"Loading summarization model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            logger.info(f"Summarization model loaded successfully on {device}")
        except Exception as e:
            raise ModelTrainingError(f"Failed to load summarization model: {str(e)}")
    
    
    def summarize(
        self,
        text: str,
        max_input_length: int = 512,
        max_summary_length: int = 130,
        min_summary_length: int = 40,
        num_beams: int = 4
    ) -> str:
        """
        Generate summary for input text.
        
        Args:
            text (str): Input text to summarize
            max_input_length (int): Maximum input length for tokenizer
            max_summary_length (int): Maximum length of generated summary
            min_summary_length (int): Minimum length of generated summary
            num_beams (int): Number of beams for beam search
            
        Returns:
            str: Generated summary
            
        Raises:
            ModelTrainingError: If summarization fails
        """
        try:
            if not self.is_loaded:
                raise ModelTrainingError("Model not loaded. Initialize first.")
            
            if not text or len(text.strip()) == 0:
                raise ModelTrainingError("Input text is empty")
            
            # Truncate text to avoid too long input
            truncated_text = text[:1000]
            
            # Tokenize input
            logger.info(f"Summarizing text of length {len(text)}")
            
            inputs = self.tokenizer(
                "summarize: " + truncated_text,
                return_tensors="pt",
                max_length=max_input_length,
                truncation=True
            )
            
            # Move inputs to device
            inputs = {
                k: v.to(self.device)
                for k, v in inputs.items()
            }
            
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
            
            logger.info(f"Summary generated. Length: {len(summary)}")
            return summary
            
        except Exception as e:
            raise ModelTrainingError(f"Summarization failed: {str(e)}")
    
    
    def summarize_batch(
        self,
        texts: list,
        max_input_length: int = 512,
        max_summary_length: int = 130,
        min_summary_length: int = 40,
        num_beams: int = 4
    ) -> list:
        """
        Generate summaries for a batch of texts.
        
        Args:
            texts (list): List of input texts
            max_input_length (int): Maximum input length
            max_summary_length (int): Maximum summary length
            min_summary_length (int): Minimum summary length
            num_beams (int): Number of beams for beam search
            
        Returns:
            list: List of summaries
        """
        try:
            logger.info(f"Summarizing batch of {len(texts)} texts")
            
            summaries = [
                self.summarize(
                    text,
                    max_input_length,
                    max_summary_length,
                    min_summary_length,
                    num_beams
                )
                for text in texts
            ]
            
            logger.info(f"Batch summarization completed")
            return summaries
            
        except Exception as e:
            raise ModelTrainingError(f"Batch summarization failed: {str(e)}")
    
    
    def get_model_info(self) -> Dict:
        """
        Get information about loaded model.
        
        Returns:
            dict: Model information
        """
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "is_loaded": self.is_loaded,
            "model_type": type(self.model).__name__,
            "tokenizer_type": type(self.tokenizer).__name__
        }


class SummarizationPipeline:
    """
    Complete summarization pipeline for dataframes.
    """
    
    def __init__(self, model_name: str = "t5-small"):
        """
        Initialize summarization pipeline.
        
        Args:
            model_name (str): Name of T5 model
        """
        self.summarizer = TextSummarizer(model_name)
        logger.info("SummarizationPipeline initialized")
    
    
    def summarize_dataframe(
        self,
        df,
        text_column: str = 'text',
        output_column: str = 'summary',
        **summarize_kwargs
    ):
        """
        Add summaries to dataframe.
        
        Args:
            df: Input dataframe
            text_column (str): Name of text column
            output_column (str): Name of output summary column
            **summarize_kwargs: Additional arguments for summarize()
            
        Returns:
            DataFrame with added summary column
        """
        import pandas as pd
        
        try:
            if text_column not in df.columns:
                raise ModelTrainingError(f"Column '{text_column}' not found")
            
            logger.info(f"Summarizing {len(df)} documents")
            
            df_result = df.copy()
            df_result[output_column] = df_result[text_column].apply(
                lambda text: self.summarizer.summarize(text, **summarize_kwargs)
            )
            
            logger.info("Dataframe summarization completed")
            return df_result
            
        except Exception as e:
            raise ModelTrainingError(f"Dataframe summarization failed: {str(e)}")
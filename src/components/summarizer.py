"""
Text Summarization Component
=============================
Handles transformer-based text summarization with memory optimizations for production.
Extracted from: 05_text_summarization.ipynb
"""

import torch
import gc  # Used for aggressive garbage collection to prevent memory leaks
from typing import Dict, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.logger import get_logger
from src.exceptions import ModelTrainingError
from src.utils import MODEL_CONFIGS

logger = get_logger(__name__)


class TextSummarizer:
    """
    Handles text summarization using pre-trained T5 transformer models.
    Optimized with Lazy Weight Loading to safely run on Render's 512MB RAM Free Tier.
    """
    
    def __init__(self, model_name: str = "t5-small", device: str = "cpu"):
        """
        Initialize the text summarizer shell. 
        Weights are completely deferred until runtime inference to save memory.
        
        Args:
            model_name (str): Name of T5 model to load
            device (str): Device to run model on ("cpu" or "cuda")
        """
        self.model_name = model_name
        self.device = torch.device(device)
        self.tokenizer = None
        self.model = None
        self.is_loaded = False  # Track if weights are actually allocated in RAM
        
        logger.info(f"TextSummarizer instance created for {model_name} (Weights deferred until inference)")
    
    def _lazy_load_weights(self):
        """
        Internal helper to map and load model layers into RAM only when needed.
        If models are already loaded, it skips allocation instantly.
        """
        if not self.is_loaded:
            try:
                logger.info(f"📥 Free-Tier Allocation Alert: Lazy loading transformer layers for {self.model_name}...")
                
                # Load tokenizer structures
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                
                # Load model matrices directly mapped to CPU device
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
                self.model.eval()
                
                self.is_loaded = True
                logger.info(f"✓ Transformer weights for {self.model_name} securely anchored in memory.")
                
                # Clear temporary cache artifacts post-load
                gc.collect()
                
            except Exception as e:
                logger.error(f"Transformer weight initialization failed: {str(e)}")
                raise ModelTrainingError(f"Failed to load transformer models lazily: {str(e)}")

    def summarize(
        self,
        text: str,
        max_input_length: int = 512,
        max_summary_length: int = 130,
        min_summary_length: int = 40,
        num_beams: int = 2  # Lowered default from 4 to 2 to minimize memory tracking spikes
    ) -> str:
        """
        Generate an abstractive summary for input text.
        
        Args:
            text (str): Input text to summarize
            max_input_length (int): Maximum input length for tokenizer
            max_summary_length (int): Maximum length of generated summary
            min_summary_length (int): Minimum length of generated summary
            num_beams (int): Number of beams for beam search
            
        Returns:
            str: Generated summary
        """
        try:
            if not text or len(text.strip()) == 0:
                raise ModelTrainingError("Input text is empty")
            
            # TRIGGER WEIGHT LOADING (Only triggers on first use)
            self._lazy_load_weights()
            
            # Safe truncation boundary to limit context window overhead
            truncated_text = text[:1000]
            logger.info(f"Summarizing text block of length {len(text)}")
            
            # Tokenize input using native prefix notation for T5
            inputs = self.tokenizer(
                "summarize: " + truncated_text,
                return_tensors="pt",
                max_length=max_input_length,
                truncation=True
            )
            
            # Move inputs to target execution device
            inputs = {
                k: v.to(self.device)
                for k, v in inputs.items()
            }
            
            # Generate summary indices under local inference context
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs["input_ids"],
                    max_length=max_summary_length,
                    min_length=min_summary_length,
                    num_beams=num_beams,
                    early_stopping=True
                )
            
            # Decode generated output indices back to plaintext strings
            summary = self.tokenizer.decode(
                summary_ids[0],
                skip_special_tokens=True
            )
            
            logger.info(f"Summary generated successfully. String length: {len(summary)}")
            
            # Explicit stack cleanup to prevent lingering graph cache in RAM
            del inputs
            del summary_ids
            gc.collect()
            
            return summary
            
        except Exception as e:
            logger.error(f"Summarization processing loop failed: {str(e)}")
            raise ModelTrainingError(f"Summarization failed: {str(e)}")
    
    def summarize_batch(
        self,
        texts: list,
        max_input_length: int = 512,
        max_summary_length: int = 130,
        min_summary_length: int = 40,
        num_beams: int = 2
    ) -> list:
        """
        Generate summaries for a batch of texts sequentially to prevent RAM spikes.
        """
        try:
            logger.info(f"Summarizing sequential batch of {len(texts)} texts")
            
            summaries = []
            # Explicitly force evaluation context over the entire batch sequence loop
            with torch.no_grad():
                for text in texts:
                    summary = self.summarize(
                        text,
                        max_input_length,
                        max_summary_length,
                        min_summary_length,
                        num_beams
                    )
                    summaries.append(summary)
                    
                    # Force clean generation leftovers right inside the loop iteration
                    del summary
                    gc.collect()
            
            logger.info(f"Batch processing completed successfully")
            return summaries
            
        except Exception as e:
            raise ModelTrainingError(f"Batch summarization failed: {str(e)}")
    
    def get_model_info(self) -> Dict:
        """
        Get info about the current loaded state.
        """
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "is_loaded": self.is_loaded,
            "model_type": type(self.model).__name__ if self.is_loaded else None,
            "tokenizer_type": type(self.tokenizer).__name__ if self.is_loaded else None
        }


class SummarizationPipeline:
    """
    Complete summarization pipeline wrappers for dataframes.
    """
    
    def __init__(self, model_name: str = "t5-small"):
        """
        Initialize the wrapper pipeline shell.
        """
        self.summarizer = TextSummarizer(model_name)
        logger.info("SummarizationPipeline instantiated.")
    
    def summarize_dataframe(
        self,
        df,
        text_column: str = 'text',
        output_column: str = 'summary',
        **summarize_kwargs
    ):
        import pandas as pd
        
        try:
            if text_column not in df.columns:
                raise ModelTrainingError(f"Target dataframe tracking column '{text_column}' not found")
            
            logger.info(f"Running DataFrame batch mapping across {len(df)} rows.")
            
            df_result = df.copy()
            
            # Explicitly isolate the dataframe iteration inside a no_grad container
            with torch.no_grad():
                df_result[output_column] = df_result[text_column].apply(
                    lambda text: self.summarizer.summarize(text, **summarize_kwargs)
                )
            
            logger.info("Dataframe tracking map iteration complete.")
            return df_result
            
        except Exception as e:
            raise ModelTrainingError(f"Dataframe summarization pipeline failed: {str(e)}")
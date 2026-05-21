"""
Prediction Pipeline
===================
Real-time inference pipeline for model predictions.
Handles single, batch, and dataframe predictions with strict memory safety optimization.
"""

import torch
import gc
from typing import Dict, List, Union, Optional
import pandas as pd
from src.logger import get_logger
from src.exceptions import PredictionError


logger = get_logger(__name__)


class PredictionPipeline:
    """
    Handles real-time predictions using trained models.
    Optimized with evaluation context guards to prevent RAM bloating on Render CPU instances.
    """
    
    def __init__(
        self,
        classifier_model=None,
        vectorizer=None,
        nlp_model=None,
        summarizer=None
    ):
        """
        Initialize prediction pipeline with pre-loaded models.
        
        Args:
            classifier_model: Trained classification model
            vectorizer: Fitted TF-IDF vectorizer
            nlp_model: Loaded spaCy NER model
            summarizer: Lazy-loaded TextSummarizer component
        """
        self.classifier = classifier_model
        self.vectorizer = vectorizer
        self.nlp = nlp_model
        self.summarizer = summarizer
        
        logger.info("PredictionPipeline initialized with memory guards.")
    
    def predict_class(self, text: str, return_proba: bool = False) -> Union[str, Dict]:
        """
        Predict document class/category.
        """
        try:
            if not text or len(text.strip()) == 0:
                raise PredictionError("Input text is empty")
            
            if self.classifier is None or self.vectorizer is None:
                raise PredictionError("Classifier or vectorizer not loaded")
            
            # Transform text
            X = self.vectorizer.transform([text])
            
            # Predict
            prediction = self.classifier.predict(X)[0]
            
            if return_proba:
                try:
                    proba = self.classifier.predict_proba(X)[0]
                    classes = self.classifier.classes_
                    
                    return {
                        "class": prediction,
                        "probabilities": {
                            cls: float(prob)
                            for cls, prob in zip(classes, proba)
                        }
                    }
                except Exception:
                    logger.warning("Model does not support predict_proba")
                    return {"class": prediction}
            
            return prediction
            
        except Exception as e:
            raise PredictionError(f"Classification prediction failed: {str(e)}")
    
    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract named entities from text using spaCy.
        """
        try:
            if not text or len(text.strip()) == 0:
                raise PredictionError("Input text is empty")
            
            if self.nlp is None:
                raise PredictionError("NER model not loaded")
            
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
            
            logger.debug(f"Extracted {len(entities)} entities")
            return entities
            
        except Exception as e:
            raise PredictionError(f"Entity extraction failed: {str(e)}")
    
    def summarize_text(
        self,
        text: str,
        max_length: int = 130,
        min_length: int = 40
    ) -> str:
        """
        Generate text summary via the lazy-loaded Transformer.
        Wraps processing context explicitly to enforce memory deallocation.
        """
        try:
            if not text or len(text.strip()) == 0:
                raise PredictionError("Input text is empty")
            
            if self.summarizer is None:
                raise PredictionError("Summarizer not loaded")
            
            # Explicitly force torch evaluation context for single downstream execution calls
            with torch.no_grad():
                summary = self.summarizer.summarize(
                    text,
                    max_summary_length=max_length,
                    min_summary_length=min_length
                )
            
            return summary
            
        except Exception as e:
            raise PredictionError(f"Summarization failed: {str(e)}")
    
    def predict(self, text: str, tasks: List[str] = None) -> Dict:
        """
        Run all or selected prediction tasks under strict resource tracking context.
        """
        try:
            if tasks is None:
                tasks = ["classification", "ner", "summarization"]
            
            results = {
                "text_length": len(text),
                "tasks_executed": []
            }
            
            # Isolate entire inference graph tracking context
            with torch.no_grad():
                # 1. Classification
                if "classification" in tasks:
                    try:
                        results["classification"] = self.predict_class(
                            text,
                            return_proba=True
                        )
                        results["tasks_executed"].append("classification")
                    except Exception as e:
                        logger.warning(f"Classification task failed: {str(e)}")
                        results["classification"] = {"error": str(e)}
                
                # 2. NER Execution
                if "ner" in tasks:
                    try:
                        results["entities"] = self.extract_entities(text)
                        results["tasks_executed"].append("ner")
                    except Exception as e:
                        logger.warning(f"NER task failed: {str(e)}")
                        results["entities"] = {"error": str(e)}
                
                # 3. Summarization execution with immediate target cleanups
                if "summarization" in tasks:
                    try:
                        results["summary"] = self.summarize_text(text)
                        results["tasks_executed"].append("summarization")
                    except Exception as e:
                        logger.warning(f"Summarization task failed: {str(e)}")
                        results["summary"] = {"error": str(e)}
            
            logger.info(f"Prediction complete. Executed tasks: {results['tasks_executed']}")
            
            # Garbage collection step immediately after completing individual documents
            gc.collect()
            return results
            
        except Exception as e:
            raise PredictionError(f"Prediction pipeline failed: {str(e)}")
    
    def predict_batch(
        self,
        texts: List[str],
        tasks: List[str] = None,
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Make predictions on batch of texts sequentially while explicitly clearing residual cache.
        """
        try:
            results = []
            logger.info(f"Starting pipeline batch inference across {len(texts)} texts.")
            
            with torch.no_grad():
                for i, text in enumerate(texts):
                    if show_progress and (i + 1) % 5 == 0:
                        logger.info(f"Processing batch block: {i + 1}/{len(texts)}")
                    
                    # Run target pipeline tasks
                    result = self.predict(text, tasks)
                    results.append(result)
                    
                    # Aggressive cleanup within sequence iterations to avoid accumulation spikes
                    del result
                    gc.collect()
            
            logger.info(f"Batch prediction complete: {len(results)} texts processed securely")
            return results
            
        except Exception as e:
            raise PredictionError(f"Batch prediction failed: {str(e)}")
    
    def predict_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = 'text',
        tasks: List[str] = None
    ) -> pd.DataFrame:
        """
        Make predictions on all texts in a dataframe with loop-level garbage isolation.
        """
        try:
            if text_column not in df.columns:
                raise PredictionError(f"Column '{text_column}' not found in dataframe tracking layout")
            
            logger.info(f"Predicting dataframe matrix context on {len(df)} rows")
            
            df_result = df.copy()
            texts = df_result[text_column].tolist()
            
            # Delegate tracking over to memory protected loop sequence
            predictions = self.predict_batch(texts, tasks, show_progress=False)
            
            # Distribute predictions into individual mapped structural columns
            target_tasks = tasks or ["classification", "ner", "summarization"]
            
            if "classification" in target_tasks:
                df_result["predicted_class"] = [
                    p.get("classification", {}).get("class", None) if isinstance(p.get("classification"), dict) else None
                    for p in predictions
                ]
            
            if "ner" in target_tasks:
                df_result["entities"] = [
                    p.get("entities", []) if isinstance(p.get("entities"), list) else []
                    for p in predictions
                ]
            
            if "summarization" in target_tasks:
                df_result["summary"] = [
                    p.get("summary", None) if isinstance(p.get("summary"), str) else None
                    for p in predictions
                ]
            
            del predictions
            gc.collect()
            
            logger.info("Dataframe tracking prediction vectors mapped successfully.")
            return df_result
            
        except Exception as e:
            raise PredictionError(f"Dataframe prediction failed: {str(e)}")


class FastAPIEndpoint:
    """
    Utility class for FastAPI integration.
    Provides request/response schema validation.
    """
    
    @staticmethod
    def validate_prediction_request(data: dict) -> bool:
        """Validate incoming prediction request."""
        if "text" not in data or not data["text"]:
            raise PredictionError("'text' field is required and cannot be empty")
        return True
    
    @staticmethod
    def format_prediction_response(
        results: dict,
        include_raw: bool = False
    ) -> dict:
        """Format prediction results for API response."""
        response = {
            "success": True,
            "data": results
        }
        
        if not include_raw and isinstance(response["data"], dict):
            # Remove internal fields safely
            response["data"].pop("tasks_executed", None)
        
        return response
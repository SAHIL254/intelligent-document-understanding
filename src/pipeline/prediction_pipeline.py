"""
Prediction Pipeline
===================
Real-time inference pipeline for model predictions.
Handles single and batch predictions with model loading.
"""

from typing import Dict, List, Union, Optional
import pandas as pd
from src.logger import get_logger
from src.exceptions import PredictionError


logger = get_logger(__name__)


class PredictionPipeline:
    """
    Handles real-time predictions using trained models.
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
            summarizer: Loaded summarizer model
        """
        self.classifier = classifier_model
        self.vectorizer = vectorizer
        self.nlp = nlp_model
        self.summarizer = summarizer
        
        logger.info("PredictionPipeline initialized")
    
    
    def predict_class(self, text: str, return_proba: bool = False) -> Union[str, Dict]:
        """
        Predict document class/category.
        
        Args:
            text (str): Input text
            return_proba (bool): Whether to return probabilities
            
        Returns:
            str or dict: Predicted class or dict with class and probabilities
            
        Raises:
            PredictionError: If prediction fails
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
                except:
                    logger.warning("Model does not support predict_proba")
                    return {"class": prediction}
            
            return prediction
            
        except Exception as e:
            raise PredictionError(f"Classification prediction failed: {str(e)}")
    
    
    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract named entities from text.
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of extracted entities
            
        Raises:
            PredictionError: If extraction fails
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
        Generate text summary.
        
        Args:
            text (str): Input text
            max_length (int): Maximum summary length
            min_length (int): Minimum summary length
            
        Returns:
            str: Generated summary
            
        Raises:
            PredictionError: If summarization fails
        """
        try:
            if not text or len(text.strip()) == 0:
                raise PredictionError("Input text is empty")
            
            if self.summarizer is None:
                raise PredictionError("Summarizer not loaded")
            
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
        Run all or selected prediction tasks.
        
        Args:
            text (str): Input text
            tasks (list, optional): Specific tasks to run
                - "classification": Text classification
                - "ner": Named entity recognition
                - "summarization": Text summarization
                If None, all tasks are run
            
        Returns:
            dict: Results from selected tasks
        """
        try:
            if tasks is None:
                tasks = ["classification", "ner", "summarization"]
            
            results = {
                "text_length": len(text),
                "tasks_executed": []
            }
            
            # Classification
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
            
            # NER
            if "ner" in tasks:
                try:
                    results["entities"] = self.extract_entities(text)
                    results["tasks_executed"].append("ner")
                except Exception as e:
                    logger.warning(f"NER task failed: {str(e)}")
                    results["entities"] = {"error": str(e)}
            
            # Summarization
            if "summarization" in tasks:
                try:
                    results["summary"] = self.summarize_text(text)
                    results["tasks_executed"].append("summarization")
                except Exception as e:
                    logger.warning(f"Summarization task failed: {str(e)}")
                    results["summary"] = {"error": str(e)}
            
            logger.info(f"Prediction complete. Executed tasks: {results['tasks_executed']}")
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
        Make predictions on batch of texts.
        
        Args:
            texts (list): List of input texts
            tasks (list, optional): Tasks to run on each text
            show_progress (bool): Whether to show progress
            
        Returns:
            list: List of prediction results
        """
        try:
            results = []
            
            for i, text in enumerate(texts):
                if show_progress and (i + 1) % 10 == 0:
                    logger.info(f"Processing batch: {i + 1}/{len(texts)}")
                
                result = self.predict(text, tasks)
                results.append(result)
            
            logger.info(f"Batch prediction complete: {len(results)} texts processed")
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
        Make predictions on all texts in a dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_column (str): Name of text column
            tasks (list, optional): Tasks to run
            
        Returns:
            pd.DataFrame: Dataframe with predictions added
        """
        try:
            if text_column not in df.columns:
                raise PredictionError(f"Column '{text_column}' not found")
            
            logger.info(f"Predicting on {len(df)} texts")
            
            df_result = df.copy()
            texts = df_result[text_column].tolist()
            
            predictions = self.predict_batch(texts, tasks)
            
            # Add predictions as separate columns
            for task in (tasks or ["classification", "ner", "summarization"]):
                if task == "classification":
                    df_result["predicted_class"] = [
                        p.get("classification", {}).get("class", None)
                        for p in predictions
                    ]
                elif task == "ner":
                    df_result["entities"] = [
                        p.get("entities", [])
                        for p in predictions
                    ]
                elif task == "summarization":
                    df_result["summary"] = [
                        p.get("summary", None)
                        for p in predictions
                    ]
            
            logger.info("Dataframe prediction complete")
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
        
        if not include_raw:
            # Remove internal fields
            response["data"].pop("tasks_executed", None)
        
        return response

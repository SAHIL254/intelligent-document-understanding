"""
Model Trainer Component
=======================
Handles model training for classification and NER tasks.
Extracted from: 03_text_classification.ipynb & 04_named_entity_recognition.ipynb
"""

import joblib
import pandas as pd
from typing import Any, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.logger import get_logger
from src.exceptions import ModelTrainingError
from src.utils import MODEL_CONFIGS, get_model_path


logger = get_logger(__name__)


class ModelTrainer:
    """
    Handles training of text classification models.
    """
    
    def __init__(self, model_type: str = "logistic_regression", **kwargs):
        """
        Initialize model trainer.
        
        Args:
            model_type (str): Type of classifier ("logistic_regression", etc.)
            **kwargs: Optional model parameters
        """
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        
        if model_type == "logistic_regression":
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                **kwargs
            )
        else:
            raise ModelTrainingError(f"Unknown model type: {model_type}")
        
        logger.info(f"ModelTrainer initialized with {model_type}")
    
    
    def train(self, X_train, y_train) -> None:
        """
        Train the classification model.
        
        Args:
            X_train: Training feature matrix (TF-IDF vectors)
            y_train: Training labels
            
        Raises:
            ModelTrainingError: If training fails
        """
        try:
            logger.info(
                f"Starting model training with {X_train.shape[0]} samples "
                f"and {X_train.shape[1]} features"
            )
            
            self.model.fit(X_train, y_train)
            self.is_trained = True
            
            logger.info("Model training completed successfully")
            
        except Exception as e:
            raise ModelTrainingError(f"Model training failed: {str(e)}")
    
    
    def predict(self, X_test) -> list:
        """
        Make predictions on new data.
        
        Args:
            X_test: Test feature matrix
            
        Returns:
            list: Predicted labels
            
        Raises:
            ModelTrainingError: If model not trained or prediction fails
        """
        try:
            if not self.is_trained:
                raise ModelTrainingError("Model not trained. Call train() first.")
            
            predictions = self.model.predict(X_test)
            logger.info(f"Predictions made for {len(predictions)} samples")
            
            return predictions
            
        except Exception as e:
            raise ModelTrainingError(f"Prediction failed: {str(e)}")
    
    
    def predict_proba(self, X_test) -> list:
        """
        Get prediction probabilities.
        
        Args:
            X_test: Test feature matrix
            
        Returns:
            list: Prediction probabilities
            
        Raises:
            ModelTrainingError: If model not trained or prediction fails
        """
        try:
            if not self.is_trained:
                raise ModelTrainingError("Model not trained. Call train() first.")
            
            probabilities = self.model.predict_proba(X_test)
            logger.info(f"Probabilities computed for {len(probabilities)} samples")
            
            return probabilities
            
        except Exception as e:
            raise ModelTrainingError(f"Probability prediction failed: {str(e)}")
    
    
    def get_model_coefficients(self, feature_names: list, top_n: int = 10) -> dict:
        """
        Get top feature coefficients for each class.
        
        Args:
            feature_names (list): List of feature names
            top_n (int): Number of top features to return
            
        Returns:
            dict: Top features per class with their coefficients
        """
        try:
            if not self.is_trained:
                raise ModelTrainingError("Model not trained. Call train() first.")
            
            classes = self.model.classes_
            coefficients = self.model.coef_
            
            top_features = {}
            
            for idx, class_name in enumerate(classes):
                class_coef = coefficients[idx]
                top_indices = class_coef.argsort()[-top_n:][::-1]
                
                top_features[class_name] = {
                    feature_names[i]: float(class_coef[i])
                    for i in top_indices
                }
            
            logger.info(f"Extracted top {top_n} features per class")
            return top_features
            
        except Exception as e:
            raise ModelTrainingError(f"Failed to get model coefficients: {str(e)}")
    
    
    def save_model(self, model_path: Optional[str] = None) -> str:
        """
        Save trained model to disk.
        
        Args:
            model_path (str, optional): Path to save model
            
        Returns:
            str: Path where model was saved
            
        Raises:
            ModelTrainingError: If model not trained or saving fails
        """
        try:
            if not self.is_trained:
                raise ModelTrainingError("Model not trained. Call train() first.")
            
            if model_path is None:
                model_path = get_model_path("text_classifier")
            
            joblib.dump(self.model, model_path)
            logger.info(f"Model saved to {model_path}")
            
            return model_path
            
        except Exception as e:
            raise ModelTrainingError(f"Failed to save model: {str(e)}")
    
    
    @staticmethod
    def load_model(model_path: str) -> Any:
        """
        Load pre-trained model from disk.
        
        Args:
            model_path (str): Path to model file
            
        Returns:
            Loaded model object
            
        Raises:
            ModelTrainingError: If loading fails
        """
        try:
            model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
            return model
            
        except Exception as e:
            raise ModelTrainingError(f"Failed to load model: {str(e)}")


class NERTrainer:
    """
    Handles loading and managing spaCy NER models.
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize NER trainer.
        
        Args:
            model_name (str): Name of spaCy model to load
        """
        self.model_name = model_name
        self.nlp = None
        logger.info(f"NERTrainer initialized with model: {model_name}")
    
    
    def load_model(self) -> Any:
        """
        Load spaCy NER model.
        
        Returns:
            spaCy Language object with NER pipeline
            
        Raises:
            ModelTrainingError: If model loading fails
        """
        try:
            import spacy
            
            logger.info(f"Loading spaCy model: {self.model_name}")
            self.nlp = spacy.load(self.model_name)
            logger.info(f"spaCy model loaded successfully")
            
            return self.nlp
            
        except Exception as e:
            raise ModelTrainingError(f"Failed to load spaCy model: {str(e)}")
    
    
    def extract_entities(self, text: str) -> list:
        """
        Extract entities from text using loaded NER model.
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of entities with their labels
            
        Raises:
            ModelTrainingError: If model not loaded
        """
        try:
            if self.nlp is None:
                raise ModelTrainingError("Model not loaded. Call load_model() first.")
            
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
            
            return entities
            
        except Exception as e:
            raise ModelTrainingError(f"Entity extraction failed: {str(e)}")
    
    
    def extract_entities_from_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = 'text'
    ) -> pd.DataFrame:
        """
        Extract entities from all texts in a dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_column (str): Name of text column
            
        Returns:
            pd.DataFrame: Dataframe with extracted entities
        """
        try:
            if self.nlp is None:
                raise ModelTrainingError("Model not loaded. Call load_model() first.")
            
            logger.info(f"Extracting entities from {len(df)} documents")
            
            df_result = df.copy()
            df_result['entities'] = df_result[text_column].apply(self.extract_entities)
            
            logger.info("Entity extraction completed")
            return df_result
            
        except Exception as e:
            raise ModelTrainingError(f"Batch entity extraction failed: {str(e)}")

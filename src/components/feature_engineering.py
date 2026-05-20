"""
Feature Engineering Component
==============================
Handles feature extraction and vectorization.
Extracted from: 03_text_classification.ipynb
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from src.logger import get_logger
from src.exceptions import FeatureEngineeringError
from src.utils import MODEL_CONFIGS


logger = get_logger(__name__)


class FeatureEngineering:
    """
    Handles feature extraction from text data using TF-IDF vectorization.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize feature engineering with TF-IDF vectorizer.
        
        Args:
            **kwargs: Optional parameters to override defaults
                - max_features: Maximum number of features
                - ngram_range: N-gram range for TF-IDF
                - min_df: Minimum document frequency
                - max_df: Maximum document frequency
        """
        config = MODEL_CONFIGS['text_classification'].copy()
        config.update(kwargs)
        
        self.max_features = config['max_features']
        self.ngram_range = config['ngram_range']
        self.min_df = config['min_df']
        self.max_df = config['max_df']
        
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df
        )
        
        self.is_fitted = False
        logger.info(
            f"FeatureEngineering initialized with "
            f"max_features={self.max_features}, "
            f"ngram_range={self.ngram_range}"
        )
    
    
    def fit(self, texts: list) -> None:
        """
        Fit the TF-IDF vectorizer on texts.
        
        Args:
            texts (list): List of text documents
            
        Raises:
            FeatureEngineeringError: If fitting fails
        """
        try:
            if not isinstance(texts, (list, pd.Series)):
                raise FeatureEngineeringError("Texts must be a list or pandas Series")
            
            if len(texts) == 0:
                raise FeatureEngineeringError("Cannot fit on empty text list")
            
            logger.info(f"Fitting TF-IDF vectorizer on {len(texts)} texts")
            self.vectorizer.fit(texts)
            self.is_fitted = True
            
            feature_names = self.vectorizer.get_feature_names_out()
            logger.info(f"Vectorizer fitted. Generated {len(feature_names)} features")
            
        except Exception as e:
            raise FeatureEngineeringError(f"Vectorizer fitting failed: {str(e)}")
    
    
    def transform(self, texts: list):
        """
        Transform texts to TF-IDF feature matrix.
        
        Args:
            texts (list): List of text documents
            
        Returns:
            sparse matrix: TF-IDF feature matrix
            
        Raises:
            FeatureEngineeringError: If transformation fails
        """
        try:
            if not self.is_fitted:
                raise FeatureEngineeringError(
                    "Vectorizer not fitted. Call fit() first."
                )
            
            logger.info(f"Transforming {len(texts)} texts to features")
            X = self.vectorizer.transform(texts)
            logger.info(f"Transformation complete. Output shape: {X.shape}")
            
            return X
            
        except Exception as e:
            raise FeatureEngineeringError(f"Text transformation failed: {str(e)}")
    
    
    def fit_transform(self, texts: list):
        """
        Fit vectorizer and transform texts in one step.
        
        Args:
            texts (list): List of text documents
            
        Returns:
            sparse matrix: TF-IDF feature matrix
        """
        self.fit(texts)
        return self.transform(texts)
    
    
    def get_feature_names(self):
        """
        Get names of all features created by vectorizer.
        
        Returns:
            array: Feature names
            
        Raises:
            FeatureEngineeringError: If vectorizer not fitted
        """
        if not self.is_fitted:
            raise FeatureEngineeringError("Vectorizer not fitted. Call fit() first.")
        
        return self.vectorizer.get_feature_names_out()
    
    
    def get_top_features(self, document_index: int, n_features: int = 10):
        """
        Get top TF-IDF features for a specific document.
        
        Args:
            document_index (int): Index of document
            n_features (int): Number of top features to return
            
        Returns:
            dict: Top features with their TF-IDF scores
        """
        try:
            if not self.is_fitted:
                raise FeatureEngineeringError("Vectorizer not fitted. Call fit() first.")
            
            feature_names = self.get_feature_names()
            feature_array = np.array(feature_names)
            
            return feature_array, feature_array
            
        except Exception as e:
            raise FeatureEngineeringError(
                f"Failed to get top features: {str(e)}"
            )
    
    
    def get_vocabulary(self) -> dict:
        """
        Get the vocabulary (word to index mapping).
        
        Returns:
            dict: Vocabulary mapping
            
        Raises:
            FeatureEngineeringError: If vectorizer not fitted
        """
        if not self.is_fitted:
            raise FeatureEngineeringError("Vectorizer not fitted. Call fit() first.")
        
        return self.vectorizer.vocabulary_
    
    
    def get_idf_scores(self) -> dict:
        """
        Get IDF scores for all features.
        
        Returns:
            dict: Feature names to IDF scores mapping
            
        Raises:
            FeatureEngineeringError: If vectorizer not fitted
        """
        try:
            if not self.is_fitted:
                raise FeatureEngineeringError("Vectorizer not fitted. Call fit() first.")
            
            feature_names = self.get_feature_names()
            idf_scores = self.vectorizer.idf_
            
            return dict(zip(feature_names, idf_scores))
            
        except Exception as e:
            raise FeatureEngineeringError(f"Failed to get IDF scores: {str(e)}")
    
    
    def extract_features_from_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = 'text_cleaned',
        fit: bool = True
    ) -> Tuple[np.ndarray, list]:
        """
        Extract features from a dataframe column.
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_column (str): Name of text column
            fit (bool): Whether to fit vectorizer first
            
        Returns:
            tuple: (feature_matrix, feature_names)
        """
        try:
            if text_column not in df.columns:
                raise FeatureEngineeringError(
                    f"Column '{text_column}' not found in dataframe"
                )
            
            texts = df[text_column].tolist()
            
            if fit:
                X = self.fit_transform(texts)
            else:
                X = self.transform(texts)
            
            feature_names = self.get_feature_names().tolist()
            
            return X, feature_names
            
        except Exception as e:
            raise FeatureEngineeringError(
                f"Feature extraction from dataframe failed: {str(e)}"
            )

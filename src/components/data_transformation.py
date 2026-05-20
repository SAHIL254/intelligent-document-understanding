"""
Data Transformation Component
==============================
Handles text preprocessing and cleaning.
Extracted from: 02_text_preprocessing.ipynb
"""

import re
import pandas as pd
from typing import List, Tuple, Optional
from src.logger import get_logger
from src.exceptions import DataTransformationError
from src.utils import STOPWORDS_EN, SPECIAL_CHARS


logger = get_logger(__name__)


class DataTransformation:
    """
    Handles text preprocessing: lowercasing, removing punctuation,
    stopword removal, and other text cleaning operations.
    """
    
    def __init__(self, remove_stopwords: bool = True, lowercase: bool = True):
        """
        Initialize data transformation.
        
        Args:
            remove_stopwords (bool): Whether to remove stopwords
            lowercase (bool): Whether to convert text to lowercase
        """
        self.remove_stopwords = remove_stopwords
        self.lowercase = lowercase
        logger.info(
            f"DataTransformation initialized - "
            f"remove_stopwords={remove_stopwords}, lowercase={lowercase}"
        )
    
    
    def clean_text(self, text: str) -> str:
        """
        Clean a single text string.
        
        Steps:
        1. Lowercase conversion (optional)
        2. Remove special characters
        3. Remove extra whitespace
        4. Stopword removal (optional)
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
            
        Raises:
            DataTransformationError: If cleaning fails
        """
        if not isinstance(text, str):
            raise DataTransformationError(f"Expected string, got {type(text)}")
        
        try:
            # 1. Lowercase
            if self.lowercase:
                text = text.lower()
            
            # 2. Remove special characters and digits
            text = re.sub(SPECIAL_CHARS, ' ', text)
            
            # 3. Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            # 4. Remove stopwords
            if self.remove_stopwords:
                words = text.split()
                words = [w for w in words if w not in STOPWORDS_EN and len(w) > 2]
                text = ' '.join(words)
            
            return text
            
        except Exception as e:
            raise DataTransformationError(
                f"Text cleaning failed for text: {text[:50]}... Error: {str(e)}"
            )
    
    
    def transform_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Apply cleaning to entire dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_column (str): Name of text column to clean
            
        Returns:
            pd.DataFrame: Dataframe with cleaned text
            
        Raises:
            DataTransformationError: If transformation fails
        """
        try:
            logger.info(f"Starting transformation of {len(df)} samples")
            
            if text_column not in df.columns:
                raise DataTransformationError(f"Column '{text_column}' not found in dataframe")
            
            # Create a copy to avoid modifying original
            df_transformed = df.copy()
            
            # Remove null values in text column
            initial_size = len(df_transformed)
            df_transformed = df_transformed[df_transformed[text_column].notna()]
            
            if len(df_transformed) < initial_size:
                logger.warning(
                    f"Removed {initial_size - len(df_transformed)} rows with null text values"
                )
            
            # Apply cleaning
            df_transformed[f'{text_column}_cleaned'] = (
                df_transformed[text_column].apply(self.clean_text)
            )
            
            logger.info(f"Transformation completed successfully for {len(df_transformed)} samples")
            return df_transformed
            
        except Exception as e:
            raise DataTransformationError(f"Dataframe transformation failed: {str(e)}")
    
    
    def tokenize(self, text: str) -> List[str]:
        """
        Simple whitespace-based tokenization.
        
        Args:
            text (str): Text to tokenize
            
        Returns:
            list: List of tokens
        """
        return text.split()
    
    
    def remove_special_characters(self, text: str) -> str:
        """
        Remove special characters from text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with special characters removed
        """
        return re.sub(SPECIAL_CHARS, ' ', text)
    
    
    def remove_extra_whitespace(self, text: str) -> str:
        """
        Remove extra whitespace and newlines.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with normalized whitespace
        """
        return re.sub(r'\s+', ' ', text).strip()
    
    
    def remove_numbers(self, text: str) -> str:
        """
        Remove all numbers from text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text without numbers
        """
        return re.sub(r'\d+', '', text)
    
    
    def get_word_frequency(self, df: pd.DataFrame, text_column: str = 'text_cleaned') -> dict:
        """
        Calculate word frequency in the corpus.
        
        Args:
            df (pd.DataFrame): Dataframe with processed text
            text_column (str): Column name containing processed text
            
        Returns:
            dict: Word frequency dictionary (top 100 words)
        """
        try:
            from collections import Counter
            
            all_words = []
            for text in df[text_column]:
                if isinstance(text, str):
                    all_words.extend(text.split())
            
            word_freq = dict(Counter(all_words).most_common(100))
            logger.info(f"Computed word frequency for {len(all_words)} words")
            
            return word_freq
            
        except Exception as e:
            raise DataTransformationError(f"Word frequency calculation failed: {str(e)}")
    
    
    def split_train_test(
        self, 
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split dataframe into training and testing sets.
        
        Args:
            df (pd.DataFrame): Input dataframe
            test_size (float): Proportion of data for testing (0-1)
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (train_df, test_df)
        """
        from sklearn.model_selection import train_test_split
        
        try:
            train_df, test_df = train_test_split(
                df,
                test_size=test_size,
                random_state=random_state
            )
            
            logger.info(
                f"Data split: train={len(train_df)} ({100*(1-test_size):.0f}%), "
                f"test={len(test_df)} ({100*test_size:.0f}%)"
            )
            
            return train_df, test_df
            
        except Exception as e:
            raise DataTransformationError(f"Train-test split failed: {str(e)}")

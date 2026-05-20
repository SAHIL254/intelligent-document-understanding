"""
Data Ingestion Component
=========================
Handles loading and validation of datasets.
Extracted from: 01_data_understanding.ipynb
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from src.logger import get_logger
from src.exceptions import DataIngestionError
from src.utils import ARTIFACTS_DIR


logger = get_logger(__name__)


class DataIngestion:
    """
    Handles data loading, validation, and basic exploration.
    """
    
    def __init__(self, dataset_path: str):
        """
        Initialize data ingestion.
        
        Args:
            dataset_path (str): Path to the dataset file (CSV)
            
        Raises:
            DataIngestionError: If file doesn't exist or can't be read
        """
        self.dataset_path = Path(dataset_path)
        
        if not self.dataset_path.exists():
            raise DataIngestionError(f"Dataset file not found: {dataset_path}")
        
        logger.info(f"Data ingestion initialized with: {dataset_path}")
        self.data = None
    
    
    def load_data(self) -> pd.DataFrame:
        """
        Load dataset from CSV file.
        
        Returns:
            pd.DataFrame: Loaded dataset
            
        Raises:
            DataIngestionError: If loading fails
        """
        try:
            logger.info(f"Loading dataset from {self.dataset_path}")
            self.data = pd.read_csv(self.dataset_path)
            logger.info(f"Dataset loaded successfully. Shape: {self.data.shape}")
            return self.data
        except Exception as e:
            raise DataIngestionError(f"Failed to load dataset: {str(e)}")
    
    
    def validate_data(self) -> bool:
        """
        Validate dataset structure and content.
        
        Returns:
            bool: True if validation passes
            
        Raises:
            DataIngestionError: If validation fails
        """
        if self.data is None:
            raise DataIngestionError("No data loaded. Call load_data() first.")
        
        try:
            # Check if essential columns exist
            required_columns = ['text', 'label']
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            
            if missing_columns:
                raise DataIngestionError(f"Missing required columns: {missing_columns}")
            
            # Check for null values in critical columns
            if self.data['text'].isnull().any():
                null_count = self.data['text'].isnull().sum()
                logger.warning(f"Found {null_count} null values in 'text' column")
            
            if self.data['label'].isnull().any():
                null_count = self.data['label'].isnull().sum()
                logger.warning(f"Found {null_count} null values in 'label' column")
            
            logger.info("Data validation passed")
            return True
            
        except Exception as e:
            raise DataIngestionError(f"Data validation failed: {str(e)}")
    
    
    def get_statistics(self) -> dict:
        """
        Get basic statistics about the dataset.
        
        Returns:
            dict: Dataset statistics
        """
        if self.data is None:
            raise DataIngestionError("No data loaded. Call load_data() first.")
        
        try:
            stats = {
                "total_samples": len(self.data),
                "num_columns": len(self.data.columns),
                "columns": list(self.data.columns),
                "shape": self.data.shape,
                "missing_values": self.data.isnull().sum().to_dict(),
                "label_distribution": self.data['label'].value_counts().to_dict(),
                "text_stats": {
                    "min_length": self.data['text'].str.len().min(),
                    "max_length": self.data['text'].str.len().max(),
                    "avg_length": self.data['text'].str.len().mean(),
                    "median_length": self.data['text'].str.len().median(),
                }
            }
            
            logger.info("Statistics computed successfully")
            return stats
            
        except Exception as e:
            raise DataIngestionError(f"Failed to compute statistics: {str(e)}")
    
    
    def get_class_distribution(self) -> dict:
        """
        Get class distribution in the dataset.
        
        Returns:
            dict: Class distribution with counts and percentages
        """
        if self.data is None:
            raise DataIngestionError("No data loaded. Call load_data() first.")
        
        distribution = {}
        total_samples = len(self.data)
        
        for label, count in self.data['label'].value_counts().items():
            distribution[label] = {
                "count": int(count),
                "percentage": round((count / total_samples) * 100, 2)
            }
        
        logger.info(f"Class distribution: {distribution}")
        return distribution
    
    
    def save_ingestion_report(self, output_path: Optional[str] = None) -> str:
        """
        Save data ingestion report as JSON.
        
        Args:
            output_path (str, optional): Path to save report
            
        Returns:
            str: Path to saved report
        """
        import json
        
        if self.data is None:
            raise DataIngestionError("No data loaded. Call load_data() first.")
        
        if output_path is None:
            output_path = str(ARTIFACTS_DIR / "ingestion_report.json")
        
        try:
            report = {
                "dataset_path": str(self.dataset_path),
                "statistics": self.get_statistics(),
                "class_distribution": self.get_class_distribution(),
                "data_samples": self.data.head(5).to_dict('records')
            }
            
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=4, default=str)
            
            logger.info(f"Ingestion report saved to {output_path}")
            return output_path
            
        except Exception as e:
            raise DataIngestionError(f"Failed to save ingestion report: {str(e)}")
    
    
    def get_data(self) -> pd.DataFrame:
        """
        Get loaded dataset.
        
        Returns:
            pd.DataFrame: The loaded dataset
        """
        if self.data is None:
            raise DataIngestionError("No data loaded. Call load_data() first.")
        return self.data

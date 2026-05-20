"""
Custom Exceptions for NLP IDU Project
=======================================
Provides domain-specific exception classes for error handling.
"""


class NLPException(Exception):
    """Base exception class for all NLP IDU exceptions."""
    
    def __init__(self, message: str, error_code: str = None):
        """
        Initialize NLP exception.
        
        Args:
            message (str): Error message
            error_code (str): Optional error code for categorization
        """
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        super().__init__(self.message)
    
    def __str__(self):
        return f"[{self.error_code}] {self.message}"


class DataIngestionError(NLPException):
    """Raised when data loading or validation fails."""
    
    def __init__(self, message: str):
        super().__init__(message, "DATA_INGESTION_ERROR")


class DataTransformationError(NLPException):
    """Raised when text preprocessing fails."""
    
    def __init__(self, message: str):
        super().__init__(message, "DATA_TRANSFORMATION_ERROR")


class FeatureEngineeringError(NLPException):
    """Raised when feature extraction or vectorization fails."""
    
    def __init__(self, message: str):
        super().__init__(message, "FEATURE_ENGINEERING_ERROR")


class ModelTrainingError(NLPException):
    """Raised when model training fails."""
    
    def __init__(self, message: str):
        super().__init__(message, "MODEL_TRAINING_ERROR")


class ModelEvaluationError(NLPException):
    """Raised when model evaluation fails."""
    
    def __init__(self, message: str):
        super().__init__(message, "MODEL_EVALUATION_ERROR")


class PredictionError(NLPException):
    """Raised when inference/prediction fails."""
    
    def __init__(self, message: str):
        super().__init__(message, "PREDICTION_ERROR")


class ConfigurationError(NLPException):
    """Raised when configuration is invalid."""
    
    def __init__(self, message: str):
        super().__init__(message, "CONFIGURATION_ERROR")


class ModelLoadingError(NLPException):
    """Raised when model loading fails."""
    
    def __init__(self, message: str):
        super().__init__(message, "MODEL_LOADING_ERROR")
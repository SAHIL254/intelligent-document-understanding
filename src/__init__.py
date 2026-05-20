"""
NLP IDU Project
===============
Integrated Data Understanding (IDU) Pipeline for Natural Language Processing

Modules:
--------
- components: Core NLP processing components (data, features, models)
- pipeline: End-to-end training and inference pipelines
- exception: Custom exception classes
- logger: Logging configuration
- utils: Utility functions and constants
"""

__version__ = "1.0.0"
__author__ = "NLP IDU Team"

from src.components import (
    DataIngestion,
    DataTransformation,
    FeatureEngineering,
    ModelTrainer,
    NERTrainer,
    ModelEvaluation,
    NEREvaluation,
    TextSummarizer,
    SummarizationPipeline,
)

from src.pipeline import (
    TrainingPipeline,
    NLPInferencePipeline,
    PredictionPipeline,
    FastAPIEndpoint,
)

from src.exceptions import (
    NLPException,
    DataIngestionError,
    DataTransformationError,
    FeatureEngineeringError,
    ModelTrainingError,
    ModelEvaluationError,
    PredictionError,
    ConfigurationError,
    ModelLoadingError,
)

from src.logger import get_logger

__all__ = [
    # Version
    '__version__',
    '__author__',
    
    # Components
    'DataIngestion',
    'DataTransformation',
    'FeatureEngineering',
    'ModelTrainer',
    'NERTrainer',
    'ModelEvaluation',
    'NEREvaluation',
    'TextSummarizer',
    'SummarizationPipeline',
    
    # Pipelines
    'TrainingPipeline',
    'NLPInferencePipeline',
    'PredictionPipeline',
    'FastAPIEndpoint',
    
    # Exceptions
    'NLPException',
    'DataIngestionError',
    'DataTransformationError',
    'FeatureEngineeringError',
    'ModelTrainingError',
    'ModelEvaluationError',
    'PredictionError',
    'ConfigurationError',
    'ModelLoadingError',
    
    # Logger
    'get_logger',
]
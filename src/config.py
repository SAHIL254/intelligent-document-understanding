"""
Configuration Module
====================

Centralized configuration for the NLP pipeline.
"""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class ModelPaths:
    """Paths to trained models."""
    classifier: Optional[str] = None
    vectorizer: Optional[str] = None
    ner_model: str = "en_core_web_sm"
    summarizer: str = "t5-small"


@dataclass
class PipelineSettings:
    """Pipeline execution settings."""
    max_text_length: int = 10000
    min_text_length: int = 20
    batch_size: int = 32
    use_gpu: bool = False


@dataclass
class ClassificationConfig:
    """Classification settings."""
    model_path: Optional[str] = None
    vectorizer_path: Optional[str] = None


@dataclass
class NERConfig:
    """NER settings."""
    model_name: str = "en_core_web_sm"


@dataclass
class SummarizationConfig:
    """Summarization settings."""
    model_name: str = "t5-small"
    max_input_length: int = 512
    max_summary_length: int = 130
    min_summary_length: int = 40
    num_beams: int = 4


@dataclass
class APIConfig:
    """API server settings."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    reload: bool = False
    log_level: str = "info"


class Config:
    """Main configuration class."""
    
    def __init__(self):
        """Initialize configuration."""
        self.models = ModelPaths()
        self.pipeline = PipelineSettings()
        self.classification = ClassificationConfig()
        self.ner = NERConfig()
        self.summarization = SummarizationConfig()
        self.api = APIConfig()
    
    @classmethod
    def from_env(cls):
        """Load configuration from environment variables."""
        import os
        
        config = cls()
        
        # Model paths
        config.models.classifier = os.getenv("CLASSIFIER_MODEL_PATH")
        config.models.vectorizer = os.getenv("VECTORIZER_MODEL_PATH")
        config.models.ner_model = os.getenv("NER_MODEL_NAME", "en_core_web_sm")
        config.models.summarizer = os.getenv("SUMMARIZER_MODEL_NAME", "t5-small")
        
        # Pipeline settings
        config.pipeline.max_text_length = int(os.getenv("MAX_TEXT_LENGTH", 10000))
        config.pipeline.min_text_length = int(os.getenv("MIN_TEXT_LENGTH", 20))
        config.pipeline.batch_size = int(os.getenv("BATCH_SIZE", 32))
        config.pipeline.use_gpu = os.getenv("USE_GPU", "false").lower() == "true"
        
        # API settings
        config.api.host = os.getenv("API_HOST", "0.0.0.0")
        config.api.port = int(os.getenv("API_PORT", 8000))
        config.api.debug = os.getenv("API_DEBUG", "false").lower() == "true"
        config.api.log_level = os.getenv("LOG_LEVEL", "info")
        
        return config
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "models": self.models.__dict__,
            "pipeline": self.pipeline.__dict__,
            "classification": self.classification.__dict__,
            "ner": self.ner.__dict__,
            "summarization": self.summarization.__dict__,
            "api": self.api.__dict__
        }


# Default configuration instance
default_config = Config()
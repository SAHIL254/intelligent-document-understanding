"""
Custom Exceptions Module
========================

Defines custom exceptions for the NLP pipeline.
Provides consistent error handling across the application.
"""

from typing import Optional, Any, Dict


class NLPPipelineException(Exception):
    """
    Base exception for NLP pipeline.
    
    All custom exceptions inherit from this.
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ):
        """
        Initialize exception.
        
        Args:
            message: Error message
            error_code: Error code for API responses
            details: Additional error details
            original_exception: Original exception that caused this
        """
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.original_exception = original_exception
        
        super().__init__(self.message)
    
    def __str__(self) -> str:
        """Return string representation."""
        return f"[{self.error_code}] {self.message}"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary for API response.
        
        Returns:
            Dictionary representation
        """
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details
        }


class ConfigurationError(NLPPipelineException):
    """
    Raised when there's a configuration error.
    
    Examples:
    - Invalid configuration file
    - Missing required config values
    - Invalid config values
    """
    
    def __init__(
        self,
        message: str,
        config_item: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize ConfigurationError.
        
        Args:
            message: Error message
            config_item: Name of the configuration item
            **kwargs: Additional arguments
        """
        if config_item:
            message = f"Configuration error for '{config_item}': {message}"
        
        super().__init__(message, error_code="CONFIG_ERROR", **kwargs)


class ModelLoadingError(NLPPipelineException):
    """
    Raised when model loading fails.
    
    Examples:
    - Model file not found
    - Model loading failed
    - Invalid model format
    - Incompatible model version
    """
    
    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize ModelLoadingError.
        
        Args:
            message: Error message
            model_name: Name of the model
            model_path: Path to model file
            **kwargs: Additional arguments
        """
        details = {}
        if model_name:
            details["model_name"] = model_name
        if model_path:
            details["model_path"] = model_path
        
        if model_name:
            message = f"Failed to load model '{model_name}': {message}"
        
        super().__init__(
            message,
            error_code="MODEL_LOADING_ERROR",
            details=details,
            **kwargs
        )


class ValidationError(NLPPipelineException):
    """
    Raised when input validation fails.
    
    Examples:
    - Text too short
    - Text too long
    - Invalid text format
    - Missing required fields
    """
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        **kwargs
    ):
        """
        Initialize ValidationError.
        
        Args:
            message: Error message
            field: Field name
            value: Invalid value
            **kwargs: Additional arguments
        """
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)
        
        if field:
            message = f"Validation failed for '{field}': {message}"
        
        super().__init__(
            message,
            error_code="VALIDATION_ERROR",
            details=details,
            **kwargs
        )


class InputError(ValidationError):
    """Raised for invalid input."""
    pass


class TextTooShortError(InputError):
    """Raised when input text is too short."""
    
    def __init__(self, min_length: int, actual_length: int):
        """
        Initialize TextTooShortError.
        
        Args:
            min_length: Minimum required length
            actual_length: Actual text length
        """
        message = (
            f"Text is too short. "
            f"Minimum {min_length} characters required, got {actual_length}"
        )
        super().__init__(
            message,
            field="text",
            details={"min_length": min_length, "actual_length": actual_length}
        )


class TextTooLongError(InputError):
    """Raised when input text is too long."""
    
    def __init__(self, max_length: int, actual_length: int):
        """
        Initialize TextTooLongError.
        
        Args:
            max_length: Maximum allowed length
            actual_length: Actual text length
        """
        message = (
            f"Text is too long. "
            f"Maximum {max_length} characters allowed, got {actual_length}"
        )
        super().__init__(
            message,
            field="text",
            details={"max_length": max_length, "actual_length": actual_length}
        )


class PipelineError(NLPPipelineException):
    """
    Raised when pipeline processing fails.
    
    Examples:
    - Component not available
    - Processing failed
    - Unexpected error during processing
    """
    
    def __init__(
        self,
        message: str,
        component: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize PipelineError.
        
        Args:
            message: Error message
            component: Component that failed
            **kwargs: Additional arguments
        """
        if component:
            message = f"Pipeline error in '{component}': {message}"
        
        super().__init__(
            message,
            error_code="PIPELINE_ERROR",
            **kwargs
        )


class ComponentNotAvailableError(PipelineError):
    """Raised when a component is not available."""
    
    def __init__(self, component_name: str):
        """
        Initialize ComponentNotAvailableError.
        
        Args:
            component_name: Name of unavailable component
        """
        message = f"Component '{component_name}' is not available or not initialized"
        super().__init__(
            message,
            component=component_name
        )


class ProcessingError(PipelineError):
    """Raised when processing fails."""
    
    def __init__(
        self,
        message: str,
        component: Optional[str] = None,
        step: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize ProcessingError.
        
        Args:
            message: Error message
            component: Component name
            step: Processing step
            **kwargs: Additional arguments
        """
        details = {}
        if step:
            details["step"] = step
        
        super().__init__(message, component=component, **kwargs)


class ClassificationError(ProcessingError):
    """Raised when classification fails."""
    
    def __init__(self, message: str, **kwargs):
        """Initialize ClassificationError."""
        super().__init__(
            message,
            component="TextClassifier",
            error_code="CLASSIFICATION_ERROR",
            **kwargs
        )


class NERError(ProcessingError):
    """Raised when NER fails."""
    
    def __init__(self, message: str, **kwargs):
        """Initialize NERError."""
        super().__init__(
            message,
            component="NamedEntityRecognizer",
            error_code="NER_ERROR",
            **kwargs
        )


class SummarizationError(ProcessingError):
    """Raised when summarization fails."""
    
    def __init__(self, message: str, **kwargs):
        """Initialize SummarizationError."""
        super().__init__(
            message,
            component="TextSummarizer",
            error_code="SUMMARIZATION_ERROR",
            **kwargs
        )


class APIError(NLPPipelineException):
    """
    Raised for API-related errors.
    
    Examples:
    - Request validation failed
    - Response generation failed
    - API endpoint error
    """
    
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        **kwargs
    ):
        """
        Initialize APIError.
        
        Args:
            message: Error message
            status_code: HTTP status code
            **kwargs: Additional arguments
        """
        super().__init__(message, error_code="API_ERROR", **kwargs)
        self.status_code = status_code
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to API response dictionary.
        
        Returns:
            API response dictionary
        """
        response = super().to_dict()
        response["status_code"] = self.status_code
        return response


class RequestValidationError(APIError):
    """Raised when request validation fails."""
    
    def __init__(self, message: str, **kwargs):
        """Initialize RequestValidationError."""
        super().__init__(
            message,
            status_code=400,
            **kwargs
        )


class NotFoundError(APIError):
    """Raised when resource is not found."""
    
    def __init__(self, resource: str, **kwargs):
        """
        Initialize NotFoundError.
        
        Args:
            resource: Resource name
            **kwargs: Additional arguments
        """
        message = f"Resource '{resource}' not found"
        super().__init__(
            message,
            status_code=404,
            **kwargs
        )


class InternalServerError(APIError):
    """Raised for internal server errors."""
    
    def __init__(self, message: str = "Internal server error", **kwargs):
        """Initialize InternalServerError."""
        super().__init__(
            message,
            status_code=500,
            **kwargs
        )


class TimeoutError(NLPPipelineException):
    """Raised when operation times out."""
    
    def __init__(self, operation: str, timeout_seconds: float, **kwargs):
        """
        Initialize TimeoutError.
        
        Args:
            operation: Operation that timed out
            timeout_seconds: Timeout duration
            **kwargs: Additional arguments
        """
        message = (
            f"Operation '{operation}' timed out "
            f"after {timeout_seconds} seconds"
        )
        super().__init__(
            message,
            error_code="TIMEOUT_ERROR",
            details={"timeout_seconds": timeout_seconds},
            **kwargs
        )


class MemoryError(NLPPipelineException):
    """Raised when memory allocation fails."""
    
    def __init__(self, operation: str, required_mb: Optional[float] = None, **kwargs):
        """
        Initialize MemoryError.
        
        Args:
            operation: Operation that failed
            required_mb: Required memory in MB
            **kwargs: Additional arguments
        """
        message = f"Insufficient memory for operation '{operation}'"
        if required_mb:
            message += f" (required: {required_mb}MB)"
        
        details = {}
        if required_mb:
            details["required_mb"] = required_mb
        
        super().__init__(
            message,
            error_code="MEMORY_ERROR",
            details=details,
            **kwargs
        )


def handle_exception(
    exc: Exception,
    logger = None,
    operation: str = "operation"
) -> NLPPipelineException:
    """
    Convert any exception to NLPPipelineException.
    
    Args:
        exc: Exception to handle
        logger: Logger instance
        operation: Operation that failed
        
    Returns:
        NLPPipelineException instance
    """
    if isinstance(exc, NLPPipelineException):
        return exc
    
    # Log the error
    if logger:
        logger.error(f"Error during {operation}: {exc}", exc_info=True)
    
    # Create appropriate exception
    error_message = str(exc)
    pipeline_exc = ProcessingError(
        error_message,
        step=operation,
        original_exception=exc
    )
    
    return pipeline_exc

"""
Logging Module
==============

Centralized logging configuration for the NLP pipeline.
Provides consistent logging across all components.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        """Format log record with colors."""
        log_level = record.levelname
        color = self.COLORS.get(log_level, self.COLORS['INFO'])
        
        # Format the message
        log_message = super().format(record)
        
        # Add color
        colored_message = f"{color}{log_message}{self.COLORS['RESET']}"
        
        return colored_message


class Logger:
    """
    Centralized logger configuration.
    
    Features:
    - Console logging with colors
    - File logging
    - Different log levels
    - Rotating file handlers
    """
    
    _loggers = {}
    
    def __init__(self, name: str, log_dir: str = "logs"):
        """
        Initialize logger.
        
        Args:
            name: Logger name (usually __name__)
            log_dir: Directory for log files
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """
        Setup logger with console and file handlers.
        
        Returns:
            Configured logger instance
        """
        if self.name in self._loggers:
            return self._loggers[self.name]
        
        # Create logger
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = ColoredFormatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler (daily rotation)
        log_file = self.log_dir / f"{self.name}.log"
        try:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10485760,  # 10MB
                backupCount=5
            )
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Could not setup file logging: {e}")
        
        # Cache logger
        self._loggers[self.name] = logger
        
        return logger
    
    def get_logger(self) -> logging.Logger:
        """
        Get configured logger instance.
        
        Returns:
            Logger instance
        """
        return self.logger


# Singleton instance holders
_default_logger = None
_component_loggers = {}


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get or create a logger instance.
    
    Args:
        name: Logger name (defaults to module name)
        
    Returns:
        Logger instance
    """
    global _default_logger
    
    if name is None:
        if _default_logger is None:
            _default_logger = Logger("nlp_pipeline").get_logger()
        return _default_logger
    
    if name not in _component_loggers:
        _component_loggers[name] = Logger(name).get_logger()
    
    return _component_loggers[name]


def setup_logging(
    name: str = "nlp_pipeline",
    level: int = logging.INFO,
    log_dir: str = "logs"
) -> logging.Logger:
    """
    Setup logging system.
    
    Args:
        name: Logger name
        level: Logging level
        log_dir: Directory for log files
        
    Returns:
        Logger instance
    """
    logger = Logger(name, log_dir=log_dir).get_logger()
    logger.setLevel(level)
    return logger


def log_function_call(logger: logging.Logger):
    """
    Decorator to log function calls.
    
    Args:
        logger: Logger instance
        
    Example:
        @log_function_call(logger)
        def my_function():
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.debug(f"Calling {func.__name__}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func.__name__} completed successfully")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} failed: {e}", exc_info=True)
                raise
        return wrapper
    return decorator


class LoggerMixin:
    """
    Mixin class to add logging to any class.
    
    Usage:
        class MyClass(LoggerMixin):
            def __init__(self):
                self.setup_logger()
            
            def my_method(self):
                self.logger.info("Doing something")
    """
    
    def setup_logger(self, name: Optional[str] = None) -> None:
        """
        Setup logger for the class.
        
        Args:
            name: Logger name (defaults to class name)
        """
        logger_name = name or self.__class__.__name__
        self.logger = get_logger(logger_name)
    
    def log_debug(self, message: str, *args, **kwargs) -> None:
        """Log debug message."""
        if hasattr(self, 'logger'):
            self.logger.debug(message, *args, **kwargs)
    
    def log_info(self, message: str, *args, **kwargs) -> None:
        """Log info message."""
        if hasattr(self, 'logger'):
            self.logger.info(message, *args, **kwargs)
    
    def log_warning(self, message: str, *args, **kwargs) -> None:
        """Log warning message."""
        if hasattr(self, 'logger'):
            self.logger.warning(message, *args, **kwargs)
    
    def log_error(self, message: str, *args, **kwargs) -> None:
        """Log error message."""
        if hasattr(self, 'logger'):
            self.logger.error(message, *args, **kwargs)
    
    def log_critical(self, message: str, *args, **kwargs) -> None:
        """Log critical message."""
        if hasattr(self, 'logger'):
            self.logger.critical(message, *args, **kwargs)


# Default logger instance
default_logger = get_logger("nlp_pipeline")

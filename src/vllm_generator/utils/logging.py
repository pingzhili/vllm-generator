"""Logging utilities using loguru."""

import sys
from typing import Optional
from loguru import logger

from ..config.schemas import LoggingConfig


class LoggerSetup:
    """Setup and configure loguru logger."""
    
    _initialized = False
    
    @classmethod
    def setup(cls, config: LoggingConfig) -> None:
        """Configure logger based on configuration."""
        if cls._initialized:
            return
        
        # Remove default handler
        logger.remove()
        
        # Add console handler
        logger.add(
            sys.stderr,
            format=config.format,
            level=config.level,
            colorize=True,
            enqueue=True,
        )
        
        # Add file handler if specified
        if config.file:
            logger.add(
                config.file,
                format=config.format,
                level=config.level,
                rotation=config.rotation,
                retention=config.retention,
                enqueue=True,
                encoding="utf-8",
            )
        
        cls._initialized = True
        logger.info("Logger initialized with level: {}", config.level)
    
    @classmethod
    def get_logger(cls, name: Optional[str] = None) -> logger:
        """Get a logger instance."""
        if name:
            return logger.bind(name=name)
        return logger


def setup_logger(config: LoggingConfig) -> None:
    """Setup logger with configuration."""
    LoggerSetup.setup(config)


def get_logger(name: Optional[str] = None) -> logger:
    """Get a logger instance."""
    return LoggerSetup.get_logger(name)
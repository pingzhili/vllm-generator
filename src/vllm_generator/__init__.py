"""vLLM Generator - A scalable data generation pipeline using vLLM models."""

__version__ = "0.1.0"
__author__ = "Your Name"

from .pipeline import Pipeline
from .config import Config

__all__ = ["Pipeline", "Config", "__version__"]
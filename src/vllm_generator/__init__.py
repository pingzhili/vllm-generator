"""vLLM Generator - A scalable data generation pipeline using vLLM models."""

__version__ = "0.1.0"
__author__ = "Your Name"

from .pipeline import GenerationPipeline
from .config import Config

__all__ = ["GenerationPipeline", "Config", "__version__"]
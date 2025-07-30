"""Base pipeline classes and interfaces."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pathlib import Path

from ..config.schemas import Config
from ..utils import get_logger


class PipelineStep(ABC):
    """Abstract base class for pipeline steps."""
    
    def __init__(self, name: str):
        """Initialize pipeline step."""
        self.name = name
        self.logger = get_logger(f"Pipeline.{name}")
    
    @abstractmethod
    async def execute(self, data: Any, context: Dict[str, Any]) -> Any:
        """Execute the pipeline step."""
        pass
    
    async def validate(self, config: Config) -> bool:
        """Validate step configuration."""
        return True


class Pipeline:
    """Base pipeline class for orchestrating steps."""
    
    def __init__(self, config: Config):
        """Initialize pipeline."""
        self.config = config
        self.logger = get_logger("Pipeline")
        self.steps: list[PipelineStep] = []
        self.context: Dict[str, Any] = {}
    
    def add_step(self, step: PipelineStep) -> None:
        """Add a step to the pipeline."""
        self.steps.append(step)
        self.logger.debug(f"Added step: {step.name}")
    
    async def validate(self) -> bool:
        """Validate all pipeline steps."""
        for step in self.steps:
            if not await step.validate(self.config):
                self.logger.error(f"Validation failed for step: {step.name}")
                return False
        return True
    
    async def execute(self, initial_data: Any = None) -> Any:
        """Execute the pipeline."""
        data = initial_data
        
        for step in self.steps:
            self.logger.info(f"Executing step: {step.name}")
            try:
                data = await step.execute(data, self.context)
            except Exception as e:
                self.logger.error(f"Step {step.name} failed: {e}")
                raise
        
        return data
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """Get value from pipeline context."""
        return self.context.get(key, default)
    
    def set_context(self, key: str, value: Any) -> None:
        """Set value in pipeline context."""
        self.context[key] = value
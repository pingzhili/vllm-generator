from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import json
import yaml
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for vLLM model"""
    
    # Model identification
    model: str
    model_revision: Optional[str] = None
    tokenizer: Optional[str] = None
    
    # Device and memory
    dtype: str = "auto"
    device: str = "cuda"
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    max_model_len: Optional[int] = None
    
    # Generation parameters
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 512
    min_tokens: int = 1
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    stop_sequences: Optional[List[str]] = None
    seed: Optional[int] = None
    best_of: Optional[int] = None
    
    # Performance settings
    swap_space: int = 4
    cpu_offload_gb: Optional[int] = None
    quantization: Optional[str] = None
    enforce_eager: bool = False
    enable_prefix_caching: bool = False
    max_num_seqs: Optional[int] = None
    
    # Additional settings
    trust_remote_code: bool = False
    disable_log_stats: bool = False
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelConfig":
        """Create config from dictionary"""
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ModelConfig":
        """Load config from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict.get("model_config", config_dict))
    
    @classmethod
    def from_json(cls, json_path: str) -> "ModelConfig":
        """Load config from JSON file"""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict.get("model_config", config_dict))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {k: v for k, v in self.__dict__.items() if v is not None}
    
    def to_vllm_args(self) -> Dict[str, Any]:
        """Convert to vLLM-compatible arguments"""
        vllm_args = {
            "model": self.model,
            "dtype": self.dtype,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "tensor_parallel_size": self.tensor_parallel_size,
            "trust_remote_code": self.trust_remote_code,
            "swap_space": self.swap_space,
            "disable_log_stats": self.disable_log_stats,
        }
        
        if self.model_revision:
            vllm_args["revision"] = self.model_revision
        if self.tokenizer:
            vllm_args["tokenizer"] = self.tokenizer
        if self.max_model_len:
            vllm_args["max_model_len"] = self.max_model_len
        if self.quantization:
            vllm_args["quantization"] = self.quantization
        if self.cpu_offload_gb:
            vllm_args["cpu_offload_gb"] = self.cpu_offload_gb
        if self.enforce_eager:
            vllm_args["enforce_eager"] = self.enforce_eager
        if self.enable_prefix_caching:
            vllm_args["enable_prefix_caching"] = self.enable_prefix_caching
        if self.max_num_seqs:
            vllm_args["max_num_seqs"] = self.max_num_seqs
        
        return vllm_args
    
    def to_sampling_params(self) -> Dict[str, Any]:
        """Convert to vLLM SamplingParams arguments"""
        sampling_params = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "min_tokens": self.min_tokens,
            "repetition_penalty": self.repetition_penalty,
            "length_penalty": self.length_penalty,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
        }
        
        if self.stop_sequences:
            sampling_params["stop"] = self.stop_sequences
        if self.seed is not None:
            sampling_params["seed"] = self.seed
        if self.best_of:
            sampling_params["best_of"] = self.best_of
        
        return sampling_params
    
    def update(self, **kwargs):
        """Update config with new values"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")
    
    def validate(self):
        """Validate configuration"""
        if not self.model:
            raise ValueError("Model name is required")
        
        if self.gpu_memory_utilization <= 0 or self.gpu_memory_utilization > 1:
            raise ValueError("gpu_memory_utilization must be between 0 and 1")
        
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        
        if self.top_p <= 0 or self.top_p > 1:
            raise ValueError("top_p must be between 0 and 1")
        
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        
        if self.tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be at least 1")


@dataclass
class GenerationConfig:
    """Configuration for generation process"""
    
    # Batch processing
    batch_size: int = 32
    prefetch_batches: int = 2
    
    # Repeat generation
    num_repeats: int = 1
    repeat_strategy: str = "independent"  # independent, temperature_schedule, diverse
    temperature_schedule: Optional[List[float]] = None
    seed_increment: int = 1
    repeat_order: str = "item_first"  # item_first (AAAA BBBB) or batch_first (ABCD ABCD)
    
    # Output settings
    aggregate_responses: bool = False
    aggregation_method: str = "first"  # first, longest, majority_vote
    
    # Processing limits
    max_samples: Optional[int] = None
    start_index: Optional[int] = None
    end_index: Optional[int] = None
    
    # Checkpointing
    checkpoint_frequency: int = 100
    resume_from_checkpoint: Optional[str] = None
    
    # Error handling
    error_handling: str = "skip"  # skip, retry, fail
    max_retries: int = 3
    timeout_per_request: Optional[float] = None
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "GenerationConfig":
        """Create config from dictionary"""
        return cls(**config_dict)
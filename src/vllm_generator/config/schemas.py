from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class ConfigSchema:
    """Schema for configuration validation"""
    
    # Model configuration
    model: str
    model_revision: Optional[str] = None
    tokenizer: Optional[str] = None
    dtype: str = "auto"
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 512
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    
    # Generation configuration
    batch_size: int = 32
    num_repeats: int = 1
    repeat_strategy: str = "independent"
    temperature_schedule: Optional[List[float]] = None
    
    # Data configuration
    input_path: str = ""
    output_path: str = ""
    question_column: str = "question"
    output_format: str = "wide"
    output_column_prefix: str = "response"
    
    # Processing configuration
    max_samples: Optional[int] = None
    start_index: Optional[int] = None
    end_index: Optional[int] = None
    checkpoint_frequency: int = 100
    resume_from_checkpoint: Optional[str] = None
    
    # Parallel configuration
    parallel_mode: str = "single"
    num_workers: int = 1
    worker_gpus: Optional[List[int]] = None
    sharding_strategy: str = "contiguous"
    
    # Error handling
    error_handling: str = "skip"
    max_retries: int = 3
    timeout_per_request: Optional[float] = None
    
    # Output configuration
    output_dir: str = "./outputs"
    save_metadata: bool = True
    overwrite: bool = False
    
    # Logging configuration
    log_level: str = "INFO"
    quiet: bool = False
    verbose: bool = False
    progress_bar: bool = True


def validate_config(config: Dict[str, Any]) -> Dict[str, List[str]]:
    """Validate configuration and return errors
    
    Returns:
        Dictionary of field names to list of error messages
    """
    errors = {}
    
    # Required fields
    if "model" not in config.get("model_config", {}):
        errors["model"] = ["Model name is required"]
    
    # Numeric validations
    model_config = config.get("model_config", {})
    
    if "temperature" in model_config and model_config["temperature"] < 0:
        errors["temperature"] = ["Temperature must be non-negative"]
    
    if "top_p" in model_config:
        top_p = model_config["top_p"]
        if top_p <= 0 or top_p > 1:
            errors["top_p"] = ["top_p must be between 0 and 1"]
    
    if "gpu_memory_utilization" in model_config:
        gpu_util = model_config["gpu_memory_utilization"]
        if gpu_util <= 0 or gpu_util > 1:
            errors["gpu_memory_utilization"] = ["gpu_memory_utilization must be between 0 and 1"]
    
    if "max_tokens" in model_config and model_config["max_tokens"] <= 0:
        errors["max_tokens"] = ["max_tokens must be positive"]
    
    # Generation config validations
    gen_config = config.get("generation_config", {})
    
    if "batch_size" in gen_config and gen_config["batch_size"] <= 0:
        errors["batch_size"] = ["batch_size must be positive"]
    
    if "num_repeats" in gen_config and gen_config["num_repeats"] <= 0:
        errors["num_repeats"] = ["num_repeats must be positive"]
    
    if "repeat_strategy" in gen_config:
        valid_strategies = ["independent", "temperature_schedule", "diverse"]
        if gen_config["repeat_strategy"] not in valid_strategies:
            errors["repeat_strategy"] = [f"repeat_strategy must be one of {valid_strategies}"]
    
    if gen_config.get("repeat_strategy") == "temperature_schedule":
        schedule = gen_config.get("temperature_schedule")
        if not schedule:
            errors["temperature_schedule"] = ["temperature_schedule required for temperature_schedule strategy"]
        elif len(schedule) != gen_config.get("num_repeats", 1):
            errors["temperature_schedule"] = ["temperature_schedule length must match num_repeats"]
    
    # Parallel config validations
    parallel_config = config.get("parallel_config", {})
    
    if "parallel_mode" in parallel_config:
        valid_modes = ["single", "multi_server", "ray"]
        if parallel_config["parallel_mode"] not in valid_modes:
            errors["parallel_mode"] = [f"parallel_mode must be one of {valid_modes}"]
    
    if "num_workers" in parallel_config and parallel_config["num_workers"] <= 0:
        errors["num_workers"] = ["num_workers must be positive"]
    
    # Data config validations
    data_config = config.get("data_config", {})
    
    if "output_format" in data_config:
        valid_formats = ["wide", "long", "nested"]
        if data_config["output_format"] not in valid_formats:
            errors["output_format"] = [f"output_format must be one of {valid_formats}"]
    
    return errors


def create_config_from_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """Create configuration dictionary from parsed arguments"""
    config = {
        "model_config": {},
        "generation_config": {},
        "data_config": {},
        "parallel_config": {},
        "logging_config": {}
    }
    
    # Model config
    model_fields = [
        "model", "model_revision", "tokenizer", "dtype", "temperature",
        "top_p", "top_k", "max_tokens", "min_tokens", "repetition_penalty",
        "length_penalty", "presence_penalty", "frequency_penalty",
        "stop_sequences", "seed", "best_of", "gpu_memory_utilization",
        "tensor_parallel_size", "max_model_len", "trust_remote_code",
        "swap_space", "cpu_offload_gb", "quantization", "enforce_eager",
        "enable_prefix_caching", "max_num_seqs", "disable_log_stats"
    ]
    
    for field in model_fields:
        if field in args and args[field] is not None:
            config["model_config"][field] = args[field]
    
    # Generation config
    generation_fields = [
        "batch_size", "num_repeats", "repeat_strategy", "temperature_schedule",
        "seed_increment", "repeat_order", "aggregate_responses", "aggregation_method",
        "max_samples", "start_index", "end_index", "checkpoint_frequency",
        "resume_from_checkpoint", "error_handling", "max_retries",
        "timeout_per_request", "prefetch_batches"
    ]
    
    for field in generation_fields:
        if field in args and args[field] is not None:
            config["generation_config"][field] = args[field]
    
    # Data config
    data_fields = [
        "question_column", "output_column_prefix", "output_format",
        "prompt_template", "system_prompt", "few_shot_examples",
        "use_chat_template", "add_bos_token",
        "add_eos_token", "preprocessing_fn", "postprocessing_fn",
        "filter_column", "filter_value", "sample_fraction"
    ]
    
    for field in data_fields:
        if field in args and args[field] is not None:
            config["data_config"][field] = args[field]
    
    # Parallel config
    parallel_fields = [
        "parallel_mode", "num_workers", "worker_gpus", "ports",
        "ray_address", "ray_num_cpus", "ray_num_gpus", "sharding_strategy",
        "shard_column", "dynamic_batching", "coordinator_port",
        "checkpoint_sync", "result_aggregation", "progress_aggregation",
        "max_workers_per_node", "worker_timeout", "health_check_interval",
        "auto_restart_failed", "load_balancing", "worker_failure_mode",
        "max_worker_retries", "checkpoint_dir", "enable_work_stealing"
    ]
    
    for field in parallel_fields:
        if field in args and args[field] is not None:
            config["parallel_config"][field] = args[field]
    
    # Logging config
    logging_fields = [
        "log_level", "quiet", "verbose", "progress_bar", "save_metadata",
        "metadata_file", "track_token_usage", "save_raw_outputs"
    ]
    
    for field in logging_fields:
        if field in args and args[field] is not None:
            config["logging_config"][field] = args[field]
    
    # Top-level fields
    if "output_dir" in args and args["output_dir"] is not None:
        config["output_dir"] = args["output_dir"]
    
    if "overwrite" in args and args["overwrite"] is not None:
        config["overwrite"] = args["overwrite"]
    
    return config
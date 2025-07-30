#!/usr/bin/env python3
"""
vLLM Generator - Scalable text generation for dataframes using vLLM
"""

import sys
import logging
import argparse
from pathlib import Path
from typing import Optional, List

from src.models import ModelConfig
from src.config import ConfigParser, ConfigSchema, validate_config
from src.config.schemas import create_config_from_args
from src.pipeline import PipelineManager
from src.utils import setup_logging, parse_args_string, parse_gpu_list
from src.utils.helpers import validate_output_path

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser with all CLI options"""
    parser = argparse.ArgumentParser(
        description="vLLM Generator - Scalable text generation for dataframes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Basic I/O Arguments
    io_group = parser.add_argument_group("Input/Output")
    io_group.add_argument("--input", "-i", required=True, help="Input parquet file path")
    io_group.add_argument("--output", "-o", help="Output parquet file path or directory")
    io_group.add_argument("--question-column", default="question", help="Column name containing questions")
    io_group.add_argument("--output-column-prefix", default="response", help="Prefix for output columns")
    io_group.add_argument("--output-format", choices=["wide", "long", "nested"], default="wide",
                         help="Format for repeated outputs")
    io_group.add_argument("--overwrite", action="store_true", help="Overwrite existing output file")
    
    # Model Configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--model", "-m", help="Model name or path")
    model_group.add_argument("--model-revision", help="Model revision/branch")
    model_group.add_argument("--tokenizer", help="Tokenizer name (if different from model)")
    model_group.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], 
                            default="auto", help="Model dtype")
    model_group.add_argument("--device", default="cuda", help="Device: cuda or cpu")
    model_group.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                            help="GPU memory fraction to use")
    model_group.add_argument("--tensor-parallel-size", type=int, default=1,
                            help="Number of GPUs for tensor parallelism")
    model_group.add_argument("--max-model-len", type=int, help="Maximum sequence length")
    model_group.add_argument("--trust-remote-code", action="store_true",
                            help="Trust remote code in model files")
    
    # Generation Parameters
    gen_group = parser.add_argument_group("Generation Parameters")
    gen_group.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    gen_group.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling")
    gen_group.add_argument("--top-k", type=int, default=-1, help="Top-k sampling")
    gen_group.add_argument("--max-tokens", type=int, default=512, help="Maximum tokens to generate")
    gen_group.add_argument("--min-tokens", type=int, default=1, help="Minimum tokens to generate")
    gen_group.add_argument("--repetition-penalty", type=float, default=1.0, help="Repetition penalty")
    gen_group.add_argument("--length-penalty", type=float, default=1.0, help="Length penalty")
    gen_group.add_argument("--presence-penalty", type=float, default=0.0, help="Presence penalty")
    gen_group.add_argument("--frequency-penalty", type=float, default=0.0, help="Frequency penalty")
    gen_group.add_argument("--stop-sequences", type=str, help="Comma-separated stop sequences")
    gen_group.add_argument("--seed", type=int, help="Random seed for reproducibility")
    gen_group.add_argument("--best-of", type=int, help="Generate best_of sequences and return best")
    
    # Repeat Generation
    repeat_group = parser.add_argument_group("Repeat Generation")
    repeat_group.add_argument("--num-repeats", "-n", type=int, default=1,
                             help="Number of times to generate per question")
    repeat_group.add_argument("--repeat-strategy", choices=["independent", "temperature_schedule", "diverse"],
                             default="independent", help="Strategy for repeat generation")
    repeat_group.add_argument("--temperature-schedule", type=str,
                             help="Temperature values for each repeat (comma-separated)")
    repeat_group.add_argument("--seed-increment", type=int, default=1,
                             help="Increment seed by this value for each repeat")
    repeat_group.add_argument("--repeat-order", 
                             choices=["item_first", "batch_first"],
                             default="item_first", 
                             help="Processing order: item_first (AAAA BBBB) or batch_first (ABCD ABCD)")
    repeat_group.add_argument("--aggregate-responses", action="store_true",
                             help="Aggregate repeated responses into single output")
    repeat_group.add_argument("--aggregation-method", 
                             choices=["majority_vote", "longest", "highest_score"],
                             default="first", help="Method for aggregating responses")
    
    # Processing Configuration
    proc_group = parser.add_argument_group("Processing Configuration")
    proc_group.add_argument("--batch-size", "-b", type=int, default=32,
                           help="Batch size for generation")
    proc_group.add_argument("--max-samples", type=int, help="Maximum samples to process (for testing)")
    proc_group.add_argument("--start-index", type=int, help="Start processing from this index")
    proc_group.add_argument("--end-index", type=int, help="End processing at this index")
    proc_group.add_argument("--checkpoint-frequency", type=int, default=100,
                           help="Save checkpoint every N batches")
    proc_group.add_argument("--resume-from-checkpoint", help="Resume from checkpoint file")
    proc_group.add_argument("--num-workers", type=int, default=4,
                           help="Number of data loading workers")
    
    # Prompt Configuration
    prompt_group = parser.add_argument_group("Prompt Configuration")
    prompt_group.add_argument("--prompt-template", help="Template with {question} placeholder")
    prompt_group.add_argument("--system-prompt", help="System prompt to prepend")
    prompt_group.add_argument("--few-shot-examples", help="Path to JSON file with few-shot examples")
    prompt_group.add_argument("--use-chat-template", action="store_true",
                             help="Use tokenizer's built-in chat template")
    prompt_group.add_argument("--add-bos-token", action="store_true",
                             help="Add beginning of sequence token")
    prompt_group.add_argument("--add-eos-token", action="store_true",
                             help="Add end of sequence token")
    
    # Performance & Resource Management
    perf_group = parser.add_argument_group("Performance & Resource Management")
    perf_group.add_argument("--swap-space", type=int, default=4,
                           help="CPU swap space size in GB")
    perf_group.add_argument("--cpu-offload-gb", type=int,
                           help="Offload model weights to CPU (GB)")
    perf_group.add_argument("--quantization", choices=["awq", "gptq", "squeezellm"],
                           help="Quantization method")
    perf_group.add_argument("--enforce-eager", action="store_true",
                           help="Disable CUDA graph optimization")
    perf_group.add_argument("--enable-prefix-caching", action="store_true",
                           help="Enable automatic prefix caching")
    perf_group.add_argument("--max-num-seqs", type=int,
                           help="Maximum number of sequences per iteration")
    perf_group.add_argument("--disable-log-stats", action="store_true",
                           help="Disable logging statistics")
    
    # Data Parallelism Configuration
    parallel_group = parser.add_argument_group("Data Parallelism Configuration")
    parallel_group.add_argument("--parallel-mode", 
                               choices=["single", "multi_server", "ray"],
                               default="single", help="Parallelism mode")
    parallel_group.add_argument("--num-workers", type=int, default=1,
                               help="Number of parallel workers", dest="parallel_workers")
    parallel_group.add_argument("--worker-gpus", type=str,
                               help="GPU IDs for workers (comma-separated, e.g., 0,1,2,3)")
    parallel_group.add_argument("--ports", type=str,
                               help="Base port for vLLM servers")
    parallel_group.add_argument("--ray-address", help="Ray cluster address (for ray mode)")
    parallel_group.add_argument("--ray-num-cpus", type=int, help="CPUs per Ray worker")
    parallel_group.add_argument("--ray-num-gpus", type=int, help="GPUs per Ray worker")
    
    # Work Distribution
    dist_group = parser.add_argument_group("Work Distribution")
    dist_group.add_argument("--sharding-strategy", 
                           choices=["round_robin", "contiguous", "hash", "balanced"],
                           default="contiguous", help="How to split data")
    dist_group.add_argument("--shard-column", help="Column to use for hash-based sharding")
    dist_group.add_argument("--dynamic-batching", action="store_true",
                           help="Enable dynamic work stealing between workers")
    dist_group.add_argument("--prefetch-batches", type=int, default=2,
                           help="Number of batches to prefetch per worker")
    
    # Output & Logging
    output_group = parser.add_argument_group("Output & Logging")
    output_group.add_argument("--output-dir", default="./outputs",
                             help="Directory for logs and artifacts")
    output_group.add_argument("--log-level", 
                             choices=["debug", "info", "warning", "error"],
                             default="info", help="Logging level")
    output_group.add_argument("--save-metadata", action="store_true", default=True,
                             help="Save generation metadata")
    output_group.add_argument("--metadata-file", help="Path to save metadata")
    output_group.add_argument("--track-token-usage", action="store_true",
                             help="Track input/output token counts")
    output_group.add_argument("--save-raw-outputs", action="store_true",
                             help="Save raw model outputs before post-processing")
    output_group.add_argument("--progress-bar", action="store_true", default=True,
                             help="Show progress bar")
    output_group.add_argument("--quiet", "-q", action="store_true",
                             help="Minimal output")
    output_group.add_argument("--verbose", "-v", action="store_true",
                             help="Verbose output")
    
    # Advanced Features
    adv_group = parser.add_argument_group("Advanced Features")
    adv_group.add_argument("--config-file", "-c", help="Path to YAML/JSON configuration file")
    adv_group.add_argument("--dry-run", action="store_true",
                          help="Validate configuration without running")
    adv_group.add_argument("--validate-only", action="store_true",
                          help="Only validate input data")
    adv_group.add_argument("--preprocessing-fn", help="Path to Python file with preprocessing function")
    adv_group.add_argument("--postprocessing-fn", help="Path to Python file with postprocessing function")
    adv_group.add_argument("--error-handling", choices=["skip", "retry", "fail"],
                          default="skip", help="Strategy for handling errors")
    adv_group.add_argument("--max-retries", type=int, default=3,
                          help="Maximum retries for failed generations")
    adv_group.add_argument("--timeout-per-request", type=float,
                          help="Timeout in seconds per generation request")
    adv_group.add_argument("--filter-column", help="Column name for filtering rows")
    adv_group.add_argument("--filter-value", help="Value to filter by")
    adv_group.add_argument("--sample-fraction", type=float,
                          help="Fraction of data to sample randomly")
    
    # Model arguments (catch-all)
    parser.add_argument("--model-args", type=str,
                       help="Additional model arguments as key=value pairs")
    
    return parser


def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else ("WARNING" if args.quiet else args.log_level.upper())
    setup_logging(level=log_level)
    
    try:
        # Load configuration
        config = {}
        
        # 1. Load from config file if provided
        if args.config_file:
            logger.info(f"Loading configuration from {args.config_file}")
            file_config = ConfigParser.load_config(args.config_file)
            config = ConfigParser.merge_configs(config, file_config)
        
        # 2. Load from environment variables
        env_config = ConfigParser.load_from_env()
        if env_config:
            logger.info("Loading configuration from environment variables")
            config = ConfigParser.merge_configs(config, env_config)
        
        # 3. Override with command line arguments
        cli_config = create_config_from_args(vars(args))
        config = ConfigParser.merge_configs(config, cli_config)
        
        # Add model args if provided
        if args.model_args:
            model_args = parse_args_string(args.model_args)
            config["model_config"] = ConfigParser.merge_configs(
                config.get("model_config", {}),
                model_args
            )
        
        # Set required fields from args
        if not config.get("model_config", {}).get("model") and args.model:
            config.setdefault("model_config", {})["model"] = args.model
        
        # Validate configuration
        errors = validate_config(config)
        if errors:
            logger.error("Configuration validation failed:")
            for field, messages in errors.items():
                for msg in messages:
                    logger.error(f"  {field}: {msg}")
            sys.exit(1)
        
        # Handle special commands
        if args.dry_run:
            logger.info("Configuration validated successfully (dry run)")
            return
        
        # Set output path
        if not args.output:
            timestamp = Path(args.input).stem
            args.output = f"{args.output_dir}/output_{timestamp}.parquet"
        
        # Validate output path
        output_path = validate_output_path(args.output, args.overwrite)
        
        # Parse worker GPUs
        if args.worker_gpus:
            worker_gpus = parse_gpu_list(args.worker_gpus)
            config.setdefault("parallel_config", {})["worker_gpus"] = worker_gpus
        
        # Parse temperature schedule
        if args.temperature_schedule:
            temps = [float(t) for t in args.temperature_schedule.split(",")]
            config.setdefault("generation_config", {})["temperature_schedule"] = temps
        
        # Create model config
        model_config = ModelConfig.from_dict(config.get("model_config", {}))
        
        # Create pipeline manager
        manager = PipelineManager(
            model_config=model_config,
            generation_config=config.get("generation_config", {}),
            parallel_mode=args.parallel_mode,
            num_workers=args.parallel_workers or 1,
            worker_gpus=config.get("parallel_config", {}).get("worker_gpus"),
            base_port=int(args.ports) if args.ports else 8000
        )
        
        # Run pipeline
        result = manager.run(
            input_path=args.input,
            output_path=str(output_path),
            question_column=args.question_column,
            output_format=args.output_format,
            **config.get("data_config", {})
        )
        
        # Print summary
        logger.info("Generation completed successfully!")
        logger.info(f"Output saved to: {result['output_path']}")
        logger.info(f"Total samples processed: {result.get('total_samples', 0)}")
        
        if "metrics" in result:
            metrics = result["metrics"]
            logger.info(f"Metrics: {metrics}")
        
    except KeyboardInterrupt:
        logger.info("Generation interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
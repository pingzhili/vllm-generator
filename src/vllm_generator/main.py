"""Main entry point for vLLM generator."""

import asyncio
import argparse
import sys
from pathlib import Path

from .config import ConfigParser, Config
from .pipeline import GenerationPipeline
from .utils import get_logger
from . import __version__


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="vllm-generator",
        description="vLLM Data Generation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate with configuration file
  vllm-generator generate --config config.yaml

  # Generate with command line arguments
  vllm-generator generate -i data.parquet -o output.parquet -m http://localhost:8000

  # Generate with multiple samples
  vllm-generator generate -c config.yaml --num-samples 5 --temperature 0.8

  # Resume from checkpoint
  vllm-generator generate -c config.yaml --resume

  # Validate configuration
  vllm-generator validate --config config.yaml

  # List available models
  vllm-generator list-models --model-url http://localhost:8000
        """
    )
    
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"%(prog)s {__version__}"
    )
    
    # Global arguments
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)"
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Log to file in addition to console"
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output"
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Generate command
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate responses using vLLM models"
    )
    add_generate_args(generate_parser)
    
    # Validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate configuration file"
    )
    validate_parser.add_argument(
        "--config", "-c",
        type=Path,
        required=True,
        help="Path to YAML configuration file"
    )
    
    # List models command
    list_parser = subparsers.add_parser(
        "list-models",
        help="List available models from vLLM servers"
    )
    list_parser.add_argument(
        "--model-url", "-m",
        type=str,
        help="vLLM server URL"
    )
    list_parser.add_argument(
        "--model-urls",
        nargs="+",
        help="Multiple vLLM server URLs"
    )
    list_parser.add_argument(
        "--config", "-c",
        type=Path,
        help="Configuration file with model URLs"
    )
    
    return parser


def add_generate_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for generate command."""
    # Configuration
    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument(
        "--config", "-c",
        type=Path,
        help="Path to YAML configuration file"
    )
    config_group.add_argument(
        "--save-config",
        type=Path,
        help="Save current configuration to file"
    )
    
    # Data options
    data_group = parser.add_argument_group("Data Options")
    data_group.add_argument(
        "--input", "-i",
        type=Path,
        help="Input parquet file path"
    )
    data_group.add_argument(
        "--output", "-o",
        type=Path,
        help="Output parquet file path"
    )
    data_group.add_argument(
        "--input-column",
        type=str,
        help="Column name containing input text (default: question)"
    )
    data_group.add_argument(
        "--output-column",
        type=str,
        help="Column name for output text (default: response)"
    )
    data_group.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Directory for saving checkpoints (default: ./checkpoints)"
    )
    data_group.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint"
    )
    
    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model-url", "-m",
        type=str,
        help="vLLM server URL (e.g., http://localhost:8000)"
    )
    model_group.add_argument(
        "--model-urls",
        nargs="+",
        help="Multiple vLLM server URLs for parallel processing"
    )
    
    # Generation parameters
    gen_group = parser.add_argument_group("Generation Parameters")
    gen_group.add_argument(
        "--num-samples", "-n",
        type=int,
        help="Number of samples per input (default: 1)"
    )
    gen_group.add_argument(
        "--temperature", "-t",
        type=float,
        help="Sampling temperature (default: 1.0)"
    )
    gen_group.add_argument(
        "--top-p",
        type=float,
        help="Top-p sampling parameter (default: 1.0)"
    )
    gen_group.add_argument(
        "--top-k",
        type=int,
        help="Top-k sampling parameter (default: -1)"
    )
    gen_group.add_argument(
        "--max-tokens",
        type=int,
        help="Maximum tokens to generate (default: 512)"
    )
    gen_group.add_argument(
        "--stop-sequences",
        nargs="+",
        help="Stop sequences for generation"
    )
    gen_group.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility"
    )
    gen_group.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable thinking mode for reasoning models"
    )
    
    # Processing options
    proc_group = parser.add_argument_group("Processing Options")
    proc_group.add_argument(
        "--batch-size", "-b",
        type=int,
        help="Batch size for processing (default: 32)"
    )
    proc_group.add_argument(
        "--num-workers", "-w",
        type=int,
        help="Number of parallel workers (default: 1)"
    )
    proc_group.add_argument(
        "--timeout",
        type=float,
        help="Request timeout in seconds (default: 300)"
    )
    proc_group.add_argument(
        "--max-retries",
        type=int,
        help="Maximum retries for failed requests (default: 3)"
    )
    proc_group.add_argument(
        "--retry-delay",
        type=float,
        help="Delay between retries in seconds (default: 1.0)"
    )
    
    # Display options
    display_group = parser.add_argument_group("Display Options")
    display_group.add_argument(
        "--progress-bar",
        action="store_true",
        default=True,
        help="Show progress bar (default)"
    )
    display_group.add_argument(
        "--no-progress-bar",
        dest="progress_bar",
        action="store_false",
        help="Disable progress bar"
    )
    display_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without making actual API calls"
    )


async def generate_command(args: argparse.Namespace) -> int:
    """Execute generate command."""
    logger = get_logger("CLI")
    
    try:
        # Load or create configuration
        if args.config:
            config = ConfigParser.from_yaml(args.config)
            # Merge CLI arguments
            cli_args = vars(args)
            config = ConfigParser.merge_cli_args(config, cli_args)
        else:
            # Create config from CLI arguments
            if not all([args.input, args.output, (args.model_url or args.model_urls)]):
                logger.error(
                    "Either provide a config file or specify --input, --output, and --model-url"
                )
                return 1
            
            config = ConfigParser.create_default_config(
                input_path=str(args.input),
                output_path=str(args.output),
                model_url=args.model_url or args.model_urls[0],
                **vars(args)
            )
        
        # Save config if requested
        if args.save_config:
            ConfigParser.to_yaml(config, args.save_config)
            logger.info(f"Saved configuration to {args.save_config}")
        
        # Run pipeline
        pipeline = GenerationPipeline(config)
        results = await pipeline.run(
            dry_run=args.dry_run,
            progress_bar=args.progress_bar
        )
        
        # Print summary
        logger.info(f"✓ Generated {results['total_processed']} responses")
        logger.info(f"✓ Processing time: {results['processing_time']:.2f}s")
        logger.info(f"✓ Throughput: {results['prompts_per_second']:.2f} prompts/s")
        
        return 0
    
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return 1


async def validate_command(args: argparse.Namespace) -> int:
    """Execute validate command."""
    logger = get_logger("CLI")
    
    try:
        config = ConfigParser.from_yaml(args.config)
        pipeline = GenerationPipeline(config)
        
        if await pipeline.validate():
            logger.info("✓ Configuration is valid")
            return 0
        else:
            logger.error("✗ Configuration validation failed")
            return 1
    
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return 1


async def list_models_command(args: argparse.Namespace) -> int:
    """Execute list-models command."""
    logger = get_logger("CLI")
    
    try:
        # Determine model URLs
        if args.config:
            config = ConfigParser.from_yaml(args.config)
            model_urls = [str(m.url) for m in config.models]
        elif args.model_urls:
            model_urls = args.model_urls
        elif args.model_url:
            model_urls = [args.model_url]
        else:
            logger.error("Provide --model-url, --model-urls, or --config")
            return 1
        
        # Create minimal config
        config_data = {
            "data": {
                "input_path": "dummy.parquet",
                "output_path": "dummy_out.parquet"
            },
            "models": [{"url": url} for url in model_urls]
        }
        config = Config(**config_data)
        
        # List models
        pipeline = GenerationPipeline(config)
        model_lists = await pipeline.list_models()
        
        # Display results
        for endpoint, models in model_lists.items():
            logger.info(f"\nEndpoint: {endpoint}")
            if models:
                for model in models:
                    logger.info(f"  - {model}")
            else:
                logger.info("  No models found")
        
        return 0
    
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        return 1


async def async_main(args: argparse.Namespace) -> int:
    """Async main function."""
    if args.command == "generate":
        return await generate_command(args)
    elif args.command == "validate":
        return await validate_command(args)
    elif args.command == "list-models":
        return await list_models_command(args)
    else:
        parser = create_parser()
        parser.print_help()
        return 1


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Configure logging
    if args.no_color:
        import os
        os.environ["NO_COLOR"] = "1"
    
    # Run async main
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    sys.exit(main())
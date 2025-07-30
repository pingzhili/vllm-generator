import pytest

from vllm_generator import setup_logging, get_logger, ProgressLogger
from vllm_generator import (
    parse_args_string, parse_value, load_module_from_path,
    parse_gpu_list, format_bytes, format_duration,
    validate_output_path, merge_configs
)


class TestLogging:
    
    def test_setup_logging(self, temp_dir):
        """Test logging setup"""
        log_file = temp_dir / "test.log"
        
        setup_logging(
            level="INFO",
            log_file=str(log_file)
        )
        
        logger = get_logger("test")
        logger.info("Test message")
        
        # Check that log file was created
        assert log_file.exists()
        
        # Check log content
        log_content = log_file.read_text()
        assert "Test message" in log_content
    
    def test_progress_logger(self):
        """Test progress logger"""
        import logging
        logger = logging.getLogger("test_progress")
        
        progress_logger = ProgressLogger(logger, interval=3)
        
        messages = []
        # Mock the logger
        logger.info = lambda msg: messages.append(msg)
        
        # Log multiple messages
        for i in range(10):
            progress_logger.log(f"Processing item {i}")
        
        # Should log at intervals
        assert len(messages) == 4  # Items 0, 3, 6, 9
        assert "[3]" in messages[1]
        assert "[6]" in messages[2]
        
        # Test force logging
        progress_logger.log("Forced message", force=True)
        assert len(messages) == 5


class TestHelpers:
    
    def test_parse_args_string(self):
        """Test parsing argument strings"""
        # Basic parsing
        args = parse_args_string("model=gpt2,temperature=0.8")
        assert args["model"] == "gpt2"
        assert args["temperature"] == 0.8
        
        # Complex parsing with quotes
        args = parse_args_string('prompt="Hello, world",max_tokens=100')
        assert args["prompt"] == "Hello, world"
        assert args["max_tokens"] == 100
        
        # Boolean and None values
        args = parse_args_string("use_cache=true,filter=none")
        assert args["use_cache"] is True
        assert args["filter"] is None
        
        # List values
        args = parse_args_string("stop_sequences=[\\n,END],temps=[0.5,0.7,0.9]")
        assert args["stop_sequences"] == ["\\n", "END"]
        assert args["temps"] == [0.5, 0.7, 0.9]
    
    def test_parse_value(self):
        """Test value parsing"""
        assert parse_value("true") is True
        assert parse_value("false") is False
        assert parse_value("none") is None
        assert parse_value("123") == 123
        assert parse_value("123.45") == 123.45
        assert parse_value("hello") == "hello"
        assert parse_value("[1,2,3]") == [1, 2, 3]
        assert parse_value("[]") == []
    
    def test_load_module_from_path(self, temp_dir):
        """Test loading Python module from file"""
        module_file = temp_dir / "custom_module.py"
        module_file.write_text("""
def custom_function(x):
    return x * 2

CUSTOM_CONSTANT = 42
""")
        
        module = load_module_from_path(str(module_file))
        
        assert hasattr(module, "custom_function")
        assert hasattr(module, "CUSTOM_CONSTANT")
        assert module.custom_function(5) == 10
        assert module.CUSTOM_CONSTANT == 42
    
    def test_parse_gpu_list(self):
        """Test GPU list parsing"""
        # Single GPUs
        assert parse_gpu_list("0,1,2,3") == [0, 1, 2, 3]
        
        # Range
        assert parse_gpu_list("0-3") == [0, 1, 2, 3]
        
        # Mixed
        assert parse_gpu_list("0,2-4,6") == [0, 2, 3, 4, 6]
        
        # Duplicates removed and sorted
        assert parse_gpu_list("3,1,2,1,3") == [1, 2, 3]
        
        # Empty
        assert parse_gpu_list("") == []
    
    def test_format_bytes(self):
        """Test byte formatting"""
        assert format_bytes(100) == "100.0 B"
        assert format_bytes(1024) == "1.0 KB"
        assert format_bytes(1024 * 1024) == "1.0 MB"
        assert format_bytes(1024 * 1024 * 1024) == "1.0 GB"
        assert format_bytes(1536 * 1024) == "1.5 MB"
    
    def test_format_duration(self):
        """Test duration formatting"""
        assert format_duration(30) == "30.0s"
        assert format_duration(90) == "1.5m"
        assert format_duration(3600) == "1.0h"
        assert format_duration(5400) == "1.5h"
    
    def test_validate_output_path(self, temp_dir):
        """Test output path validation"""
        # New file - should work
        new_file = temp_dir / "new_output.parquet"
        validated = validate_output_path(str(new_file))
        assert validated == new_file
        
        # Existing file without overwrite - should fail
        existing_file = temp_dir / "existing.parquet"
        existing_file.write_text("test")
        
        with pytest.raises(FileExistsError):
            validate_output_path(str(existing_file), overwrite=False)
        
        # Existing file with overwrite - should work
        validated = validate_output_path(str(existing_file), overwrite=True)
        assert validated == existing_file
        
        # Non-existent parent directory - should create it
        nested_file = temp_dir / "new_dir" / "output.parquet"
        validated = validate_output_path(str(nested_file))
        assert validated.parent.exists()
    
    def test_merge_configs(self):
        """Test configuration merging"""
        base = {
            "model": "gpt2",
            "params": {
                "temperature": 0.7,
                "max_tokens": 100
            },
            "features": ["feature1", "feature2"]
        }
        
        override = {
            "params": {
                "temperature": 0.9,
                "top_p": 0.95
            },
            "features": ["feature3"],
            "new_field": "value"
        }
        
        merged = merge_configs(base, override)
        
        assert merged["model"] == "gpt2"  # Preserved
        assert merged["params"]["temperature"] == 0.9  # Overridden
        assert merged["params"]["max_tokens"] == 100  # Preserved
        assert merged["params"]["top_p"] == 0.95  # Added
        assert merged["features"] == ["feature3"]  # Replaced (not merged)
        assert merged["new_field"] == "value"  # Added
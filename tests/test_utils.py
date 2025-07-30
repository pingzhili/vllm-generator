"""Tests for utility functions."""

import pytest
from pathlib import Path
import pandas as pd
import json

from vllm_generator.utils import (
    generate_batch_id,
    chunk_dataframe,
    distribute_items,
    format_duration,
    ensure_directory,
    get_timestamp,
    safe_json_loads,
    truncate_text,
    merge_dicts,
    calculate_statistics
)


class TestUtilities:
    """Test utility functions."""
    
    def test_generate_batch_id(self):
        """Test batch ID generation."""
        # String input
        id1 = generate_batch_id("test_string")
        assert len(id1) == 8
        assert id1 == generate_batch_id("test_string")  # Should be deterministic
        
        # Dict input
        data = {"key": "value", "number": 123}
        id2 = generate_batch_id(data)
        assert len(id2) == 8
        assert id2 == generate_batch_id(data)  # Should be deterministic
        
        # Different inputs should give different IDs
        assert id1 != id2
    
    def test_chunk_dataframe(self):
        """Test dataframe chunking."""
        df = pd.DataFrame({"col": range(10)})
        
        chunks = chunk_dataframe(df, 3)
        
        assert len(chunks) == 4  # 10 rows / 3 chunk size = 4 chunks
        assert len(chunks[0]) == 3
        assert len(chunks[1]) == 3
        assert len(chunks[2]) == 3
        assert len(chunks[3]) == 1
    
    def test_distribute_items(self):
        """Test item distribution."""
        items = list(range(10))
        
        # Distribute among 3 workers
        distributed = distribute_items(items, 3)
        
        assert len(distributed) == 3
        assert len(distributed[0]) == 4  # Worker 0 gets items 0, 3, 6, 9
        assert len(distributed[1]) == 3  # Worker 1 gets items 1, 4, 7
        assert len(distributed[2]) == 3  # Worker 2 gets items 2, 5, 8
        
        # Verify all items are distributed
        all_items = []
        for chunk in distributed:
            all_items.extend(chunk)
        assert sorted(all_items) == items
        
        # Test with more workers than items
        distributed = distribute_items([1, 2], 5)
        assert len(distributed) == 2  # Only 2 non-empty chunks
    
    def test_distribute_items_invalid(self):
        """Test distribute_items with invalid input."""
        with pytest.raises(ValueError):
            distribute_items([1, 2, 3], 0)
    
    def test_format_duration(self):
        """Test duration formatting."""
        assert format_duration(45.5) == "45.5s"
        assert format_duration(90) == "1.5m"
        assert format_duration(3900) == "1.1h"
        assert format_duration(7200) == "2.0h"
    
    def test_ensure_directory(self, temp_dir):
        """Test directory creation."""
        new_dir = temp_dir / "nested" / "directory"
        
        result = ensure_directory(new_dir)
        
        assert new_dir.exists()
        assert new_dir.is_dir()
        assert result == new_dir
        
        # Should not fail if directory already exists
        result2 = ensure_directory(new_dir)
        assert result2 == new_dir
    
    def test_get_timestamp(self):
        """Test timestamp generation."""
        ts1 = get_timestamp()
        assert len(ts1) == 15  # YYYYMMDD_HHMMSS
        assert ts1[:8].isdigit()  # Date part
        assert ts1[8] == "_"
        assert ts1[9:].isdigit()  # Time part
    
    def test_safe_json_loads(self):
        """Test safe JSON loading."""
        # Valid JSON
        assert safe_json_loads('{"key": "value"}') == {"key": "value"}
        assert safe_json_loads('[1, 2, 3]') == [1, 2, 3]
        
        # Invalid JSON
        assert safe_json_loads('invalid json', default={}) == {}
        assert safe_json_loads('invalid json', default=None) is None
        assert safe_json_loads('', default=[]) == []
    
    def test_truncate_text(self):
        """Test text truncation."""
        # Short text
        assert truncate_text("Short", max_length=10) == "Short"
        
        # Long text
        long_text = "This is a very long text that needs truncation"
        assert truncate_text(long_text, max_length=20) == "This is a very lo..."
        assert len(truncate_text(long_text, max_length=20)) == 20
        
        # Custom suffix
        assert truncate_text(long_text, max_length=20, suffix="[...]") == "This is a very[...]"
    
    def test_merge_dicts(self):
        """Test dictionary merging."""
        base = {
            "a": 1,
            "b": {"x": 2, "y": 3},
            "c": [1, 2, 3]
        }
        
        override = {
            "a": 10,
            "b": {"y": 30, "z": 40},
            "d": "new"
        }
        
        result = merge_dicts(base, override)
        
        assert result["a"] == 10  # Overridden
        assert result["b"]["x"] == 2  # Preserved
        assert result["b"]["y"] == 30  # Overridden
        assert result["b"]["z"] == 40  # Added
        assert result["c"] == [1, 2, 3]  # Preserved
        assert result["d"] == "new"  # Added
        
        # Original dicts should be unchanged
        assert base["a"] == 1
        assert base["b"]["y"] == 3
    
    def test_calculate_statistics(self):
        """Test statistics calculation."""
        # Normal case
        values = [1, 2, 3, 4, 5]
        stats = calculate_statistics(values)
        
        assert stats["count"] == 5
        assert stats["mean"] == 3.0
        assert stats["min"] == 1
        assert stats["max"] == 5
        assert stats["std"] > 0
        
        # Single value
        stats = calculate_statistics([42])
        assert stats["count"] == 1
        assert stats["mean"] == 42
        assert stats["std"] == 0
        
        # Empty list
        stats = calculate_statistics([])
        assert stats["count"] == 0
        assert stats["mean"] == 0
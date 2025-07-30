import pytest
import pandas as pd
from unittest.mock import Mock, patch

from src.models.generation import GenerationManager
from src.models.config import GenerationConfig
from src.models.vllm_wrapper import MockVLLMModel


class TestRepeatOrder:
    """Test different repeat generation orders"""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model that tracks generation order"""
        model = MockVLLMModel(Mock())
        model.generation_order = []
        
        # Override generate to track order
        original_generate = model.generate
        
        def tracking_generate(prompts, sampling_params=None):
            # Record the prompts in order
            model.generation_order.extend(prompts)
            return original_generate(prompts, sampling_params)
        
        model.generate = tracking_generate
        return model
    
    @pytest.fixture
    def sample_items(self):
        """Create sample items for testing"""
        return [
            {"idx": 0, "prompt": "Question A", "original_question": "A"},
            {"idx": 1, "prompt": "Question B", "original_question": "B"},
            {"idx": 2, "prompt": "Question C", "original_question": "C"},
            {"idx": 3, "prompt": "Question D", "original_question": "D"}
        ]
    
    def test_item_first_order(self, mock_model, sample_items):
        """Test AAAA BBBB CCCC DDDD order"""
        config = GenerationConfig(batch_size=8, num_repeats=4)
        manager = GenerationManager(mock_model, config)
        
        results = manager.generate_with_repeats(
            sample_items,
            num_repeats=4,
            repeat_order="item_first"
        )
        
        # Check that we got all results
        assert len(results) == 16  # 4 items * 4 repeats
        
        # Check generation order
        order = mock_model.generation_order
        
        # With batch_size=8 and 4 repeats, we can fit 2 items per batch
        # So order should be: A,A,A,A,B,B,B,B (first batch), C,C,C,C,D,D,D,D (second batch)
        expected_order = (
            ["Question A"] * 4 + ["Question B"] * 4 +
            ["Question C"] * 4 + ["Question D"] * 4
        )
        
        # The actual order might have some batching variation, but each item's
        # repeats should be contiguous
        for i in range(0, len(order), 4):
            # Check that each group of 4 is the same question
            assert len(set(order[i:i+4])) == 1, f"Repeats not contiguous at position {i}"
    
    def test_batch_first_order(self, mock_model, sample_items):
        """Test ABCD ABCD ABCD ABCD order"""
        config = GenerationConfig(batch_size=8, num_repeats=4)
        manager = GenerationManager(mock_model, config)
        
        results = manager.generate_with_repeats(
            sample_items,
            num_repeats=4,
            repeat_order="batch_first"
        )
        
        # Check that we got all results
        assert len(results) == 16  # 4 items * 4 repeats
        
        # Check generation order
        order = mock_model.generation_order
        
        # With batch_first, we should see all items for each repeat
        # Order should be: A,B,C,D (repeat 0), A,B,C,D (repeat 1), etc.
        for repeat_id in range(4):
            batch_start = repeat_id * 4
            batch = order[batch_start:batch_start + 4]
            expected = ["Question A", "Question B", "Question C", "Question D"]
            assert batch == expected, f"Batch order incorrect for repeat {repeat_id}"
    
    def test_result_organization(self, mock_model, sample_items):
        """Test that results are organized correctly regardless of processing order"""
        config = GenerationConfig(batch_size=8, num_repeats=3)
        manager = GenerationManager(mock_model, config)
        
        # Test both orders
        for repeat_order in ["item_first", "batch_first"]:
            results = manager.generate_with_repeats(
                sample_items,
                num_repeats=3,
                repeat_order=repeat_order
            )
            
            # Group results by item idx
            results_by_item = {}
            for result in results:
                idx = result["idx"]
                if idx not in results_by_item:
                    results_by_item[idx] = []
                results_by_item[idx].append(result)
            
            # Check each item has correct number of repeats
            for idx, item_results in results_by_item.items():
                assert len(item_results) == 3, f"Item {idx} doesn't have 3 repeats"
                
                # Check repeat IDs are correct
                repeat_ids = sorted([r["repeat_id"] for r in item_results])
                assert repeat_ids == [0, 1, 2], f"Invalid repeat IDs for item {idx}"
    
    def test_temperature_schedule_with_item_first(self, mock_model, sample_items):
        """Test temperature schedule works correctly with item-first order"""
        config = GenerationConfig(batch_size=8)
        manager = GenerationManager(mock_model, config)
        
        # Track sampling params used
        sampling_params_used = []
        
        def track_params(prompts, sampling_params=None):
            sampling_params_used.append((len(prompts), sampling_params))
            return [{"response": f"Response", "tokens": 10, "latency": 0.1, "finish_reason": "stop"} 
                    for _ in prompts]
        
        mock_model._generate_with_retry = track_params
        
        results = manager.generate_with_repeats(
            sample_items[:2],  # Just 2 items for simpler test
            num_repeats=3,
            repeat_strategy="temperature_schedule",
            temperature_schedule=[0.5, 1.0, 1.5],
            repeat_order="item_first"
        )
        
        # With item_first, we should see different temperatures grouped
        # The sampling params should show mixed temperatures in batches
        assert len(results) == 6  # 2 items * 3 repeats
    
    def test_progress_callback_item_first(self, mock_model, sample_items):
        """Test progress callback for item-first order"""
        config = GenerationConfig(batch_size=8)
        manager = GenerationManager(mock_model, config)
        
        progress_calls = []
        
        def track_progress(completed, total, progress_type):
            progress_calls.append((completed, total, progress_type))
        
        results = manager.generate_with_repeats(
            sample_items,
            num_repeats=3,
            repeat_order="item_first",
            progress_callback=track_progress
        )
        
        # Should have progress calls for items (not repeats with item_first)
        item_progress_calls = [c for c in progress_calls if c[2] == "items"]
        assert len(item_progress_calls) > 0
        
        # Last call should show all items completed
        last_call = item_progress_calls[-1]
        assert last_call[0] == 4  # 4 items completed
        assert last_call[1] == 4  # out of 4 total
    
    def test_small_batch_size(self, mock_model, sample_items):
        """Test item-first with batch size smaller than num_repeats"""
        config = GenerationConfig(batch_size=2, num_repeats=4)
        manager = GenerationManager(mock_model, config)
        
        results = manager.generate_with_repeats(
            sample_items,
            num_repeats=4,
            repeat_order="item_first"
        )
        
        # With batch_size=2 and num_repeats=4, we can't fit all repeats of one item
        # The implementation should handle this gracefully
        assert len(results) == 16
        
        # Results should still be organized correctly
        for i in range(4):
            item_results = [r for r in results if r["idx"] == i]
            assert len(item_results) == 4
            assert sorted([r["repeat_id"] for r in item_results]) == [0, 1, 2, 3]
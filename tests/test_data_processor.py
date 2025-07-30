"""Tests for data processor."""

import pandas as pd

from vllm_generator.data import DataProcessor


class TestDataProcessor:
    """Test data processor functionality."""
    
    def test_prepare_prompts_no_template(self):
        """Test preparing prompts without template."""
        processor = DataProcessor()
        texts = ["Q1", "Q2", "Q3"]
        
        prompts = processor.prepare_prompts(texts)
        assert prompts == texts
    
    def test_prepare_prompts_with_template(self):
        """Test preparing prompts with template."""
        processor = DataProcessor()
        texts = ["Q1", "Q2"]
        template = "Question: {text}\nAnswer:"
        
        prompts = processor.prepare_prompts(texts, prompt_template=template)
        
        assert prompts[0] == "Question: Q1\nAnswer:"
        assert prompts[1] == "Question: Q2\nAnswer:"
    
    def test_prepare_prompts_with_system(self):
        """Test preparing prompts with system prompt."""
        processor = DataProcessor()
        texts = ["Q1"]
        template = "Q: {text}"
        system = "You are a helpful assistant."
        
        prompts = processor.prepare_prompts(
            texts,
            prompt_template=template,
            system_prompt=system
        )
        
        assert prompts[0] == "You are a helpful assistant.\n\nQ: Q1"
    
    def test_process_responses_dict(self, mock_vllm_response):
        """Test processing dictionary responses."""
        processor = DataProcessor()
        responses = [mock_vllm_response, mock_vllm_response]
        
        texts = processor.process_responses(responses)
        
        assert len(texts) == 2
        assert texts[0] == "This is a test response"
        assert texts[1] == "This is a test response"
    
    def test_process_responses_string(self):
        """Test processing string responses."""
        processor = DataProcessor()
        responses = ["Response 1", "Response 2"]
        
        texts = processor.process_responses(responses)
        
        assert texts == ["Response 1", "Response 2"]
    
    def test_process_responses_mixed(self, mock_vllm_response):
        """Test processing mixed response types."""
        processor = DataProcessor()
        responses = [
            mock_vllm_response,
            "Direct string response",
            {"text": "Simple dict response"}
        ]
        
        texts = processor.process_responses(responses)
        
        assert texts[0] == "This is a test response"
        assert texts[1] == "Direct string response"
        assert texts[2] == "Simple dict response"
    
    def test_group_responses_by_input(self):
        """Test grouping responses by input."""
        processor = DataProcessor()
        
        # Single sample per input
        responses = ["R1", "R2", "R3"]
        grouped = processor.group_responses_by_input(responses, num_samples=1)
        assert grouped == [["R1"], ["R2"], ["R3"]]
        
        # Multiple samples per input
        responses = ["R1a", "R1b", "R1c", "R2a", "R2b", "R2c"]
        grouped = processor.group_responses_by_input(responses, num_samples=3)
        assert grouped == [["R1a", "R1b", "R1c"], ["R2a", "R2b", "R2c"]]
    
    def test_flatten_responses(self):
        """Test flattening grouped responses."""
        processor = DataProcessor()
        grouped = [["R1a", "R1b"], ["R2a", "R2b"], ["R3a", "R3b"]]
        
        flattened = processor.flatten_responses(grouped)
        
        assert flattened == ["R1a", "R1b", "R2a", "R2b", "R3a", "R3b"]
    
    def test_create_response_dataframe_single(self):
        """Test creating response dataframe with single responses."""
        processor = DataProcessor()
        
        input_df = pd.DataFrame({
            "question": ["Q1", "Q2", "Q3"],
            "id": [1, 2, 3]
        })
        
        responses = ["R1", "R2", "R3"]
        
        output_df = processor.create_response_dataframe(
            input_df,
            responses,
            "question",
            "response"
        )
        
        assert len(output_df) == 3
        assert output_df["response"].tolist() == ["R1", "R2", "R3"]
        assert output_df["id"].tolist() == [1, 2, 3]
    
    def test_create_response_dataframe_multiple(self):
        """Test creating response dataframe with multiple responses."""
        processor = DataProcessor()
        
        input_df = pd.DataFrame({
            "question": ["Q1", "Q2"],
            "id": [1, 2]
        })
        
        responses = [["R1a", "R1b"], ["R2a", "R2b", "R2c"]]
        
        output_df = processor.create_response_dataframe(
            input_df,
            responses,
            "question",
            "response"
        )
        
        assert len(output_df) == 5  # 2 + 3 responses
        assert output_df["response"].tolist() == ["R1a", "R1b", "R2a", "R2b", "R2c"]
        assert output_df["sample_idx"].tolist() == [0, 1, 0, 1, 2]
    
    def test_validate_responses(self):
        """Test response validation."""
        processor = DataProcessor()
        
        # Valid count
        assert processor.validate_responses(["R1", "R2", "R3"], expected_count=3)
        
        # Invalid count
        assert not processor.validate_responses(["R1", "R2"], expected_count=3)
    
    def test_clean_text(self):
        """Test text cleaning."""
        processor = DataProcessor()
        
        text = "  This   has    extra   spaces  \n\n  and newlines  "
        cleaned = processor.clean_text(text)
        
        assert cleaned == "This has extra spaces and newlines"
    
    def test_truncate_responses(self):
        """Test response truncation."""
        processor = DataProcessor()
        
        responses = [
            "Short response",
            "This is a very long response that exceeds the maximum length limit"
        ]
        
        truncated = processor.truncate_responses(responses, max_length=20)
        
        assert truncated[0] == "Short response"
        assert truncated[1] == "This is a very long "
        assert len(truncated[1]) == 20
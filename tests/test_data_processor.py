import pytest
import pandas as pd
import json
from pathlib import Path

from src.data.processor import DataProcessor


class TestDataProcessor:
    
    def test_basic_processing(self, sample_dataframe):
        """Test basic data processing"""
        processor = DataProcessor()
        items = processor.process_batch(sample_dataframe, question_column="question")
        
        assert len(items) == len(sample_dataframe)
        assert all("prompt" in item for item in items)
        assert all("original_question" in item for item in items)
        assert items[0]["prompt"] == sample_dataframe.iloc[0]["question"]
    
    def test_prompt_template(self, sample_dataframe):
        """Test custom prompt template"""
        processor = DataProcessor(
            prompt_template="Question: {question}\nAnswer:"
        )
        items = processor.process_batch(sample_dataframe)
        
        expected = f"Question: {sample_dataframe.iloc[0]['question']}\nAnswer:"
        assert items[0]["prompt"] == expected
    
    def test_use_chat_template(self, sample_dataframe):
        """Test use_chat_template with mock tokenizer"""
        # Create a mock tokenizer
        class MockTokenizer:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                formatted = []
                for msg in messages:
                    formatted.append(f"{msg['role']}: {msg['content']}")
                result = "\n".join(formatted)
                if add_generation_prompt:
                    result += "\nAssistant:"
                return result
        
        processor = DataProcessor(
            system_prompt="You are a helpful assistant.",
            use_chat_template=True,
            tokenizer=MockTokenizer()
        )
        items = processor.process_batch(sample_dataframe)
        
        prompt = items[0]["prompt"]
        assert "system: You are a helpful assistant." in prompt
        assert "user:" in prompt
        assert "Assistant:" in prompt
    
    def test_without_chat_template(self, sample_dataframe):
        """Test processing without chat template"""
        processor = DataProcessor(
            prompt_template="Q: {question}\nA:",
            use_chat_template=False
        )
        items = processor.process_batch(sample_dataframe.head(1))
        
        expected = f"Q: {sample_dataframe.iloc[0]['question']}\nA:"
        assert items[0]["prompt"] == expected
    
    def test_set_tokenizer(self, sample_dataframe):
        """Test setting tokenizer after initialization"""
        processor = DataProcessor(
            use_chat_template=True
        )
        
        # Initially no tokenizer
        assert processor.tokenizer is None
        
        # Set tokenizer
        class MockTokenizer:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                return "Mocked chat template output"
        
        processor.set_tokenizer(MockTokenizer())
        assert processor.tokenizer is not None
        
        items = processor.process_batch(sample_dataframe.head(1))
        assert items[0]["prompt"] == "Mocked chat template output"
    
    def test_few_shot_examples(self, sample_dataframe, temp_dir):
        """Test few-shot examples with chat template"""
        # Create few-shot examples file
        examples = [
            {"question": "What is 2+2?", "answer": "4"},
            {"question": "What color is the sky?", "answer": "Blue"}
        ]
        examples_file = temp_dir / "examples.json"
        with open(examples_file, 'w') as f:
            json.dump(examples, f)
        
        # Create mock tokenizer
        class MockTokenizer:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                formatted = []
                for msg in messages:
                    formatted.append(f"{msg['role']}: {msg['content']}")
                result = "\n".join(formatted)
                if add_generation_prompt:
                    result += "\nAssistant:"
                return result
        
        # Load and use examples
        few_shot = DataProcessor.load_few_shot_examples(str(examples_file))
        processor = DataProcessor(
            few_shot_examples=few_shot,
            use_chat_template=True,
            tokenizer=MockTokenizer()
        )
        
        items = processor.process_batch(sample_dataframe.head(1))
        prompt = items[0]["prompt"]
        
        # Check few-shot examples are included
        assert "user: What is 2+2?" in prompt
        assert "assistant: 4" in prompt
        assert "user: What color is the sky?" in prompt
        assert "assistant: Blue" in prompt
    
    def test_special_tokens(self, sample_dataframe):
        """Test adding special tokens without chat template"""
        processor = DataProcessor(
            add_bos_token=True,
            add_eos_token=True,
            use_chat_template=False
        )
        items = processor.process_batch(sample_dataframe.head(1))
        
        prompt = items[0]["prompt"]
        assert prompt.startswith("<s>")
        assert prompt.endswith("</s>")
    
    def test_special_tokens_with_chat_template(self, sample_dataframe):
        """Test that special tokens are not added when using chat template"""
        class MockTokenizer:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                return "Chat template output"
        
        processor = DataProcessor(
            add_bos_token=True,
            add_eos_token=True,
            use_chat_template=True,
            tokenizer=MockTokenizer()
        )
        items = processor.process_batch(sample_dataframe.head(1))
        
        # Special tokens should not be added when using chat template
        prompt = items[0]["prompt"]
        assert prompt == "Chat template output"
        assert not prompt.startswith("<s>")
        assert not prompt.endswith("</s>")
    
    def test_metadata_columns(self, sample_dataframe):
        """Test including metadata columns"""
        processor = DataProcessor()
        items = processor.process_batch(
            sample_dataframe,
            metadata_columns=["category", "difficulty"]
        )
        
        assert "category" in items[0]
        assert "difficulty" in items[0]
        assert items[0]["category"] == "Geography"
        assert items[0]["difficulty"] == "Easy"
    
    def test_create_repeat_prompts(self):
        """Test creating prompts for repeat generation"""
        processor = DataProcessor()
        question = "What is AI?"
        
        # Independent strategy
        prompts = processor.create_repeat_prompts(
            question, 
            num_repeats=3,
            repeat_strategy="independent"
        )
        assert len(prompts) == 3
        assert all(p["prompt"] == question for p in prompts)
        
        # Temperature schedule strategy
        prompts = processor.create_repeat_prompts(
            question,
            num_repeats=3,
            repeat_strategy="temperature_schedule",
            temperature_schedule=[0.5, 1.0, 1.5]
        )
        assert len(prompts) == 3
        assert prompts[0]["temperature"] == 0.5
        assert prompts[1]["temperature"] == 1.0
        assert prompts[2]["temperature"] == 1.5
        
        # Diverse strategy
        prompts = processor.create_repeat_prompts(
            question,
            num_repeats=3,
            repeat_strategy="diverse"
        )
        assert len(prompts) == 3
        assert all("Variation" in p["prompt"] for p in prompts)
    
    def test_preprocessing_function(self, sample_dataframe, temp_dir):
        """Test custom preprocessing function"""
        # Create preprocessing function file
        preprocess_file = temp_dir / "preprocess.py"
        with open(preprocess_file, 'w') as f:
            f.write("""
def preprocess(question, row):
    return f"[{row['category']}] {question}"
""")
        
        processor = DataProcessor(preprocessing_fn=str(preprocess_file))
        items = processor.process_batch(sample_dataframe.head(1))
        
        assert items[0]["prompt"] == "[Geography] What is the capital of France?"
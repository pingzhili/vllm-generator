import logging
from typing import Optional, List, Dict, Any, Callable
import pandas as pd
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class DataProcessor:
    """Handles data preprocessing and prompt formatting"""
    
    def __init__(
        self,
        prompt_template: Optional[str] = None,
        system_prompt: Optional[str] = None,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
        use_chat_template: bool = False,
        add_bos_token: bool = False,
        add_eos_token: bool = False,
        preprocessing_fn: Optional[Callable] = None,
        tokenizer: Optional[Any] = None
    ):
        self.prompt_template = prompt_template or "{question}"
        self.system_prompt = system_prompt
        self.few_shot_examples = few_shot_examples
        self.use_chat_template = use_chat_template
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.preprocessing_fn = preprocessing_fn
        self.tokenizer = tokenizer
        
        # Load custom preprocessing function if provided
        if isinstance(preprocessing_fn, str):
            self.preprocessing_fn = self._load_preprocessing_fn(preprocessing_fn)
    
    def process_batch(
        self,
        df: pd.DataFrame,
        question_column: str = "question",
        metadata_columns: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Process a batch of questions into model inputs"""
        processed_items = []
        
        for idx, row in df.iterrows():
            question = row[question_column]
            
            # Apply custom preprocessing if provided
            if self.preprocessing_fn:
                question = self.preprocessing_fn(question, row)
            
            # Format prompt
            prompt = self._format_prompt(question)
            
            # Prepare item
            item = {
                "idx": idx,
                "prompt": prompt,
                "original_question": row[question_column]
            }
            
            # Add metadata columns if requested
            if metadata_columns:
                for col in metadata_columns:
                    if col in row:
                        item[col] = row[col]
            
            processed_items.append(item)
        
        return processed_items
    
    def _format_prompt(self, question: str) -> str:
        """Format a single question into a prompt"""
        # Apply template
        prompt = self.prompt_template.format(question=question)
        
        # Use tokenizer's chat template if requested
        if self.use_chat_template and self.tokenizer:
            messages = self._build_messages(prompt)
            prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        
        # Add special tokens (only if not using chat template)
        elif not self.use_chat_template:
            if self.add_bos_token:
                prompt = "<s>" + prompt
            if self.add_eos_token:
                prompt = prompt + "</s>"
        
        return prompt
    
    def _build_messages(self, prompt: str) -> List[Dict[str, str]]:
        """Build messages list for chat template"""
        messages = []
        
        # Add system prompt if provided
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        # Add few-shot examples if provided
        if self.few_shot_examples:
            for example in self.few_shot_examples:
                messages.append({"role": "user", "content": example["question"]})
                messages.append({"role": "assistant", "content": example["answer"]})
        
        # Add the actual question
        messages.append({"role": "user", "content": prompt})
        
        return messages
    
    def _load_preprocessing_fn(self, path: str) -> Callable:
        """Load custom preprocessing function from file"""
        import importlib.util
        
        spec = importlib.util.spec_from_file_location("preprocessing", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if not hasattr(module, "preprocess"):
            raise ValueError(f"Preprocessing file {path} must contain a 'preprocess' function")
        
        return module.preprocess
    
    @staticmethod
    def load_few_shot_examples(path: str) -> List[Dict[str, str]]:
        """Load few-shot examples from JSON file"""
        with open(path, 'r') as f:
            examples = json.load(f)
        
        # Validate format
        for example in examples:
            if "question" not in example or "answer" not in example:
                raise ValueError("Few-shot examples must have 'question' and 'answer' fields")
        
        return examples
    
    def create_repeat_prompts(
        self,
        question: str,
        num_repeats: int,
        repeat_strategy: str = "independent",
        temperature_schedule: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """Create multiple prompts for repeat generation"""
        prompts = []
        
        if repeat_strategy == "independent":
            # Same prompt repeated
            base_prompt = self._format_prompt(question)
            for i in range(num_repeats):
                prompts.append({
                    "prompt": base_prompt,
                    "repeat_id": i,
                    "temperature": None  # Use default
                })
        
        elif repeat_strategy == "temperature_schedule":
            # Different temperatures
            if not temperature_schedule or len(temperature_schedule) != num_repeats:
                raise ValueError("Temperature schedule must match num_repeats")
            
            base_prompt = self._format_prompt(question)
            for i, temp in enumerate(temperature_schedule):
                prompts.append({
                    "prompt": base_prompt,
                    "repeat_id": i,
                    "temperature": temp
                })
        
        elif repeat_strategy == "diverse":
            # Add variation to prompts
            for i in range(num_repeats):
                # Add a variation prefix
                variation_prefix = f"(Variation {i+1}) "
                varied_question = variation_prefix + question
                prompt = self._format_prompt(varied_question)
                prompts.append({
                    "prompt": prompt,
                    "repeat_id": i,
                    "temperature": None
                })
        
        else:
            raise ValueError(f"Unknown repeat strategy: {repeat_strategy}")
        
        return prompts
    
    def set_tokenizer(self, tokenizer: Any) -> None:
        """Set tokenizer for chat template formatting"""
        self.tokenizer = tokenizer
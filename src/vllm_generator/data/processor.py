"""Data processor for handling input/output transformations."""

from typing import List, Dict, Any, Optional, Union
import pandas as pd

from ..utils import get_logger


class DataProcessor:
    """Process and transform data for generation."""
    
    def __init__(self):
        """Initialize data processor."""
        self.logger = get_logger("DataProcessor")
    
    def prepare_prompts(
        self,
        texts: List[str],
        prompt_template: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> List[str]:
        """Prepare prompts from raw texts."""
        if not prompt_template:
            return texts
        
        prompts = []
        for text in texts:
            prompt = prompt_template.format(text=text)
            if system_prompt:
                prompt = f"{system_prompt}\n\n{prompt}"
            prompts.append(prompt)
        
        return prompts
    
    def process_responses(
        self,
        responses: List[Dict[str, Any]],
        extract_field: str = "text"
    ) -> List[str]:
        """Process raw responses from vLLM."""
        processed = []
        
        for response in responses:
            if isinstance(response, dict):
                # Extract text field
                text = response.get(extract_field, "")
                
                # Handle nested response structure
                if "choices" in response and isinstance(response["choices"], list):
                    if response["choices"]:
                        text = response["choices"][0].get("text", "")
                
                processed.append(text)
            elif isinstance(response, str):
                processed.append(response)
            else:
                self.logger.warning(f"Unexpected response type: {type(response)}")
                processed.append("")
        
        return processed
    
    def group_responses_by_input(
        self,
        responses: List[str],
        num_samples: int
    ) -> List[List[str]]:
        """Group responses by input when multiple samples per input."""
        if num_samples == 1:
            return [[r] for r in responses]
        
        grouped = []
        for i in range(0, len(responses), num_samples):
            group = responses[i:i + num_samples]
            grouped.append(group)
        
        return grouped
    
    def flatten_responses(
        self,
        grouped_responses: List[List[str]]
    ) -> List[str]:
        """Flatten grouped responses into single list."""
        flattened = []
        for group in grouped_responses:
            flattened.extend(group)
        return flattened
    
    def create_response_dataframe(
        self,
        input_df: pd.DataFrame,
        responses: Union[List[str], List[List[str]]],
        input_column: str,
        output_column: str,
        copy_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Create output dataframe with responses."""
        # Handle single responses per input
        if responses and isinstance(responses[0], str):
            output_df = input_df.copy()
            output_df[output_column] = responses
            return output_df
        
        # Handle multiple responses per input
        rows = []
        for idx, response_list in enumerate(responses):
            for sample_idx, response in enumerate(response_list):
                row = input_df.iloc[idx].to_dict()
                row[output_column] = response
                row["sample_idx"] = sample_idx
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def validate_responses(
        self,
        responses: List[Any],
        expected_count: int
    ) -> bool:
        """Validate response count matches expected."""
        if len(responses) != expected_count:
            self.logger.error(
                f"Response count mismatch: got {len(responses)}, expected {expected_count}"
            )
            return False
        return True
    
    def clean_text(self, text: str) -> str:
        """Clean text by removing extra whitespace."""
        return " ".join(text.split())
    
    def truncate_responses(
        self,
        responses: List[str],
        max_length: Optional[int] = None
    ) -> List[str]:
        """Truncate responses to maximum length."""
        if not max_length:
            return responses
        
        truncated = []
        for response in responses:
            if len(response) > max_length:
                self.logger.debug(f"Truncating response from {len(response)} to {max_length}")
                response = response[:max_length]
            truncated.append(response)
        
        return truncated
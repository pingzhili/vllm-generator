import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import pandas as pd
import json
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class DataWriter:
    """Handles writing generation results back to parquet files"""
    
    def __init__(
        self,
        output_format: str = "wide",
        output_column_prefix: str = "response",
        save_metadata: bool = False,
        compress: bool = True
    ):
        self.output_format = output_format
        self.output_column_prefix = output_column_prefix
        self.save_metadata = save_metadata
        self.compress = compress
        self.compression = 'snappy' if compress else None
    
    def write_results(
        self,
        original_df: pd.DataFrame,
        results: List[Dict[str, Any]],
        output_path: str,
        num_repeats: int = 1,
        metadata: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Write generation results to parquet file"""
        logger.info(f"Writing results to {output_path}")
        
        # Create output dataframe based on format
        if self.output_format == "wide":
            output_df = self._create_wide_format(original_df, results, num_repeats)
        elif self.output_format == "long":
            output_df = self._create_long_format(original_df, results, num_repeats)
        elif self.output_format == "nested":
            output_df = self._create_nested_format(original_df, results, num_repeats)
        else:
            raise ValueError(f"Unknown output format: {self.output_format}")
        
        # Add metadata if requested
        if self.save_metadata and metadata:
            output_df.attrs = metadata
        
        # Write to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_parquet(output_path, compression=self.compression)
        
        logger.info(f"Wrote {len(output_df)} rows to {output_path}")
        return output_df
    
    def _create_wide_format(
        self,
        original_df: pd.DataFrame,
        results: List[Dict[str, Any]],
        num_repeats: int
    ) -> pd.DataFrame:
        """Create wide format output (one column per repeat)"""
        output_df = original_df.copy()
        
        # Initialize columns
        for i in range(num_repeats):
            col_name = f"{self.output_column_prefix}_{i}"
            output_df[col_name] = None
            
            # Add metadata columns if tracking
            if self.save_metadata:
                output_df[f"{col_name}_tokens"] = None
                output_df[f"{col_name}_latency"] = None
        
        # Fill in results
        for result in results:
            idx = result["idx"]
            repeat_id = result.get("repeat_id", 0)
            col_name = f"{self.output_column_prefix}_{repeat_id}"
            
            output_df.at[idx, col_name] = result["response"]
            
            if self.save_metadata:
                if "tokens" in result:
                    output_df.at[idx, f"{col_name}_tokens"] = result["tokens"]
                if "latency" in result:
                    output_df.at[idx, f"{col_name}_latency"] = result["latency"]
        
        return output_df
    
    def _create_long_format(
        self,
        original_df: pd.DataFrame,
        results: List[Dict[str, Any]],
        num_repeats: int
    ) -> pd.DataFrame:
        """Create long format output (multiple rows per original question)"""
        rows = []
        
        for idx in original_df.index:
            for repeat_id in range(num_repeats):
                row = original_df.loc[idx].to_dict()
                row["repeat_id"] = repeat_id
                row[self.output_column_prefix] = None
                
                if self.save_metadata:
                    row["tokens"] = None
                    row["latency"] = None
                
                rows.append(row)
        
        output_df = pd.DataFrame(rows)
        
        # Fill in results
        for result in results:
            idx = result["idx"]
            repeat_id = result.get("repeat_id", 0)
            
            # Find the corresponding row
            mask = (output_df.index == idx * num_repeats + repeat_id)
            output_df.loc[mask, self.output_column_prefix] = result["response"]
            
            if self.save_metadata:
                if "tokens" in result:
                    output_df.loc[mask, "tokens"] = result["tokens"]
                if "latency" in result:
                    output_df.loc[mask, "latency"] = result["latency"]
        
        return output_df
    
    def _create_nested_format(
        self,
        original_df: pd.DataFrame,
        results: List[Dict[str, Any]],
        num_repeats: int
    ) -> pd.DataFrame:
        """Create nested format (list of responses in single column)"""
        output_df = original_df.copy()
        
        # Initialize with empty lists
        output_df[self.output_column_prefix] = [[] for _ in range(len(output_df))]
        
        if self.save_metadata:
            output_df[f"{self.output_column_prefix}_metadata"] = [[] for _ in range(len(output_df))]
        
        # Group results by index
        for result in results:
            idx = result["idx"]
            response = result["response"]
            
            output_df.at[idx, self.output_column_prefix].append(response)
            
            if self.save_metadata:
                metadata = {
                    "repeat_id": result.get("repeat_id", 0),
                    "tokens": result.get("tokens"),
                    "latency": result.get("latency")
                }
                output_df.at[idx, f"{self.output_column_prefix}_metadata"].append(metadata)
        
        return output_df
    
    def write_metadata(
        self,
        metadata: Dict[str, Any],
        output_path: str
    ):
        """Write generation metadata to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add timestamp
        metadata["timestamp"] = datetime.now().isoformat()
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Wrote metadata to {output_path}")
    
    def write_checkpoint(
        self,
        checkpoint_data: Dict[str, Any],
        checkpoint_path: str
    ):
        """Write checkpoint for resuming interrupted generation"""
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add checkpoint metadata
        checkpoint_data["checkpoint_time"] = datetime.now().isoformat()
        checkpoint_data["checkpoint_version"] = "1.0"
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
        
        logger.info(f"Wrote checkpoint to {checkpoint_path}")
    
    @staticmethod
    def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
        """Load checkpoint data"""
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint_data
    
    def merge_results(
        self,
        result_files: List[str],
        output_path: str,
        strategy: str = "sequential"
    ) -> pd.DataFrame:
        """Merge multiple result files (e.g., from parallel workers)"""
        logger.info(f"Merging {len(result_files)} result files")
        
        dfs = []
        for file in result_files:
            df = pd.read_parquet(file)
            dfs.append(df)
        
        if strategy == "sequential":
            # Simple concatenation
            merged_df = pd.concat(dfs, ignore_index=True)
        elif strategy == "sorted":
            # Sort by original index
            merged_df = pd.concat(dfs).sort_index()
        elif strategy == "as_completed":
            # Keep order as completed (already in dfs order)
            merged_df = pd.concat(dfs, ignore_index=True)
        else:
            raise ValueError(f"Unknown merge strategy: {strategy}")
        
        # Write merged results
        merged_df.to_parquet(output_path, compression=self.compression)
        logger.info(f"Merged {len(merged_df)} rows to {output_path}")
        
        return merged_df
    
    def aggregate_responses(
        self,
        df: pd.DataFrame,
        method: str = "first",
        response_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Aggregate multiple responses into single response"""
        if response_columns is None:
            # Find all response columns
            response_columns = [col for col in df.columns if col.startswith(self.output_column_prefix)]
        
        if method == "first":
            # Take first non-null response
            df[f"{self.output_column_prefix}_aggregated"] = df[response_columns].bfill(axis=1).iloc[:, 0]
        
        elif method == "longest":
            # Take longest response
            def get_longest(row):
                responses = [row[col] for col in response_columns if pd.notna(row[col])]
                return max(responses, key=len) if responses else None
            
            df[f"{self.output_column_prefix}_aggregated"] = df.apply(get_longest, axis=1)
        
        elif method == "majority_vote":
            # Most common response
            def get_majority(row):
                responses = [row[col] for col in response_columns if pd.notna(row[col])]
                if not responses:
                    return None
                # Simple majority vote
                from collections import Counter
                counter = Counter(responses)
                return counter.most_common(1)[0][0]
            
            df[f"{self.output_column_prefix}_aggregated"] = df.apply(get_majority, axis=1)
        
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        
        return df
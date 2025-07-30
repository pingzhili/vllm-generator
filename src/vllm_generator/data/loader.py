import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Iterator, Tuple
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and sharding of parquet dataframes"""
    
    def __init__(
        self,
        input_path: str,
        question_column: str = "question",
        chunk_size: Optional[int] = None,
        validate_columns: bool = True
    ):
        self.input_path = Path(input_path)
        self.question_column = question_column
        self.chunk_size = chunk_size
        self.validate_columns = validate_columns
        self._total_rows = None
        
    def load(self) -> pd.DataFrame:
        """Load the entire parquet file"""
        logger.info(f"Loading data from {self.input_path}")
        df = pd.read_parquet(self.input_path)
        
        if self.validate_columns:
            self._validate_dataframe(df)
            
        self._total_rows = len(df)
        logger.info(f"Loaded {self._total_rows} rows")
        return df
    
    def load_chunks(self) -> Iterator[pd.DataFrame]:
        """Load parquet file in chunks"""
        logger.info(f"Loading data in chunks from {self.input_path}")
        
        # Get total rows for progress tracking
        if self._total_rows is None:
            df_head = pd.read_parquet(self.input_path, columns=[self.question_column])
            self._total_rows = len(df_head)
        
        # Read in chunks
        chunk_iter = pd.read_parquet(
            self.input_path,
            engine='pyarrow',
            chunksize=self.chunk_size
        )
        
        for chunk in chunk_iter:
            if self.validate_columns:
                self._validate_dataframe(chunk)
            yield chunk
    
    def create_shards(
        self,
        num_shards: int,
        output_dir: Optional[str] = None,
        strategy: str = "contiguous"
    ) -> List[Dict[str, Any]]:
        """Split the dataset into shards for parallel processing"""
        logger.info(f"Creating {num_shards} shards using {strategy} strategy")
        
        df = self.load()
        total_rows = len(df)
        
        if strategy == "contiguous":
            # Simple contiguous splitting
            shard_size = total_rows // num_shards
            remainder = total_rows % num_shards
            
            shards = []
            start_idx = 0
            
            for i in range(num_shards):
                # Distribute remainder across first shards
                current_shard_size = shard_size + (1 if i < remainder else 0)
                end_idx = start_idx + current_shard_size
                
                shard_info = {
                    "shard_id": i,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "num_rows": current_shard_size
                }
                
                if output_dir:
                    # Save shard to disk
                    shard_path = Path(output_dir) / f"shard_{i}.parquet"
                    shard_path.parent.mkdir(parents=True, exist_ok=True)
                    df.iloc[start_idx:end_idx].to_parquet(shard_path)
                    shard_info["path"] = str(shard_path)
                else:
                    # Store in memory
                    shard_info["data"] = df.iloc[start_idx:end_idx]
                
                shards.append(shard_info)
                start_idx = end_idx
                
        elif strategy == "round_robin":
            # Round-robin distribution
            shards = [{"shard_id": i, "indices": []} for i in range(num_shards)]
            
            for idx in range(total_rows):
                shard_idx = idx % num_shards
                shards[shard_idx]["indices"].append(idx)
            
            for shard in shards:
                indices = shard["indices"]
                shard["num_rows"] = len(indices)
                
                if output_dir:
                    shard_path = Path(output_dir) / f"shard_{shard['shard_id']}.parquet"
                    shard_path.parent.mkdir(parents=True, exist_ok=True)
                    df.iloc[indices].to_parquet(shard_path)
                    shard["path"] = str(shard_path)
                else:
                    shard["data"] = df.iloc[indices]
                    
        elif strategy == "hash":
            # Hash-based sharding (requires shard_column)
            raise NotImplementedError("Hash-based sharding not yet implemented")
        
        elif strategy == "balanced":
            # Balance by estimated compute cost (token count)
            logger.info("Estimating compute cost per row for balanced sharding")
            
            # Estimate tokens (rough approximation)
            df["_estimated_tokens"] = df[self.question_column].str.len() / 4
            
            # Sort by estimated cost
            df_sorted = df.sort_values("_estimated_tokens", ascending=False)
            
            # Distribute greedily
            shard_costs = [0] * num_shards
            shard_indices = [[] for _ in range(num_shards)]
            
            for idx, row in df_sorted.iterrows():
                # Assign to shard with lowest current cost
                min_shard = np.argmin(shard_costs)
                shard_indices[min_shard].append(idx)
                shard_costs[min_shard] += row["_estimated_tokens"]
            
            # Create shards
            shards = []
            for i, indices in enumerate(shard_indices):
                shard_info = {
                    "shard_id": i,
                    "num_rows": len(indices),
                    "estimated_cost": shard_costs[i]
                }
                
                if output_dir:
                    shard_path = Path(output_dir) / f"shard_{i}.parquet"
                    shard_path.parent.mkdir(parents=True, exist_ok=True)
                    df.loc[indices].drop(columns=["_estimated_tokens"]).to_parquet(shard_path)
                    shard_info["path"] = str(shard_path)
                else:
                    shard_info["data"] = df.loc[indices].drop(columns=["_estimated_tokens"])
                
                shards.append(shard_info)
            
            # Clean up
            df.drop(columns=["_estimated_tokens"], inplace=True)
        
        else:
            raise ValueError(f"Unknown sharding strategy: {strategy}")
        
        logger.info(f"Created {len(shards)} shards")
        return shards
    
    def load_subset(
        self,
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
        max_samples: Optional[int] = None,
        sample_fraction: Optional[float] = None,
        filter_column: Optional[str] = None,
        filter_value: Optional[Any] = None
    ) -> pd.DataFrame:
        """Load a subset of the data with various filtering options"""
        df = self.load()
        
        # Apply filtering
        if filter_column and filter_value is not None:
            df = df[df[filter_column] == filter_value]
            logger.info(f"Filtered to {len(df)} rows where {filter_column}={filter_value}")
        
        # Apply sampling
        if sample_fraction is not None:
            df = df.sample(frac=sample_fraction, random_state=42)
            logger.info(f"Sampled {len(df)} rows ({sample_fraction*100:.1f}%)")
        
        # Apply index slicing
        if start_index is not None or end_index is not None:
            start = start_index or 0
            end = end_index or len(df)
            df = df.iloc[start:end]
            logger.info(f"Sliced to rows {start}:{end}")
        
        # Apply max samples
        if max_samples is not None and len(df) > max_samples:
            df = df.head(max_samples)
            logger.info(f"Limited to {max_samples} samples")
        
        return df
    
    def _validate_dataframe(self, df: pd.DataFrame):
        """Validate that required columns exist"""
        if self.question_column not in df.columns:
            raise ValueError(
                f"Question column '{self.question_column}' not found in dataframe. "
                f"Available columns: {list(df.columns)}"
            )
    
    @property
    def total_rows(self) -> Optional[int]:
        """Get total number of rows (if loaded)"""
        return self._total_rows
    
    def estimate_tokens(self, df: pd.DataFrame) -> pd.Series:
        """Estimate token count for each question"""
        # Simple heuristic: ~4 characters per token
        return df[self.question_column].str.len() / 4
    
    @staticmethod
    def merge_shards(shard_paths: List[str], output_path: str):
        """Merge multiple shard files into a single parquet file"""
        logger.info(f"Merging {len(shard_paths)} shards into {output_path}")
        
        dfs = []
        for path in shard_paths:
            dfs.append(pd.read_parquet(path))
        
        merged_df = pd.concat(dfs, ignore_index=True)
        merged_df.to_parquet(output_path)
        logger.info(f"Merged {len(merged_df)} rows into {output_path}")
        
        return merged_df
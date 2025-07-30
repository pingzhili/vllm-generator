#!/usr/bin/env python3
"""
Merge split parquet files back into a single file.

Usage:
    python merge_splits.py output.parquet 4
    python merge_splits.py /path/to/results.parquet 8
    python merge_splits.py --help
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
from typing import List


def find_split_files(base_path: Path, num_splits: int) -> List[Path]:
    """Find all split files for a given base path."""
    parent_dir = base_path.parent
    stem = base_path.stem
    suffix = base_path.suffix
    
    split_files = []
    missing_files = []
    
    for i in range(1, num_splits + 1):
        split_name = f"{stem}-{i}-of-{num_splits}{suffix}"
        split_path = parent_dir / split_name
        
        if split_path.exists():
            split_files.append(split_path)
        else:
            missing_files.append(split_path)
    
    if missing_files:
        print(f"Warning: Missing {len(missing_files)} split files:")
        for f in missing_files:
            print(f"  - {f}")
        print()
    
    if not split_files:
        raise FileNotFoundError(f"No split files found for {base_path} with {num_splits} splits")
    
    return sorted(split_files)


def merge_parquet_files(split_files: List[Path], output_path: Path) -> None:
    """Merge multiple parquet files into one."""
    print(f"Merging {len(split_files)} split files...")
    
    dataframes = []
    total_rows = 0
    
    for split_file in split_files:
        print(f"  Reading {split_file.name}...")
        df = pd.read_parquet(split_file)
        dataframes.append(df)
        total_rows += len(df)
        print(f"    {len(df)} rows")
    
    print(f"\nCombining {total_rows} total rows...")
    merged_df = pd.concat(dataframes, ignore_index=True)
    
    print(f"Saving to {output_path}...")
    merged_df.to_parquet(output_path, index=False)
    
    print(f"✓ Successfully merged {len(split_files)} files into {output_path}")
    print(f"✓ Total rows: {len(merged_df)}")
    
    # Show sample statistics if multiple samples per input
    if 'sample_idx' in merged_df.columns:
        samples_per_input = merged_df['sample_idx'].max() + 1
        unique_inputs = len(merged_df['sample_idx'].value_counts())
        print(f"✓ {unique_inputs} unique inputs with {samples_per_input} samples each")


def main():
    parser = argparse.ArgumentParser(
        description="Merge split parquet files back into a single file",
        epilog="""
Examples:
  # Merge 4 splits of results.parquet
  python merge_splits.py results.parquet 4
  
  # Merge 8 splits with full path
  python merge_splits.py /path/to/output.parquet 8
  
  # This will look for files like:
  # results-1-of-4.parquet, results-2-of-4.parquet, etc.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "output_path",
        type=Path,
        help="Base output path (e.g., results.parquet)"
    )
    
    parser.add_argument(
        "num_splits",
        type=int,
        help="Number of splits to merge"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be merged without actually doing it"
    )
    
    args = parser.parse_args()
    
    try:
        # Find split files
        split_files = find_split_files(args.output_path, args.num_splits)
        
        print(f"Found {len(split_files)} split files:")
        for f in split_files:
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"  - {f.name} ({size_mb:.1f} MB)")
        
        if args.dry_run:
            print(f"\nDry run: Would merge into {args.output_path}")
            return
        
        # Check if output file already exists
        if args.output_path.exists():
            response = input(f"\n{args.output_path} already exists. Overwrite? (y/N): ")
            if response.lower() not in ['y', 'yes']:
                print("Aborted.")
                return
        
        # Merge files
        merge_parquet_files(split_files, args.output_path)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
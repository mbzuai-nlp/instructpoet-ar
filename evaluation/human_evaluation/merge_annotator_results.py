#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to merge multiple annotator results into a single file.

This script helps when you have multiple annotators evaluating the same samples.
It combines their evaluations and can calculate agreement metrics.

Author: Merge Results Script
Date: December 29, 2025
"""

import pandas as pd
from pathlib import Path
import argparse
from typing import List


def merge_annotator_results(
    annotator_files: List[str],
    output_path: Path,
    annotator_names: List[str] = None
) -> pd.DataFrame:
    """
    Merge results from multiple annotators.
    
    Args:
        annotator_files: List of paths to annotator Excel/CSV files
        output_path: Path to save merged results
        annotator_names: Names for annotators (optional)
    
    Returns:
        Merged dataframe
    """
    print("Merging annotator results...")
    
    dfs = []
    names = annotator_names or [f"annotator_{i+1}" for i in range(len(annotator_files))]
    
    for i, file_path in enumerate(annotator_files):
        file_path = Path(file_path)
        
        if file_path.suffix == '.xlsx':
            df = pd.read_excel(file_path)
        elif file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # Add annotator name
        df['annotator'] = names[i]
        
        print(f"  ✓ Loaded {len(df)} rows from {file_path.name} ({names[i]})")
        dfs.append(df)
    
    # Combine all dataframes
    merged_df = pd.concat(dfs, axis=0, ignore_index=True)
    print(f"  ✓ Combined into {len(merged_df)} rows")
    
    # Save merged results
    if output_path.suffix == '.xlsx':
        merged_df.to_excel(output_path, index=False)
    else:
        merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"  ✓ Saved merged results to {output_path}")
    
    return merged_df


def create_annotator_summary(merged_df: pd.DataFrame, output_dir: Path):
    """
    Create a summary of annotator responses.
    
    Args:
        merged_df: Merged dataframe from multiple annotators
        output_dir: Output directory
    """
    print("\nCreating annotator summary...")
    
    summary_path = output_dir / "annotator_summary.txt"
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("Annotator Summary\n")
        f.write("=" * 60 + "\n\n")
        
        # Count by annotator
        annotator_counts = merged_df['annotator'].value_counts()
        f.write("Evaluations per Annotator:\n")
        for annotator, count in annotator_counts.items():
            f.write(f"  {annotator}: {count} evaluations\n")
        f.write("\n")
        
        # Coverage by sample
        sample_coverage = merged_df.groupby('sample_id')['annotator'].nunique()
        f.write(f"Sample Coverage:\n")
        f.write(f"  Total unique samples: {len(sample_coverage)}\n")
        f.write(f"  Samples with 1 annotator: {(sample_coverage == 1).sum()}\n")
        f.write(f"  Samples with 2 annotators: {(sample_coverage == 2).sum()}\n")
        f.write(f"  Samples with 3+ annotators: {(sample_coverage >= 3).sum()}\n")
        f.write("\n")
        
        # Completion status
        score_cols = [
            'الالتزام_بالقيود',
            'الطلاقة_اللغوية_والإيقاع',
            'التماسك_والمعنى',
            'الشاعرية_والجمال_الفني'
        ]
        
        f.write("Completion Status:\n")
        for col in score_cols:
            completed = merged_df[col].notna().sum()
            total = len(merged_df)
            pct = (completed / total) * 100
            f.write(f"  {col}: {completed}/{total} ({pct:.1f}%)\n")
    
    print(f"  ✓ Saved summary to {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple annotator evaluation results"
    )
    parser.add_argument(
        "--files",
        type=str,
        nargs='+',
        required=True,
        help="Paths to annotator evaluation files (Excel or CSV)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save merged results"
    )
    parser.add_argument(
        "--annotator_names",
        type=str,
        nargs='+',
        default=None,
        help="Names for annotators (optional)"
    )
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Merging Annotator Results")
    print("=" * 60)
    print(f"Input files: {len(args.files)}")
    print(f"Output file: {output_path}")
    print("=" * 60)
    
    # Merge results
    merged_df = merge_annotator_results(
        args.files,
        output_path,
        args.annotator_names
    )
    
    # Create summary
    create_annotator_summary(merged_df, output_dir)
    
    print("\n" + "=" * 60)
    print("✅ Merge Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to analyze human evaluation results for Arabic poetry models.

This script:
1. Loads completed evaluation sheets
2. Merges with model mapping
3. Calculates aggregate scores per model
4. Computes inter-annotator agreement (if multiple annotators)
5. Generates statistical analysis and visualizations

Author: Analysis Script
Date: December 29, 2025
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
from scipy import stats


def load_evaluation_results(eval_file: Path) -> pd.DataFrame:
    """Load evaluation results from Excel or CSV file."""
    print(f"Loading evaluation results from {eval_file}...")
    
    if eval_file.suffix == '.xlsx':
        df = pd.read_excel(eval_file)
    elif eval_file.suffix == '.csv':
        df = pd.read_csv(eval_file)
    else:
        raise ValueError(f"Unsupported file format: {eval_file.suffix}")
    
    print(f"  Loaded {len(df)} evaluation entries")
    return df


def load_model_mapping(mapping_file: Path) -> Dict[str, str]:
    """Load model ID to name mapping."""
    print(f"Loading model mapping from {mapping_file}...")
    
    with open(mapping_file, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    
    print(f"  Loaded mapping for {len(mapping)} models")
    return mapping


def calculate_scores(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """Calculate aggregate scores per model."""
    print("\nCalculating aggregate scores...")
    
    # Map model IDs to actual names
    df['model_name'] = df['model_id'].map(mapping)
    
    # Define score columns
    score_cols = [
        'الالتزام_بالقيود',
        'الطلاقة_اللغوية_والإيقاع',
        'التماسك_والمعنى',
        'الشاعرية_والجمال_الفني'
    ]
    
    # Convert to numeric (in case they're strings)
    for col in score_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Calculate average score
    df['average_score'] = df[score_cols].mean(axis=1)
    
    # Aggregate by model
    model_scores = df.groupby('model_name')[score_cols + ['average_score']].agg(['mean', 'std', 'count'])
    
    print("  ✓ Calculated scores per model")
    
    return model_scores


def calculate_inter_annotator_agreement(dfs: List[pd.DataFrame]) -> Dict[str, float]:
    """
    Calculate inter-annotator agreement using Krippendorff's Alpha or ICC.
    
    Args:
        dfs: List of dataframes from different annotators
    
    Returns:
        Dictionary with agreement metrics
    """
    print("\nCalculating inter-annotator agreement...")
    
    if len(dfs) < 2:
        print("  ⚠ Only one annotator - skipping agreement calculation")
        return {}
    
    # Merge dataframes on sample_id and model_id
    merged = dfs[0][['sample_id', 'model_id']].copy()
    
    score_cols = [
        'الالتزام_بالقيود',
        'الطلاقة_اللغوية_والإيقاع',
        'التماسك_والمعنى',
        'الشاعرية_والجمال_الفني'
    ]
    
    agreement_scores = {}
    
    for col in score_cols:
        # Collect scores from all annotators
        annotator_scores = []
        for i, df in enumerate(dfs):
            scores = df.set_index(['sample_id', 'model_id'])[col]
            annotator_scores.append(scores)
        
        # Combine into matrix
        score_matrix = pd.concat(annotator_scores, axis=1)
        score_matrix.columns = [f'annotator_{i+1}' for i in range(len(dfs))]
        
        # Calculate correlation between annotators (simple measure)
        correlations = []
        for i in range(len(dfs)):
            for j in range(i+1, len(dfs)):
                corr = score_matrix.iloc[:, i].corr(score_matrix.iloc[:, j])
                if not np.isnan(corr):
                    correlations.append(corr)
        
        if correlations:
            agreement_scores[col] = np.mean(correlations)
    
    print("  ✓ Calculated inter-annotator agreement")
    
    return agreement_scores


def perform_statistical_tests(df: pd.DataFrame, mapping: Dict[str, str]) -> Dict:
    """
    Perform statistical significance tests between models.
    
    Args:
        df: Evaluation dataframe
        mapping: Model ID mapping
    
    Returns:
        Dictionary with test results
    """
    print("\nPerforming statistical tests...")
    
    df['model_name'] = df['model_id'].map(mapping)
    
    score_cols = [
        'الالتزام_بالقيود',
        'الطلاقة_اللغوية_والإيقاع',
        'التماسك_والمعنى',
        'الشاعرية_والجمال_الفني',
        'average_score'
    ]
    
    # Calculate average score if not present
    if 'average_score' not in df.columns:
        df['average_score'] = df[score_cols[:-1]].mean(axis=1)
    
    results = {}
    
    for col in score_cols:
        # Group scores by model
        model_groups = [group[col].dropna().values for name, group in df.groupby('model_name')]
        
        # Perform ANOVA
        if len(model_groups) > 2:
            f_stat, p_value = stats.f_oneway(*model_groups)
            results[col] = {
                'test': 'ANOVA',
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        elif len(model_groups) == 2:
            t_stat, p_value = stats.ttest_ind(model_groups[0], model_groups[1])
            results[col] = {
                'test': 't-test',
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
    
    print("  ✓ Completed statistical tests")
    
    return results


def generate_report(
    model_scores: pd.DataFrame,
    agreement_scores: Dict[str, float],
    stat_tests: Dict,
    output_path: Path
):
    """Generate a comprehensive analysis report."""
    print(f"\nGenerating report at {output_path}...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Human Evaluation Results - Analysis Report\n")
        f.write("=" * 80 + "\n\n")
        
        # Model Scores
        f.write("1. Model Scores Summary\n")
        f.write("-" * 80 + "\n\n")
        f.write(model_scores.to_string())
        f.write("\n\n")
        
        # Ranking
        f.write("2. Model Rankings (by Average Score)\n")
        f.write("-" * 80 + "\n\n")
        rankings = model_scores['average_score']['mean'].sort_values(ascending=False)
        for rank, (model, score) in enumerate(rankings.items(), 1):
            f.write(f"  {rank}. {model}: {score:.3f}\n")
        f.write("\n")
        
        # Best model per criterion
        f.write("3. Best Model per Criterion\n")
        f.write("-" * 80 + "\n\n")
        criteria = [
            'الالتزام_بالقيود',
            'الطلاقة_اللغوية_والإيقاع',
            'التماسك_والمعنى',
            'الشاعرية_والجمال_الفني'
        ]
        
        for criterion in criteria:
            best_model = model_scores[criterion]['mean'].idxmax()
            best_score = model_scores[criterion]['mean'].max()
            f.write(f"  {criterion}: {best_model} ({best_score:.3f})\n")
        f.write("\n")
        
        # Inter-annotator agreement
        if agreement_scores:
            f.write("4. Inter-Annotator Agreement (Average Correlation)\n")
            f.write("-" * 80 + "\n\n")
            for criterion, score in agreement_scores.items():
                f.write(f"  {criterion}: {score:.3f}\n")
            f.write("\n")
        
        # Statistical tests
        if stat_tests:
            f.write("5. Statistical Significance Tests\n")
            f.write("-" * 80 + "\n\n")
            for criterion, results in stat_tests.items():
                test_type = results['test']
                p_value = results['p_value']
                significant = "✓ Significant" if results['significant'] else "✗ Not Significant"
                f.write(f"  {criterion}:\n")
                f.write(f"    Test: {test_type}\n")
                f.write(f"    p-value: {p_value:.4f}\n")
                f.write(f"    Result: {significant} (α=0.05)\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("End of Report\n")
        f.write("=" * 80 + "\n")
    
    print(f"  ✓ Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze human evaluation results for Arabic poetry models"
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        required=True,
        help="Path to completed evaluation file (Excel or CSV)"
    )
    parser.add_argument(
        "--mapping_file",
        type=str,
        required=True,
        help="Path to model mapping JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Output directory for analysis results"
    )
    parser.add_argument(
        "--additional_files",
        type=str,
        nargs='+',
        default=None,
        help="Additional evaluation files from other annotators (for agreement calculation)"
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    eval_file = Path(args.eval_file)
    mapping_file = Path(args.mapping_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Human Evaluation Results Analysis")
    print("=" * 80)
    print(f"Evaluation file: {eval_file}")
    print(f"Mapping file: {mapping_file}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    # Load data
    df = load_evaluation_results(eval_file)
    mapping = load_model_mapping(mapping_file)
    
    # Load additional files if provided
    dfs = [df]
    if args.additional_files:
        for file_path in args.additional_files:
            additional_df = load_evaluation_results(Path(file_path))
            dfs.append(additional_df)
        print(f"\n✓ Loaded {len(dfs)} evaluation files total")
    
    # Calculate scores
    model_scores = calculate_scores(df, mapping)
    
    # Calculate inter-annotator agreement if multiple annotators
    agreement_scores = calculate_inter_annotator_agreement(dfs) if len(dfs) > 1 else {}
    
    # Perform statistical tests
    stat_tests = perform_statistical_tests(df, mapping)
    
    # Generate report
    task_name = eval_file.stem.split('_')[0]
    report_path = output_dir / f"{task_name}_analysis_report.txt"
    generate_report(model_scores, agreement_scores, stat_tests, report_path)
    
    # Save detailed scores to CSV
    scores_path = output_dir / f"{task_name}_model_scores.csv"
    model_scores.to_csv(scores_path)
    print(f"  ✓ Saved detailed scores to {scores_path}")
    
    # Save results with model names
    df_with_names = df.copy()
    df_with_names['model_name'] = df_with_names['model_id'].map(mapping)
    results_path = output_dir / f"{task_name}_results_with_names.csv"
    df_with_names.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"  ✓ Saved results with model names to {results_path}")
    
    print("\n" + "=" * 80)
    print("✅ Analysis Complete!")
    print("=" * 80)
    print("\nGenerated files:")
    print(f"  1. {report_path.name} - Analysis report")
    print(f"  2. {scores_path.name} - Model scores (CSV)")
    print(f"  3. {results_path.name} - Results with model names")
    print("=" * 80)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Extract evaluation results from dialect continuation experiments and format for Google Sheets.
Models will be on rows, subtasks on columns.
"""

import json
import os
from pathlib import Path
from typing import Dict, List
import csv


def find_latest_results_file(model_dir: Path) -> Path:
    """Find the most recent results JSON file in a model directory."""
    json_files = list(model_dir.glob("results_*.json"))
    if not json_files:
        raise FileNotFoundError(f"No results file found in {model_dir}")
    # Sort by modification time and get the most recent
    return max(json_files, key=lambda p: p.stat().st_mtime)


def extract_results(base_dir: str) -> Dict[str, Dict[str, float]]:
    """
    Extract results from all model directories.
    
    Returns:
        Dict mapping model names to their subtask results.
    """
    base_path = Path(base_dir)
    continuation_dir = base_path / "dialectical_poetry_analysis_continuation"
    
    if not continuation_dir.exists():
        raise FileNotFoundError(f"Directory not found: {continuation_dir}")
    
    results = {}
    
    # Get all model directories
    model_dirs = [d for d in continuation_dir.iterdir() if d.is_dir()]
    
    for model_dir in sorted(model_dirs):
        model_name = model_dir.name
        
        try:
            # Find the latest results file
            results_file = find_latest_results_file(model_dir)
            
            # Load the results
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract subtask results
            model_results = {}
            
            # Get the overall accuracy
            if "dialectical_poetry_analysis_continuation" in data["results"]:
                overall = data["results"]["dialectical_poetry_analysis_continuation"]
                model_results["Overall"] = overall.get("acc,none", 0.0)
            
            # Get individual subtask accuracies
            for subtask_name, subtask_data in data["results"].items():
                if subtask_name != "dialectical_poetry_analysis_continuation":
                    # Clean up the subtask name for better readability
                    clean_name = subtask_name.replace("dialectical_continuation_", "")
                    model_results[clean_name] = subtask_data.get("acc,none", 0.0)
            
            results[model_name] = model_results
            
        except Exception as e:
            print(f"Warning: Could not process {model_name}: {e}")
    
    return results


def format_for_google_sheets(results: Dict[str, Dict[str, float]]) -> str:
    """
    Format results as tab-separated values for easy pasting into Google Sheets.
    
    Args:
        results: Dict mapping model names to their subtask results
        
    Returns:
        Tab-separated string ready for pasting
    """
    if not results:
        return "No results found"
    
    # Get all unique subtasks (columns)
    all_subtasks = set()
    for model_results in results.values():
        all_subtasks.update(model_results.keys())
    
    # Sort subtasks: Overall first, then alphabetically
    subtasks = ["Overall"] + sorted([s for s in all_subtasks if s != "Overall"])
    
    # Create header row
    output_lines = ["Model\t" + "\t".join(subtasks)]
    
    # Create data rows
    for model_name in sorted(results.keys()):
        row = [model_name]
        for subtask in subtasks:
            value = results[model_name].get(subtask, "")
            if isinstance(value, float):
                # Format as percentage with 2 decimal places
                row.append(f"{value*100:.2f}%")
            else:
                row.append(str(value))
        output_lines.append("\t".join(row))
    
    return "\n".join(output_lines)


def save_to_csv(results: Dict[str, Dict[str, float]], output_file: str):
    """
    Save results to CSV file.
    
    Args:
        results: Dict mapping model names to their subtask results
        output_file: Path to output CSV file
    """
    if not results:
        print("No results to save")
        return
    
    # Get all unique subtasks (columns)
    all_subtasks = set()
    for model_results in results.values():
        all_subtasks.update(model_results.keys())
    
    # Sort subtasks: Overall first, then alphabetically
    subtasks = ["Overall"] + sorted([s for s in all_subtasks if s != "Overall"])
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(["Model"] + subtasks)
        
        # Write data rows
        for model_name in sorted(results.keys()):
            row = [model_name]
            for subtask in subtasks:
                value = results[model_name].get(subtask, "")
                if isinstance(value, float):
                    # Keep as decimal for CSV
                    row.append(f"{value:.4f}")
                else:
                    row.append(str(value))
            writer.writerow(row)
    
    print(f"Results saved to: {output_file}")


def main():
    # Base directory containing evaluation results
    base_dir = Path(__file__).resolve().parent / "evaluation_results_dialect_continuation"

    print("Extracting results from dialect continuation evaluation...")
    print(f"Base directory: {base_dir}\n")
    
    # Extract results
    results = extract_results(str(base_dir))
    
    if not results:
        print("No results found!")
        return
    
    print(f"Found results for {len(results)} models\n")
    
    # Format for Google Sheets (tab-separated)
    formatted_output = format_for_google_sheets(results)
    
    # Print to console (ready to copy-paste)
    print("=" * 80)
    print("RESULTS (Copy and paste into Google Sheets):")
    print("=" * 80)
    print(formatted_output)
    print("=" * 80)
    
    # Also save to CSV file
    output_csv = "dialect_continuation_results.csv"
    save_to_csv(results, output_csv)
    
    print(f"\nTotal models: {len(results)}")
    print(f"Total subtasks: {len(results[list(results.keys())[0]]) if results else 0}")


if __name__ == "__main__":
    main()

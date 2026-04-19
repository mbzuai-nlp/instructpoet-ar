#!/usr/bin/env python3
"""
Script to extract statistics from dialectical IFT data and generate LaTeX tables.

This script reads statistics from the dialectical_IFT_DATA/stats directory
and generates the tables shown in the research paper.
"""

import os
import re
from typing import Dict, List, Tuple


class IFTStatsExtractor:
    """Extract and format statistics from IFT dataset files."""

    def __init__(self, stats_dir: str):
        self.stats_dir = stats_dir

        # Mapping from corruption template indices to corruption types
        self.corruption_mapping = {
            "corruption_template_0.0": "full_style",
            "corruption_template_1.0": "rhyme_structure",
            "corruption_template_2.0": "rhyme_content",
            "corruption_template_3.0": "rhyme_substitution",
            "corruption_template_4.0": "era_corruption",
            "corruption_template_5.0": "meter_transformation",
            "corruption_template_6.0": "meter_destruction",
            "corruption_template_7.0": "meter_inconsistency",
        }

    def read_stats_file(self, filepath: str) -> Dict[str, int]:
        """Read a stats file and return template statistics."""
        stats = {}

        if not os.path.exists(filepath):
            print(f"Warning: File not found: {filepath}")
            return stats

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract template statistics section
        lines = content.split("\n")
        in_template_section = False

        for line in lines:
            line = line.strip()

            if line == "Template Statistics":
                in_template_section = True
                continue
            elif line == "Dialect Statistics" or line.startswith("===="):
                if line.startswith("===="):
                    continue
                else:
                    in_template_section = False
                    break

            if in_template_section and line and ":" in line:
                # Parse template line
                parts = line.split(": ")
                if len(parts) == 2:
                    template = parts[0].strip()
                    count = int(parts[1].replace(",", ""))
                    stats[template] = count

        return stats

    def get_task_stats(self, task: str, split: str) -> Dict[str, int]:
        """Get statistics for a specific task and split."""
        if task == "analysis":
            filename = "analysis_stats.txt"
        elif task == "continuation":
            filename = "continuation_stats.txt"
        elif task == "corruption":
            filename = "corruption_stats.txt"
        elif task == "generation":
            filename = "generation_stats.txt"
        else:
            raise ValueError(f"Unknown task: {task}")

        filepath = os.path.join(self.stats_dir, task, split, filename)
        return self.read_stats_file(filepath)

    def format_number(self, num: int) -> str:
        """Format number with comma separator for thousands."""
        return f"{num:,}"

    def generate_overall_stats_table(self) -> str:
        """Generate the overall statistics table."""
        tasks = ["analysis", "continuation", "corruption", "generation"]
        splits = ["train", "test"]

        # Map corruption to restoration in display
        task_display_names = {
            "analysis": "Analysis",
            "continuation": "Continuation",
            "corruption": "Restoration",
            "generation": "Generation",
        }

        results = []

        for task in tasks:
            for split in splits:
                stats = self.get_task_stats(task, split)
                total_samples = sum(stats.values())
                num_subtasks = len(stats)

                # Format split name (capitalize first letter)
                split_display = split.capitalize()

                # Create row
                if split == "train":
                    task_name = task_display_names[task]
                else:
                    task_name = ""  # Empty for test rows

                results.append(
                    f"{task_name:12s} & {split_display:5s} & {self.format_number(total_samples):>9s} & {num_subtasks:2d} \\\\"
                )

        # Create LaTeX table
        latex = """\\begin{table}[t]
\\centering
\\begin{tabular}{lcccc}
\\toprule
\\textbf{Task} & \\textbf{Split} & \\textbf{Total Samples} & \\textbf{\\# Subtasks} \\\\
\\midrule
"""

        # Add rows with proper spacing between tasks
        for i, row in enumerate(results):
            latex += row + "\n"
            # Add spacing after each task (every 2 rows)
            if i % 2 == 1 and i < len(results) - 1:
                latex += ""

        latex += """\\bottomrule
\\end{tabular}
\\caption{Overall statistics for Arabic poetry IFT dataset across tasks and data splits.}
\\label{tab:poetry_ift_overall_stats}
\\end{table}"""

        return latex

    def generate_analysis_subtasks_table(self) -> str:
        """Generate detailed analysis subtasks table."""
        train_stats = self.get_task_stats("analysis", "train")
        test_stats = self.get_task_stats("analysis", "test")

        latex = """\\begin{table*}[t]
\\centering
\\scriptsize
\\setlength{\\tabcolsep}{4pt}
\\begin{tabularx}{\\textwidth}{Xr}
\\toprule
\\textbf{Subtask (Input $\\rightarrow$ Output)} & \\textbf{Samples Count} \\\\
\\midrule
\\multicolumn{2}{c}{\\textit{Train Split}} \\\\
\\midrule
"""

        # Sort train stats by count (descending)
        train_sorted = sorted(train_stats.items(), key=lambda x: x[1], reverse=True)

        for template, count in train_sorted:
            # Clean up template format
            clean_template = self.clean_template_name(template, "analysis")
            latex += f"{clean_template} & {self.format_number(count)} \\\\\n"

        latex += """\\midrule
\\multicolumn{2}{c}{\\textit{Test Split}} \\\\
\\midrule
"""

        # Sort test stats by count (descending)
        test_sorted = sorted(test_stats.items(), key=lambda x: x[1], reverse=True)

        for template, count in test_sorted:
            clean_template = self.clean_template_name(template, "analysis")
            latex += f"{clean_template} & {self.format_number(count)} \\\\\n"

        latex += """\\bottomrule
\\end{tabularx}
\\caption{Detailed statistics per subtask for the \\textit{Analysis} task in the Arabic poetry IFT dataset.}
\\label{tab:ift_analysis_subtasks}
\\end{table*}"""

        return latex

    def generate_continuation_subtasks_table(self) -> str:
        """Generate detailed continuation subtasks table."""
        train_stats = self.get_task_stats("continuation", "train")
        test_stats = self.get_task_stats("continuation", "test")

        latex = """\\begin{table*}[t]
\\centering
\\scriptsize
\\setlength{\\tabcolsep}{4pt}
\\begin{tabularx}{\\textwidth}{Xr}
\\toprule
\\textbf{Subtask (Input $\\rightarrow$ Output)} & \\textbf{Samples Count} \\\\
\\midrule
\\multicolumn{2}{c}{\\textit{Train Split}} \\\\
\\midrule
"""

        train_sorted = sorted(train_stats.items(), key=lambda x: x[1], reverse=True)

        for template, count in train_sorted:
            clean_template = self.clean_template_name(template, "continuation")
            latex += f"{clean_template} & {self.format_number(count)} \\\\\n"

        latex += """\\midrule
\\multicolumn{2}{c}{\\textit{Test Split}} \\\\
\\midrule
"""

        test_sorted = sorted(test_stats.items(), key=lambda x: x[1], reverse=True)

        for template, count in test_sorted:
            clean_template = self.clean_template_name(template, "continuation")
            latex += f"{clean_template} & {self.format_number(count)} \\\\\n"

        latex += """\\bottomrule
\\end{tabularx}
\\caption{Detailed statistics per subtask for the \\textit{Continuation} task in the Arabic poetry IFT dataset.}
\\label{tab:ift_continuation_subtasks}
\\end{table*}"""

        return latex

    def generate_corruption_subtasks_table(self) -> str:
        """Generate detailed corruption subtasks table."""
        train_stats = self.get_task_stats("corruption", "train")
        test_stats = self.get_task_stats("corruption", "test")

        # Convert template numbers to corruption types
        train_corruption_stats = {}
        for template, count in train_stats.items():
            corruption_type = self.corruption_mapping.get(template, template)
            train_corruption_stats[corruption_type] = count

        test_corruption_stats = {}
        for template, count in test_stats.items():
            corruption_type = self.corruption_mapping.get(template, template)
            test_corruption_stats[corruption_type] = count

        latex = """\\begin{table*}[t]
\\centering
\\scriptsize
\\setlength{\\tabcolsep}{4pt}
\\begin{tabularx}{\\textwidth}{Xr}
\\toprule
\\textbf{Subtask (Corruption Type)} & \\textbf{Samples Count} \\\\
\\midrule
\\multicolumn{2}{c}{\\textit{Train Split}} \\\\
\\midrule
"""

        # Sort by count (descending)
        train_sorted = sorted(
            train_corruption_stats.items(), key=lambda x: x[1], reverse=True
        )

        for corruption_type, count in train_sorted:
            latex += f"{corruption_type} & {self.format_number(count)} \\\\\n"

        latex += """\\midrule
\\multicolumn{2}{c}{\\textit{Test Split}} \\\\
\\midrule
"""

        test_sorted = sorted(
            test_corruption_stats.items(), key=lambda x: x[1], reverse=True
        )

        for corruption_type, count in test_sorted:
            latex += f"{corruption_type} & {self.format_number(count)} \\\\\n"

        latex += """\\bottomrule
\\end{tabularx}
\\caption{Detailed statistics per subtask for the \\textit{Corruption (Restoration)} task in the Arabic poetry IFT dataset.}
\\label{tab:ift_corruption_subtasks}
\\end{table*}"""

        return latex

    def generate_generation_subtasks_table(self) -> str:
        """Generate detailed generation subtasks table."""
        train_stats = self.get_task_stats("generation", "train")
        test_stats = self.get_task_stats("generation", "test")

        latex = """\\begin{table*}[t]
\\centering
\\scriptsize
\\setlength{\\tabcolsep}{4pt}
\\begin{tabularx}{\\textwidth}{Xr}
\\toprule
\\textbf{Subtask (Input $\\rightarrow$ Output)} & \\textbf{Samples Count} \\\\
\\midrule
\\multicolumn{2}{c}{\\textit{Train Split}} \\\\
\\midrule
"""

        train_sorted = sorted(train_stats.items(), key=lambda x: x[1], reverse=True)

        for template, count in train_sorted:
            clean_template = self.clean_template_name(template, "generation")
            latex += f"{clean_template} & {self.format_number(count)} \\\\\n"

        latex += """\\midrule
\\multicolumn{2}{c}{\\textit{Test Split}} \\\\
\\midrule
"""

        test_sorted = sorted(test_stats.items(), key=lambda x: x[1], reverse=True)

        for template, count in test_sorted:
            clean_template = self.clean_template_name(template, "generation")
            latex += f"{clean_template} & {self.format_number(count)} \\\\\n"

        latex += """\\bottomrule
\\end{tabularx}
\\caption{Detailed statistics per subtask for the \\textit{Generation} task in the Arabic poetry IFT dataset.}
\\label{tab:ift_generation_subtasks}
\\end{table*}"""

        return latex

    def clean_template_name(self, template: str, task: str) -> str:
        """Clean up template names for display in tables."""
        # Remove tuple parentheses and quotes
        template = template.replace("(", "").replace(")", "").replace("'", "")

        if task == "continuation":
            # For continuation, replace " -> None" with " -> poem_continuation"
            template = template.replace(
                " -> None", " $\\rightarrow$ poem\\_continuation"
            )
        elif task == "analysis":
            # For analysis, format the arrow properly
            if " -> " in template:
                parts = template.split(" -> ")
                inputs = parts[0]
                output = parts[1]
                # Clean up commas in inputs
                inputs = inputs.replace(", ", ", ")
                template = f"{inputs} $\\rightarrow$ {output}"
        elif task == "generation":
            # For generation, format the arrow properly
            if " -> " in template:
                parts = template.split(" -> ")
                inputs = parts[0]
                output = parts[1]
                # Clean up commas in inputs
                inputs = inputs.replace(", ", ", ")
                template = f"{inputs} $\\rightarrow$ {output}"

        # Escape underscores for LaTeX
        template = template.replace("_", "\\_")

        return template

    def generate_all_tables(self) -> str:
        """Generate all tables and return as a single string."""
        tables = []

        print("Generating overall statistics table...")
        tables.append(self.generate_overall_stats_table())

        print("Generating analysis subtasks table...")
        tables.append(self.generate_analysis_subtasks_table())

        print("Generating continuation subtasks table...")
        tables.append(self.generate_continuation_subtasks_table())

        print("Generating corruption subtasks table...")
        tables.append(self.generate_corruption_subtasks_table())

        print("Generating generation subtasks table...")
        tables.append(self.generate_generation_subtasks_table())

        return "\n\n\n".join(tables)


def main():
    """Main function to run the stats extraction."""
    stats_dir = (
        "/path/to/data/dialectical_IFT_DATA/stats"
    )

    if not os.path.exists(stats_dir):
        print(f"Error: Stats directory not found: {stats_dir}")
        return

    extractor = IFTStatsExtractor(stats_dir)

    # Generate all tables
    all_tables = extractor.generate_all_tables()

    # Write to output file
    output_file = "ift_dataset_tables.tex"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(all_tables)

    print(f"\nAll tables have been written to: {output_file}")

    # Also print to console for immediate viewing
    print("\n" + "=" * 80)
    print("GENERATED LATEX TABLES")
    print("=" * 80)
    print(all_tables)


if __name__ == "__main__":
    main()

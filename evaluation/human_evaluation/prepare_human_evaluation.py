#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to prepare data for human evaluation of Arabic poetry generation models.

This script:
1. Loads model predictions from 4 different models
2. Samples 100 random generation examples
3. Creates a blind evaluation sheet with unique IDs
4. Generates both the evaluation sheet and a mapping file

Author: Prepared for Human Evaluation
Date: December 29, 2025
"""

import pandas as pd
import numpy as np
import random
import json
from pathlib import Path
from typing import List, Dict, Tuple
import argparse


# Model paths configuration
# Maps directory name to internal path structure
MODEL_PATHS = {
    "ALLaM-7B-Instruct-preview_base": {
        "dir_name": "ALLaM-7B-Instruct-preview_base",
        "internal_path": "ALLaM-7B-Instruct-preview/base/base-model"
    },
    "ALLaM-7B-Instruct-preview_random_checkpoint-42216": {
        "dir_name": "ALLaM-7B-Instruct-preview_random_checkpoint-42216",
        "internal_path": "ALLaM-7B-Instruct-preview/random/checkpoint-42216"
    },
    "Qwen3-8B_base": {
        "dir_name": "Qwen3-8B_base",
        "internal_path": "Qwen3-8B/base/base-model"
    },
    "Qwen3-8B_random_checkpoint-35000": {
        "dir_name": "Qwen3-8B_random_checkpoint-35000",
        "internal_path": "Qwen3-8B/random/checkpoint-35000"
    }
}


def load_model_predictions(base_dir: Path, task: str = "generation") -> Dict[str, pd.DataFrame]:
    """
    Load model predictions from all 4 models.
    
    Args:
        base_dir: Base directory containing model outputs
        task: Task name (generation, continuation, corruption)
    
    Returns:
        Dictionary mapping model name to dataframe
    """
    print(f"Loading {task} predictions from {base_dir}...")
    
    model_dfs = {}
    
    for model_name, paths_config in MODEL_PATHS.items():
        dir_name = paths_config["dir_name"]
        internal_path = paths_config["internal_path"]
        file_path = base_dir / dir_name / internal_path / task / f"{task}_ift_with_predictions.tsv"
        
        if not file_path.exists():
            print(f"WARNING: File not found: {file_path}")
            continue
        
        try:
            df = pd.read_csv(file_path, sep='\t')
            model_dfs[model_name] = df
            print(f"  ✓ Loaded {len(df)} samples from {model_name}")
        except Exception as e:
            print(f"  ✗ Error loading {model_name}: {e}")
    
    return model_dfs


def sample_common_examples(model_dfs: Dict[str, pd.DataFrame], n_samples: int = 100, seed: int = 42) -> pd.DataFrame:
    """
    Sample n_samples that are common across all models (same input).
    
    Args:
        model_dfs: Dictionary of model dataframes
        n_samples: Number of samples to select
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with sampled inputs
    """
    print(f"\nSampling {n_samples} common examples...")
    
    # Find common inputs across all models
    common_inputs = None
    
    for model_name, df in model_dfs.items():
        if common_inputs is None:
            common_inputs = set(df['input'].unique())
        else:
            common_inputs = common_inputs.intersection(set(df['input'].unique()))
    
    print(f"  Found {len(common_inputs)} common inputs across all models")
    
    # Sample from common inputs
    random.seed(seed)
    sampled_inputs = random.sample(list(common_inputs), min(n_samples, len(common_inputs)))
    
    # Create dataframe with sampled inputs
    sampled_df = pd.DataFrame({'input': sampled_inputs})
    
    print(f"  Sampled {len(sampled_df)} examples")
    
    return sampled_df


def create_evaluation_dataset(
    sampled_df: pd.DataFrame,
    model_dfs: Dict[str, pd.DataFrame],
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create the evaluation dataset with blind model IDs and mapping.
    
    Args:
        sampled_df: DataFrame with sampled inputs
        model_dfs: Dictionary of model dataframes
        seed: Random seed for ID generation
    
    Returns:
        Tuple of (evaluation_df, mapping_df)
    """
    print("\nCreating evaluation dataset...")
    
    # Create model ID mapping (randomized)
    model_names = list(model_dfs.keys())
    random.seed(seed)
    shuffled_indices = list(range(1, len(model_names) + 1))
    random.shuffle(shuffled_indices)
    
    model_id_mapping = {
        model_name: f"model_{idx}"
        for model_name, idx in zip(model_names, shuffled_indices)
    }
    
    print(f"  Created blind model IDs")
    
    # Create evaluation rows
    eval_rows = []
    
    for idx, row in sampled_df.iterrows():
        input_text = row['input']
        
        # Get predictions from each model for this input
        for model_name, df in model_dfs.items():
            model_row = df[df['input'] == input_text]
            
            if len(model_row) == 0:
                print(f"  WARNING: No prediction found for input in {model_name}")
                continue
            
            model_row = model_row.iloc[0]
            
            # Create evaluation entry
            eval_entry = {
                'sample_id': f"sample_{idx + 1:03d}",
                'model_id': model_id_mapping[model_name],
                'instruction': input_text,
                'model_generation': model_row['model_generation'],
                'dialect': model_row.get('dialect', ''),
                'ground_truth': model_row.get('output', ''),
                # Evaluation columns (empty for annotators to fill)
                'الالتزام_بالقيود': '',
                'الطلاقة_اللغوية_والإيقاع': '',
                'التماسك_والمعنى': '',
                'الشاعرية_والجمال_الفني': '',
                'notes': ''
            }
            
            eval_rows.append(eval_entry)
    
    eval_df = pd.DataFrame(eval_rows)
    
    # Create mapping dataframe
    mapping_rows = [
        {'model_id': model_id, 'actual_model_name': model_name}
        for model_name, model_id in model_id_mapping.items()
    ]
    mapping_df = pd.DataFrame(mapping_rows)
    
    print(f"  Created {len(eval_df)} evaluation entries")
    
    return eval_df, mapping_df


def create_evaluation_sheet(
    eval_df: pd.DataFrame,
    output_path: Path,
    seed: int = 42
):
    """
    Create a formatted evaluation sheet for human annotators.
    
    Args:
        eval_df: Evaluation dataframe
        output_path: Path to save the Excel file
        seed: Random seed for shuffling
    """
    print(f"\nCreating evaluation sheet at {output_path}...")
    
    # Shuffle the dataframe to avoid ordering bias (e.g., all samples from one model together)
    shuffled_df = eval_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"  Shuffled {len(shuffled_df)} rows for blind evaluation")
    
    # Create a cleaner version for annotators
    annotator_df = shuffled_df[[
        'sample_id',
        'model_id',
        'instruction',
        'model_generation',
        'الالتزام_بالقيود',
        'الطلاقة_اللغوية_والإيقاع',
        'التماسك_والمعنى',
        'الشاعرية_والجمال_الفني',
        'notes'
    ]].copy()
    
    # Save to Excel with formatting
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        annotator_df.to_excel(writer, sheet_name='Evaluation', index=False)
        
        # Get the worksheet
        worksheet = writer.sheets['Evaluation']
        
        # Adjust column widths
        worksheet.column_dimensions['A'].width = 12  # sample_id
        worksheet.column_dimensions['B'].width = 12  # model_id
        worksheet.column_dimensions['C'].width = 50  # instruction
        worksheet.column_dimensions['D'].width = 50  # model_generation
        worksheet.column_dimensions['E'].width = 20  # الالتزام_بالقيود
        worksheet.column_dimensions['F'].width = 25  # الطلاقة_اللغوية_والإيقاع
        worksheet.column_dimensions['G'].width = 20  # التماسك_والمعنى
        worksheet.column_dimensions['H'].width = 25  # الشاعرية_والجمال_الفني
        worksheet.column_dimensions['I'].width = 30  # notes
    
    print(f"  ✓ Saved evaluation sheet")


def create_instructions_sheet(output_dir: Path):
    """
    Create an instructions sheet for annotators in Arabic.
    
    Args:
        output_dir: Directory to save the instructions
    """
    instructions = """
# تعليمات التقييم البشري للشعر العربي

## نظرة عامة
أنت ستقوم بتقييم قصائد عربية تم إنشاؤها بواسطة نماذج ذكاء اصطناعي مختلفة. 
المهمة هي تقييم جودة القصائد المُولدة بناءً على أربعة معايير.

## معايير التقييم

### 1. الالتزام بالقيود (Compliance) [1-5]
- هل التزمت القصيدة بالتعليمات المُعطاة؟
- هل تم اتباع المواضيع والكلمات المفتاحية المطلوبة؟
- هل تم الالتزام بالبحر والقافية إن وُجدت في التعليمات؟

**معايير التقييم:**
- 1: لم تلتزم بالقيود على الإطلاق
- 2: التزمت ببعض القيود فقط
- 3: التزمت بأغلب القيود مع بعض الأخطاء
- 4: التزمت بمعظم القيود بشكل جيد
- 5: التزمت بجميع القيود بشكل كامل ودقيق

### 2. الطلاقة اللغوية والإيقاع (Fluency) [1-5]
- هل اللغة صحيحة نحوياً وإملائياً؟
- هل الإيقاع سلس ومنسجم؟
- هل النص سهل القراءة وطبيعي؟

**معايير التقييم:**
- 1: أخطاء نحوية كثيرة وإيقاع متقطع
- 2: أخطاء نحوية ملحوظة وإيقاع ضعيف
- 3: أخطاء قليلة وإيقاع مقبول
- 4: لغة سليمة وإيقاع جيد
- 5: لغة ممتازة وإيقاع متقن

### 3. التماسك والمعنى (Coherence) [1-5]
- هل الأفكار متسقة ومترابطة؟
- هل المعنى واضح ومفهوم؟
- هل هناك تسلسل منطقي بين الأبيات؟

**معايير التقييم:**
- 1: لا يوجد تماسك أو ترابط بين الأفكار
- 2: بعض التماسك ولكن المعنى غير واضح
- 3: تماسك مقبول مع بعض الغموض
- 4: تماسك جيد ومعنى واضح
- 5: تماسك ممتاز ومعنى عميق ومترابط

### 4. الشاعرية والجمال الفني (Poetic Quality) [1-5]
- هل تحتوي القصيدة على صور بلاغية وفنية؟
- هل هناك جمال في التعبير والأسلوب؟
- هل تثير القصيدة المشاعر والخيال؟

**معايير التقييم:**
- 1: لا توجد أي عناصر شاعرية أو جمالية
- 2: عناصر شاعرية ضعيفة جداً
- 3: بعض العناصر الشاعرية البسيطة
- 4: شاعرية جيدة مع صور بلاغية
- 5: شاعرية ممتازة وجمال فني راقٍ

## كيفية التقييم

1. اقرأ **التعليمات (instruction)** بعناية لفهم المطلوب
2. اقرأ **النص المُولد (model_generation)** عدة مرات
3. قيّم كل معيار من المعايير الأربعة بشكل مستقل
4. ضع درجة من 1 إلى 5 في كل خانة
5. إذا كان لديك ملاحظات إضافية، اكتبها في خانة **notes**

## ملاحظات مهمة

- التقييم أعمى (blind): أنت لا تعرف أي نموذج أنتج أي قصيدة
- كن موضوعياً ومنصفاً في التقييم
- لا تدع تفضيلك الشخصي للموضوع يؤثر على التقييم
- ركز على المعايير المحددة فقط

## أمثلة

**مثال على تقييم جيد:**
- الالتزام بالقيود: 5 (التزمت بجميع الكلمات المفتاحية والموضوع)
- الطلاقة اللغوية: 4 (لغة سليمة مع خطأ إملائي بسيط)
- التماسك والمعنى: 5 (أفكار مترابطة ومعنى عميق)
- الشاعرية والجمال: 4 (صور بلاغية جميلة)

**مثال على تقييم ضعيف:**
- الالتزام بالقيود: 2 (لم تلتزم بنصف الكلمات المفتاحية)
- الطلاقة اللغوية: 3 (أخطاء نحوية قليلة)
- التماسك والمعنى: 2 (أفكار متناثرة وغير مترابطة)
- الشاعرية والجمال: 3 (شاعرية بسيطة)

---

شكراً لمساهمتك في تحسين جودة نماذج إنشاء الشعر العربي!
"""
    
    instructions_path = output_dir / "تعليمات_التقييم.txt"
    with open(instructions_path, 'w', encoding='utf-8') as f:
        f.write(instructions)
    
    print(f"  ✓ Created instructions file: {instructions_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare data for human evaluation of Arabic poetry models"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/path/to/model_outputs",
        help="Base directory containing model outputs"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).resolve().parent),
        help="Output directory for evaluation files"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="generation",
        choices=["generation", "continuation", "corruption"],
        help="Task to evaluate"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Preparing Human Evaluation Data")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Number of samples: {args.n_samples}")
    print(f"Random seed: {args.seed}")
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Step 1: Load model predictions
    model_dfs = load_model_predictions(base_dir, args.task)
    
    if len(model_dfs) == 0:
        print("\n❌ ERROR: No model predictions loaded. Please check the paths.")
        return
    
    print(f"\n✓ Successfully loaded predictions from {len(model_dfs)} models")
    
    # Step 2: Sample common examples
    sampled_df = sample_common_examples(model_dfs, args.n_samples, args.seed)
    
    # Step 3: Create evaluation dataset
    eval_df, mapping_df = create_evaluation_dataset(sampled_df, model_dfs, args.seed)
    
    # Step 4: Save files
    print("\nSaving output files...")
    
    # Save full evaluation dataset (with ground truth for reference)
    full_eval_path = output_dir / f"{args.task}_evaluation_full.csv"
    eval_df.to_csv(full_eval_path, index=False, encoding='utf-8-sig')
    print(f"  ✓ Saved full evaluation data: {full_eval_path}")
    
    # Save evaluation sheet for annotators (Excel)
    eval_sheet_path = output_dir / f"{args.task}_evaluation_sheet.xlsx"
    create_evaluation_sheet(eval_df, eval_sheet_path, seed=args.seed)
    
    # Save mapping file (keep this secret!)
    mapping_path = output_dir / f"{args.task}_model_mapping.json"
    mapping_dict = dict(zip(mapping_df['model_id'], mapping_df['actual_model_name']))
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(mapping_dict, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved model mapping (CONFIDENTIAL): {mapping_path}")
    
    # Save statistics
    stats_path = output_dir / f"{args.task}_evaluation_stats.txt"
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("Human Evaluation Dataset Statistics\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Task: {args.task}\n")
        f.write(f"Number of unique samples: {len(sampled_df)}\n")
        f.write(f"Number of models: {len(model_dfs)}\n")
        f.write(f"Total evaluation entries: {len(eval_df)}\n")
        f.write(f"Entries per sample: {len(eval_df) // len(sampled_df)}\n\n")
        f.write("Models included:\n")
        for model_name in model_dfs.keys():
            f.write(f"  - {model_name}\n")
        f.write(f"\nRandom seed: {args.seed}\n")
    print(f"  ✓ Saved statistics: {stats_path}")
    
    # Create instructions file
    create_instructions_sheet(output_dir)
    
    print("\n" + "=" * 60)
    print("✅ SUCCESS! All files created successfully")
    print("=" * 60)
    print("\nFiles created:")
    print(f"  1. {eval_sheet_path.name} - For annotators (Excel)")
    print(f"  2. {full_eval_path.name} - Full dataset (CSV)")
    print(f"  3. {mapping_path.name} - Model mapping (JSON, CONFIDENTIAL)")
    print(f"  4. {stats_path.name} - Statistics")
    print(f"  5. تعليمات_التقييم.txt - Instructions in Arabic")
    print("\n" + "=" * 60)
    print("Next steps:")
    print("  1. Send the Excel sheet to annotators")
    print("  2. Keep the mapping file confidential")
    print("  3. Share the instructions file with annotators")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""To generate MCQ given an Arabic cooking recipe"""
import logging
import pandas as pd
from pathlib import Path
from pydantic import BaseModel
from jinja2 import Template as JinjaTemplate
import json
import random

logger = logging.getLogger(__name__)

MODEL_NAME="gemini-2.0-flash"
# ONEDRIVE = Path("/mnt/data/users/anwarvic/ONEDRIVE")
# Paths
PROMPTS_PATH = Path("")
DATA_PATH = Path("")
OUTPUT_PATH = Path("")



# def load_dataset(self) -> Optional[pd.DataFrame]:
#     """Load poetry dataset"""
#     try:
#         df = pd.read_csv(WholeDatasetConfig.DATASET_PATH)
#         self.logger.info(f"Successfully loaded {len(df)} poems from dataset")
            
#         # Filter for poems with complete data
#         complete_poems = df.dropna(subset=['poem_text', 'poet_name', 'poet_era', 'meter', 'rhyme_letter'])
#         self.logger.info(f"Found {len(complete_poems)} poems with complete data")
            
#         # Apply test mode limitations if enabled
#         if WholeDatasetConfig.TEST_MODE:
#             complete_poems = complete_poems.head(WholeDatasetConfig.TEST_POEMS_COUNT)
#             self.logger.info(f" TEST MODE: Limited to {len(complete_poems)} poems for testing")
#         else:
#             self.logger.info(f"FULL MODE: Processing all {len(complete_poems)} poems")
            
#         return complete_poems
#     except Exception as e:
#         self.logger.error(f"Error loading dataset: {e}")
#     return None
        
# def load_dataset() -> pd.DataFrame:
#     """Loads a dataset from a file and returns a pandas DataFrame"""
#     # excel_filepath = ONEDRIVE / "Data/jais/raw/cuisine/benchmark/hard_subset.xlsx"
#     excel_filepath = "C:\\path\\to\\de_dupped_test_with_keywords.csv"
#     df = pd.read_excel(excel_filepath, engine='openpyxl')
#     logger.info(f"Loaded {len(df)} entries from {excel_filepath}.")
#     return df

def load_dataset() -> pd.DataFrame:
    """Loads the poetry dataset from CSV."""
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded {len(df)} poems from {DATA_PATH}.")
    return df


# def prepare_prompt(df: pd.DataFrame) -> list[dict]:
#     """Prepare the prompt for each entry in the dataset."""
#     # create the prompt column
#     df["prompt"] = df.apply(
#         lambda row: JinjaTemplate(PROMPT).render(**row),
#         axis=1
#     )
#     logger.info(f"Prepared prompts for {len(df)} entries.")
#     df = df[["id", "prompt"]]
#     # Convert DataFrame to list of dictionaries
#     list_json = df.to_dict(orient='records')
#     return list_json


def prepare_prompt(prompts_csv: str, row_num: int, data_csv: str, data_row: int):
    """
    Prepares a prompt by retrieving it from prompts_csv and replacing placeholders
    with values from a specific row in data_csv.

    Args:
        prompts_csv (str): Path to the CSV containing prompts with placeholders.
        row_num (int): Row index (0-based) of the prompt.
        data_csv (str): Path to the CSV containing actual poem data.
        data_row (int): Row index (0-based) of the poem data to fill placeholders.

    Returns:
        tuple: (filled_prompt, corruption_type)
    """
    # Load CSVs with explicit UTF-8 encoding
    df_prompts = pd.read_csv(prompts_csv, encoding='utf-8-sig')
    df_data = pd.read_csv(data_csv, encoding='utf-8')

    # Get the relevant rows
    row_prompt = df_prompts.iloc[row_num]
    row_data = df_data.iloc[data_row]

    # Extract prompt template
    prompt_template = row_prompt["Prompt to alter original poetry (for Gemini 2.5)"]

    # Define possible eras and meters
    eras = [
        "العصر الجاهلي", "العصر الإسلامي", "عصر صدر الإسلام",
        "العصر الأموي", "العصر العباسي", "العصر الأندلسي", 
        "عصر المماليك", "العصر العثماني", "عصر النهضة",
        "العصر الحديث", "العصر المعاصر"
    ]
    meters = ["الكامل", "الطويل", "البسيط", "الوافر", "الرجز", "المتقارب"]

    # Replace placeholders using f-strings for better Unicode handling
    filled_prompt = prompt_template
    for col in df_data.columns:
        placeholder = f"{{{col}}}"
        if placeholder in filled_prompt:
            # Ensure value is a string before replacing
            value = str(row_data[col])
            filled_prompt = filled_prompt.replace(placeholder, value)

    # Handle target_era
    if "{target_era}" in filled_prompt:
        poet_era = str(row_data.get("poet_era", "")).strip()
        candidate_eras = [era for era in eras if era != poet_era]
        target_era = random.choice(candidate_eras) if candidate_eras else poet_era
        filled_prompt = filled_prompt.replace("{target_era}", target_era)

    # Handle target_meter
    if "{target_meter}" in filled_prompt:
        poet_meter = str(row_data.get("meter", "")).strip()
        candidate_meters = [m for m in meters if m != poet_meter]
        target_meter = random.choice(candidate_meters) if candidate_meters else poet_meter
        filled_prompt = filled_prompt.replace("{target_meter}", target_meter)

    # Extract type from placeholder column
    corruption_type = row_prompt["corruption_type"].strip()
    print(filled_prompt)
   
    return filled_prompt, corruption_type



# def prepare_prompt(prompts_csv: str, row_num: int, data_csv: str, data_row: int):
#     """
#     Prepares a prompt by retrieving it from prompts_csv and replacing placeholders
#     with values from a specific row in data_csv.

#     Args:
#         prompts_csv (str): Path to the CSV containing prompts with placeholders.
#         row_num (int): Row index (0-based) of the prompt.
#         data_csv (str): Path to the CSV containing actual poem data.
#         data_row (int): Row index (0-based) of the poem data to fill placeholders.

#     Returns:
#         tuple: (filled_prompt, corruption_type)
#     """
#     # Load CSVs
#     df_prompts = pd.read_csv(prompts_csv)
#     df_data = pd.read_csv(data_csv)

#     # Get the relevant rows
#     row_prompt = df_prompts.iloc[row_num]
#     row_data = df_data.iloc[data_row]

#     # Extract prompt template
#     prompt_template = row_prompt["Prompt to alter original poetry (for Gemini 2.5)"]

#     # Define possible eras and meters
#     eras = [
#         "العصر الجاهلي", "العصر الإسلامي", "عصر صدر الإسلام",
#         "العصر الأموي", "العصر العباسي", "العصر الأندلسي", 
#         "عصر المماليك", "العصر العثماني", "عصر النهضة",
#         "العصر الحديث", "العصر المعاصر"
#     ]
#     meters = ["الكامل", "الطويل", "البسيط", "الوافر", "الرجز", "المتقارب"]

#     # Replace placeholders
#     filled_prompt = prompt_template
#     for col in df_data.columns:
#         placeholder = "{" + col + "}"
#         if placeholder in filled_prompt:
#             filled_prompt = filled_prompt.replace(placeholder, " " + str(row_data[col]) + " ")

    

#     # Handle target_era
#     if "{target_era}" in filled_prompt:
#         poet_era = str(row_data.get("poet_era", "")).strip()
#         candidate_eras = [era for era in eras if era != poet_era]
#         target_era = random.choice(candidate_eras) if candidate_eras else poet_era
#         filled_prompt = filled_prompt.replace("{target_era}", " " + target_era + " ")

#     # Handle target_meter
#     if "{target_meter}" in filled_prompt:
#         poet_meter = str(row_data.get("meter", "")).strip()
#         candidate_meters = [m for m in meters if m != poet_meter]
#         target_meter = random.choice(candidate_meters) if candidate_meters else poet_meter
#         filled_prompt = filled_prompt.replace("{target_meter}", " " + target_meter + " ")

#     # Extract type from placeholder column
#     corruption_type = row_prompt["corruption_type"].strip()
   
#     return filled_prompt, corruption_type


# class Data(BaseModel):
#     corrupted_poem: str
    
# # class DataEntry(BaseModel):
# #     mcq: list[MCQ, MCQ, MCQ]


# def get_output_path() -> Path:
#     # return ONEDRIVE / "Data/jais/raw/cuisine/benchmark/hard_subset/hard_subset_v2.jsonl"
#     return  "C:\\path\\to\\Res.jsonl"

class Data(BaseModel):
    corrupted_poem: str


def get_output_path() -> Path:
    """Return the path where outputs should be saved."""
    return OUTPUT_PATH

# def postprocess_output() -> Data:
#     """Post-process the output data into a list of DataEntry objects."""
#     # load input dataset
#     in_df = load_dataset()
#     # convert the output to JSON format
#     out_jsonl_filepath = get_output_path()
#     out_df = pd.read_json(out_jsonl_filepath, lines=True)
#     # merge the two DataFrames
#     merged_df = pd.merge(in_df, out_df, on="id", how="inner")
#     # keep only the necessary columns
#     merged_df = merged_df[["id", "Name", "Steps", "Ingredients", "mcq"]]
#     # Explode mcq and keep id/index aligned
#     exploded_df = merged_df.explode("mcq").reset_index(drop=False)  # Keep original index for later
#     mcq_df = pd.json_normalize(exploded_df["mcq"])  # Normalize exploded dicts
#     mcq_df["id"] = exploded_df["id"]   # Add original index back to align
#     # merge mcq_df with merged_df on "id"
#     final_df = pd.merge(mcq_df, merged_df, on="id", how="left")
#     # keep only the necessary columns
#     final_df = final_df[["id", "Name", "Steps", "Ingredients", "question", "options", "correct_answer"]]
#     # write to excel file
#     out_path = out_jsonl_filepath.parent
#     final_df.to_excel(
#         out_path / "mcq_common_subset_v1.xlsx",
#         index=False,
#         engine='openpyxl'
#     )
"""To generate corrupted poetry prompts and handle outputs"""
import logging
import pandas as pd
from pathlib import Path
from pydantic import BaseModel
import random

logger = logging.getLogger(__name__)




def load_dataset() -> pd.DataFrame:
    """Loads the poetry dataset from CSV."""
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded {len(df)} poems from {DATA_PATH}.")
    return df


def prepare_prompt(prompts_csv: str, row_num: int, data_csv: str, data_row: int):
    """
    Prepares a prompt by retrieving it from prompts_csv and replacing placeholders
    with values from a specific row in data_csv.
    """
    # Load CSVs
    df_prompts = pd.read_csv(prompts_csv)
    df_data = pd.read_csv(data_csv)

    # Get the relevant rows
    row_prompt = df_prompts.iloc[row_num]
    row_data = df_data.iloc[data_row]

    # Extract prompt template
    prompt_template = row_prompt["Prompt to alter original poetry (for Gemini 2.5)"]

    # Define possible eras and meters
    eras = [
        "العصر الجاهلي", "العصر الإسلامي", "عصر صدر الإسلام",
        "العصر الأموي", "العصر العباسي", "العصر الأندلسي", 
        "عصر المماليك", "العصر العثماني", "عصر النهضة",
        "العصر الحديث", "العصر المعاصر"
    ]
    meters = ["الكامل", "الطويل", "البسيط", "الوافر", "الرجز", "المتقارب"]

    # Replace placeholders with actual values
    filled_prompt = prompt_template
    for col in df_data.columns:
        placeholder = "{" + col + "}"
        if placeholder in filled_prompt:
            filled_prompt = filled_prompt.replace(placeholder, "\t" + str(row_data[col]) + "\t")

    # Handle {target_era}
    if "{target_era}" in filled_prompt:
        poet_era = str(row_data.get("poet_era", "")).strip()
        candidate_eras = [era for era in eras if era != poet_era]
        target_era = random.choice(candidate_eras) if candidate_eras else poet_era
        filled_prompt = filled_prompt.replace("{target_era}", "\t" + target_era + "\t")

    # Handle {target_meter}
    if "{target_meter}" in filled_prompt:
        poet_meter = str(row_data.get("meter", "")).strip()
        candidate_meters = [m for m in meters if m != poet_meter]
        target_meter = random.choice(candidate_meters) if candidate_meters else poet_meter
        filled_prompt = filled_prompt.replace("{target_meter}", "\t" + target_meter + "\t")

    # Extract type from placeholder column
    corruption_type = row_prompt["corruption_type"].strip()

    return filled_prompt, corruption_type


class Data(BaseModel):
    id: str
    corrupted_text: str
    original_poem: str
    poet_name: str
    corruption_type: str



def get_output_path() -> Path:
    """Return the path where outputs should be saved."""
    return OUTPUT_PATH


# def postprocess_output():
#     """Post-process model outputs and save final results."""
#     in_df = load_dataset()
#     out_jsonl_filepath = get_output_path()

#     # Load outputs (model-generated corrupted poems)
#     out_df = pd.read_json(out_jsonl_filepath, lines=True)

#     # Merge on poem id
#     merged_df = pd.merge(in_df, out_df, on="id", how="inner")

#     # Keep only useful columns (assuming corrupted_poem is in output JSON)
#     if "corrupted_poem" in merged_df.columns:
#         final_df = merged_df[["id", "poem_text", "poet_name", "poet_era", "meter", "corrupted_poem"]]
#     else:
#         logger.warning("No 'corrupted_poem' found in outputs.")
#         final_df = merged_df

#     # Save as Excel
#     out_path = out_jsonl_filepath.parent
#     final_path = out_path / "corrupted_poems.xlsx"
#     final_df.to_excel(final_path, index=False, engine='openpyxl')
#     logger.info(f"Final results written to {final_path}")
def postprocess_output():
    """Postprocess the generated output and merge with original data."""
    try:
        import json  # Make sure json is imported
        
        # Load the original data
        in_df = load_dataset()
        print(f"Original data shape: {in_df.shape}")
        
        # Load the generated output
        out_file = get_output_path()
        print(f"Output file: {out_file}")
        
        out_data = []
        with open(out_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    out_data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON line: {e}")
                    continue
        
        out_df = pd.DataFrame(out_data)
        print(f"Output data shape: {out_df.shape}")
        print(f"Output IDs: {out_df['id'].tolist() if 'id' in out_df.columns else 'No ID column'}")
        
        # Check if we have the expected ID format
        if 'id' in out_df.columns:
            # Extract prompt_num and data_row with proper error handling
            id_extract = out_df['id'].str.extract(r'p(?P<prompt_num>\d+)_r(?P<data_row>\d+)')
            
            # Check for any NaN values (IDs that didn't match the pattern)
            nan_mask = id_extract.isna().any(axis=1)
            if nan_mask.any():
                print(f"Warning: {nan_mask.sum()} IDs don't match the expected pattern:")
                invalid_ids = out_df.loc[nan_mask, 'id'].tolist()
                print(f"Invalid IDs: {invalid_ids}")
                
                # Remove rows with invalid IDs
                out_df = out_df[~nan_mask].copy()
                id_extract = id_extract[~nan_mask]
            
            # Convert to integers, handling any remaining errors
            out_df['prompt_num'] = pd.to_numeric(id_extract['prompt_num'], errors='coerce')
            out_df['data_row'] = pd.to_numeric(id_extract['data_row'], errors='coerce')
            
            # Remove any rows that still have NaN values after conversion
            nan_mask = out_df[['prompt_num', 'data_row']].isna().any(axis=1)
            if nan_mask.any():
                print(f"Removing {nan_mask.sum()} rows with invalid numeric values")
                out_df = out_df[~nan_mask]
            
            print(f"Processed output data shape: {out_df.shape}")
            
            # Merge based on the data_row (which should correspond to the original dataset index)
            if not out_df.empty:
                # Make sure data_row values are within the bounds of the original dataset
                valid_rows = out_df['data_row'] < len(in_df)
                if not valid_rows.all():
                    print(f"Warning: Some data_row values are out of bounds")
                    out_df = out_df[valid_rows]
                
                if not out_df.empty:
                    # Reset index of original data for merging
                    in_df_reset = in_df.reset_index()
                    
                    # Merge on the data_row (which should match the original index)
                    merged_df = out_df.merge(in_df_reset, left_on='data_row', right_on='index', how='left')
                    
                    # Save the merged data
                    output_path = get_output_path().with_suffix('.csv')
                    merged_df.to_csv(output_path, index=False, encoding='utf-8')
                    print(f"Merged data saved to: {output_path}")
                    print(f"Merged data shape: {merged_df.shape}")
                else:
                    print("No valid rows to merge after filtering")
            else:
                print("No valid output data to process")
        
    except Exception as e:
        print(f"Error in postprocess_output: {e}")
        import traceback
        traceback.print_exc()

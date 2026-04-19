import time
import yaml
import json
import logging
import pandas as pd
from tqdm import tqdm
import google.generativeai as genai
from pathlib import Path
from threading import Lock
from pydantic import BaseModel
from tqdm.contrib.concurrent import thread_map
import re
from typing import List, Dict, Any

# add path
import sys
sys.path.append(str(Path(__file__).resolve().parent))

from poem_corrupt import *

LOG_PATH =""
PROMPT_PATH = "" 
DATA_PATH = ""
BATCH_SIZE = 5
MAX_PROMPTS = 8
MAX_DATA_ROWS = 10

def load_yaml(yaml_filepath: Path) -> dict:
    if not yaml_filepath.exists():
        raise FileNotFoundError(
            f"Credentials file {yaml_filepath} does not exist."
        )
    with open(yaml_filepath, "r") as f:
        return yaml.safe_load(f)

# load important variables
credentials_filepath = Path(__file__).resolve().parent.parent.parent / "creds.yaml"
VARS = load_yaml(credentials_filepath)

# setup logger
log_filepath = Path(LOG_PATH +"gemini_batch_translate.log")
log_filepath.parent.mkdir(parents=True, exist_ok=True)
log_format = "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d (%(funcName)s) - %(message)s"
# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filepath, mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def load_gemini(api_key: str):
    """Load and configure Gemini client"""
    try:
        genai.configure(api_key=api_key)
        logger.info("Gemini client configured successfully.")
        return genai
    except Exception as e:
        logger.error(f"Failed to configure Gemini: {e}")
        raise

def gen_gemini_batch(client, schema: BaseModel, batch_prompts: List[Dict[str, Any]], max_retries=3, retry_delay=5):
    """Generate content for a batch of prompts using Gemini."""
    for attempt in range(max_retries):
        try:
            # Create batch prompt
            batch_content = create_batch_prompt(batch_prompts)
            
            # Create the model instance
            model = client.GenerativeModel(MODEL_NAME)
            
            # Generate content for the batch
            response = model.generate_content(batch_content)
            
            if response.text:
                # Parse batch response
                batch_results = parse_batch_response(response.text, batch_prompts)
                return batch_results, response
            else:
                logger.warning(f"Empty response for batch attempt {attempt + 1}")
                
        except Exception as e:
            logger.warning(f"Batch attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    
    raise RuntimeError("Failed to generate content for batch after multiple attempts.")

def create_batch_prompt(batch_prompts: List[Dict[str, Any]]) -> str:
    """Create a single prompt for a batch of requests."""
    batch_content = "Please process the following batch of poetry modification requests. " \
                   "For each request, provide the modified poem in JSON format as specified.\n\n"
    
    for i, prompt_data in enumerate(batch_prompts):
        batch_content += f"--- REQUEST {i+1} ---\n"
        batch_content += f"ID: {prompt_data['id']}\n"
        batch_content += f"PROMPT: {prompt_data['prompt']}\n\n"
    
    batch_content += "--- RESPONSE FORMAT ---\n"
    batch_content += "Please respond with a JSON array where each element corresponds to a request:\n"
    batch_content += "[\n"
    batch_content += "  {\n"
    batch_content += '    "id": "request_id",\n'
    batch_content += '    "corrupted_poem": "modified_poem_text"\n'
    batch_content += "  },\n"
    batch_content += "  ...\n"
    batch_content += "]\n"
    
    return batch_content

def parse_batch_response(response_text: str, batch_prompts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Parse the batch response and match results with original prompts."""
    try:
        # Try to parse as JSON array first
        if response_text.strip().startswith('['):
            parsed_results = json.loads(response_text)
            results = []
            for result in parsed_results:
                if isinstance(result, dict) and 'id' in result and 'corrupted_poem' in result:
                    results.append({
                        'id': result['id'],
                        'corrupted_poem': result['corrupted_poem']
                    })
            return results
        
        # Fallback: try to extract individual JSON objects
        json_pattern = r'\{[^{}]*\}'
        json_matches = re.findall(json_pattern, response_text)
        
        results = []
        for match in json_matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, dict) and 'id' in parsed and 'corrupted_poem' in parsed:
                    results.append({
                        'id': parsed['id'],
                        'corrupted_poem': parsed['corrupted_poem']
                    })
            except json.JSONDecodeError:
                continue
        
        return results
        
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse batch response as JSON: {e}")
        # Fallback: create results with original text
        return [{
            'id': prompt['id'],
            'corrupted_poem': response_text  # Use full response as fallback
        } for prompt in batch_prompts]

def prepare_batched_prompts(prompts_csv: str, data_csv: str, batch_size=5, max_prompts=8, max_data_rows=50):
    """Prepare prompts in batches for processing."""
    df_prompts = pd.read_csv(prompts_csv, encoding='utf-8-sig')
    df_data = pd.read_csv(data_csv, encoding='utf-8')
    
    batched_prompts = []
    batch_counter = 0
    
    # Limit the number of prompts and data rows for testing
    prompt_range = range(min(max_prompts, len(df_prompts)))
    data_range = range(min(max_data_rows, len(df_data)))
    
    for prompt_num in prompt_range:
        current_batch = []
        
        for data_row in data_range:
            try:
                filled_prompt, corruption_type = prepare_prompt(
                    prompts_csv, prompt_num, data_csv, data_row
                )
                
                current_batch.append({
                    "id": f"p{prompt_num}_r{data_row}",
                    "prompt": filled_prompt,
                    "corruption_type": corruption_type,
                    "original_poem": df_data.iloc[data_row].get("poem_text", ""),
                    "poet_name": df_data.iloc[data_row].get("poet_name", ""),
                    "data_row": data_row,
                    "prompt_num": prompt_num
                })
                
                # If batch is full, add to batched prompts
                if len(current_batch) >= batch_size:
                    batched_prompts.append(current_batch)
                    batch_counter += 1
                    current_batch = []
                    
            except Exception as e:
                logger.error(f"Error preparing prompt p{prompt_num}_r{data_row}: {e}")
                continue
        
        # Add remaining prompts in the current batch
        if current_batch:
            batched_prompts.append(current_batch)
            batch_counter += 1
    
    logger.info(f"Prepared {batch_counter} batches with {len(batched_prompts)} total batches")
    return batched_prompts

def process_batch_wrapper(args):
    """Wrapper function for processing a batch of prompts."""
    batch, client, schema, max_retries, retry_delay = args
    
    try:
        batch_results, response = gen_gemini_batch(
            client, schema, batch, max_retries, retry_delay
        )
        
        # Enhance results with original batch data
        enhanced_results = []
        for result in batch_results:
            # Find matching original prompt data
            original_data = next((item for item in batch if item['id'] == result['id']), None)
            if original_data:
                enhanced_result = {
                    "id": result['id'],
                    "corrupted_poem": result['corrupted_poem'],
                    "original_poem": original_data.get("original_poem", ""),
                    "poet_name": original_data.get("poet_name", ""),
                    "corruption_type": original_data.get("corruption_type", ""),
                    "prompt_used": original_data.get("prompt", ""),  # Save the prompt used
                    "data_row": original_data.get("data_row", ""),
                    "prompt_num": original_data.get("prompt_num", "")
                }
                enhanced_results.append(enhanced_result)
            else:
                # If no matching data, create minimal result
                enhanced_results.append({
                    "id": result['id'],
                    "corrupted_poem": result['corrupted_poem'],
                    "prompt_used": "Unknown"  # Fallback
                })
        
        # Estimate token count
        total_prompt_length = sum(len(item['prompt']) for item in batch)
        total_response_length = len(response.text) if response and response.text else 0
        token_count = (total_prompt_length + total_response_length) // 4  # Rough estimate
        
        return enhanced_results, token_count
        
    except Exception as e:
        logger.error(f"Failed to process batch: {e}")
        # Return empty results for failed batch
        return [], 0

def process_data_with_gemini_batch(
    client, schema: BaseModel,
    batched_prompts: List[List[Dict]],
    out_jsonl_filepath: Path,
    max_retries=3, retry_delay=5,
    max_workers=5,
    force_reprocess=False
):
    """Process data in batches using Gemini and save to JSONL."""
    if not out_jsonl_filepath.parent.exists():
        out_jsonl_filepath.parent.mkdir(parents=True, exist_ok=True)
    
    written_ids = set()
    
    # Skip reading existing file if force_reprocess is True
    if not force_reprocess and out_jsonl_filepath.exists():
        with open(out_jsonl_filepath, 'r', encoding='utf-8') as fin:
            for line in fin:
                try:
                    obj = json.loads(line)
                    written_ids.add(obj['id'])
                except json.JSONDecodeError:
                    continue

    # Filter out batches that are already completely processed
    batches_to_process = []
    for batch in batched_prompts:
        unprocessed_batch = [item for item in batch if item['id'] not in written_ids]
        if unprocessed_batch:
            batches_to_process.append(unprocessed_batch)

    logger.info(
        f"Skipping {len(written_ids)} already processed entries. "
        f"Processing {len(batches_to_process)} new batches."
    )

    if not batches_to_process:
        logger.info("No new batches to process.")
        return

    # Prepare arguments for thread_map
    args_list = [
        (batch, client, schema, max_retries, retry_delay)
        for batch in batches_to_process
    ]
    
    total_tokens = 0
    lock = Lock()
    
    with open(out_jsonl_filepath, 'a', encoding='utf-8-sig') as fout:
        # Process batches with progress bar
        for i in tqdm(range(0, len(args_list), max_workers), desc="Processing batches"):
            batch_args = args_list[i:i + max_workers]
            
            results = thread_map(
                process_batch_wrapper,
                batch_args,
                max_workers=min(max_workers, len(batch_args)),
                desc=f"Batch {i//max_workers+1}"
            )
            
            # Write out the results
            for batch_results, tokens_used in results:
                if batch_results:
                    with lock:
                        for result in batch_results:
                            # Ensure all required fields are present
                            result.setdefault('prompt_used', 'Unknown')
                            result.setdefault('original_poem', '')
                            result.setdefault('poet_name', '')
                            result.setdefault('corruption_type', '')
                            result.setdefault('data_row', '')
                            result.setdefault('prompt_num', '')
                            
                            fout.write(json.dumps(result, ensure_ascii=False) + '\n')
                        total_tokens += tokens_used
            
            fout.flush()
            logger.info(f"Tokens used so far: {total_tokens}")
            logger.info(f"Processed {i + len(batch_args)}/{len(args_list)} batches")

    logger.info(f"Finished processing. Output saved to {out_jsonl_filepath}")
    logger.info(f"Total estimated tokens used: {total_tokens}")

if __name__ == "__main__":
    # Load Gemini client
    gemini = load_gemini(VARS["gemini_key"])
    
    # Prepare batched prompts
    logger.info("Preparing batched prompts...")
    batched_prompts = prepare_batched_prompts(
        prompts_csv=PROMPT_PATH,
        data_csv=DATA_PATH,
        batch_size= BATCH_SIZE,  # Process 5 prompts per batch
        max_prompts=MAX_PROMPTS,  # Limit to first 4 prompt templates
        max_data_rows=MAX_DATA_ROWS  # Limit to first 2 data rows
    )
    
    logger.info(f"Prepared {len(batched_prompts)} batches for processing")
    
    # Get output path
    out_jsonl_filepath = get_output_path()
    
    # Run batch generation
    process_data_with_gemini_batch(
        client=gemini,
        schema=Data,
        batched_prompts=batched_prompts,
        out_jsonl_filepath=out_jsonl_filepath,
        max_workers=3,  # Reduced workers for batch processing
        force_reprocess=True
    )
    
    # Postprocess results
    postprocess_output()
    logger.info("Done postprocessing output data!")
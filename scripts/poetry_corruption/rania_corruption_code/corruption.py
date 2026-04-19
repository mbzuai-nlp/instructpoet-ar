"""
Fixed Single Poem Corruption Tester with 200 Workers
Fixes all parameter mismatches and JSON serialization issues
"""

import os
import json
import time
import pandas as pd
import numpy as np
import google.generativeai as genai
from datetime import datetime
import logging
from typing import Dict, List, Any, Optional
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Configuration
class WholeDatasetConfig:
    GEMINI_API_KEY = " "
    GEMINI_MODEL = "gemini-2.5-flash"
    DATASET_PATH = ""
    OUTPUT_DIR = ""
    
    TEST_MODE = False #you can test the methods and make sure that they are right before appling it on the entire data points
    
    TEST_POEMS_COUNT = 5  
    TEST_CORRUPTION_TYPES = None  
    MAX_WORKERS = 200  
    REQUESTS_PER_MINUTE = 60
    DELAY_BETWEEN_REQUESTS = 60 / REQUESTS_PER_MINUTE
    MAX_RETRIES = 3
    
    SAVE_INTERMEDIATE_EVERY = 1000  
    
    DEBUG_MODE = True 


def get_prompt(corruption_type, **kwargs):
    """Main prompt function that routes to specific prompt creators"""
    
    if corruption_type == "rhyme_structure_destruction":
        return create_rhyme_structure_destruction_prompt(**kwargs)
    elif corruption_type == "rhyme_content_corruption":
        return create_rhyme_content_corruption_prompt(**kwargs)
    elif corruption_type == "rhyme_substitution_corruption":
        return create_rhyme_substitution_corruption_prompt(**kwargs)
    elif corruption_type == "era_corruption":
        return create_era_corruption_prompt(**kwargs)
    elif corruption_type == "meter_destruction":
        return create_meter_destruction_prompt(**kwargs)
    elif corruption_type == "meter_transformation":
        return create_meter_transformation_prompt(**kwargs)
    elif corruption_type == "meter_inconsistency":
        return create_meter_inconsistency_prompt(**kwargs)
    else:
        raise ValueError(f"Unknown corruption type: {corruption_type}")


def create_rhyme_structure_destruction_prompt(poem_text, rhyme, **kwargs):
    """FIXED: Accepts all parameters your code sends"""
    prompt = f"""You are an expert Arabic poetry editor. Your task is to eliminate ALL rhyming patterns from the given poem while preserving the exact meaning and meter.

TASK: Rewrite the Arabic poem so that NO lines rhyme with each other.

CURRENT RHYME PATTERN: {rhyme}

REQUIREMENTS:
1. Replace ONLY the rhyming words at line endings.
2. Use synonyms or contextually fitting words with completely different sound patterns.
3. Preserve the exact meaning of each line.
4. Maintain the original Arabic meter (البحر الشعري).
5. Keep all grammatical structures intact.
6. Ensure NO lines rhyme anymore (target: unique ending for each line).

CRITICAL: Provide ONLY the rewritten poem. No explanations, analysis, or commentary.

Original Poem:
{poem_text}

Rewritten Poem:"""
    
    return prompt


def create_rhyme_content_corruption_prompt(poet_name, poet_era, poem_text, rhyme, **kwargs):
    """FIXED: Accepts all parameters including current_meter"""
    prompt = f"""You are an expert Arabic poetry editor. Your task is to CORRUPT the poem's content while maintaining PERFECT rhyming patterns.

TASK:
Rewrite the Arabic poem to keep the exact same rhymes but use completely inappropriate or absurd words that destroy the original meaning.

CURRENT RHYME PATTERN: {rhyme}
MAINTAIN THIS EXACT PATTERN: {rhyme}

REQUIREMENTS:
1. Keep the EXACT same rhyme scheme and rhyming sounds.
2. Replace rhyming words with inappropriate, absurd, or nonsensical alternatives that still rhyme perfectly.
3. Create maximum semantic chaos and contextual inappropriateness.
4. Use words that feel absurdly out of place given the original tone or theme.
5. Preserve the original rhyme pattern: {rhyme}.
6. Maintain correct Arabic grammar and meter.

CRITICAL:
Provide ONLY the rewritten poem. No explanations, analysis, or commentary.

Poet: {poet_name} ({poet_era})

Original Poem:
{poem_text}

Rewritten Poem:"""
    
    return prompt


def create_rhyme_substitution_corruption_prompt(poem_text, rhyme, **kwargs):
    """FIXED: Accepts all parameters your code sends"""
    prompt = f"""You are an expert Arabic poetry editor. Your task is to CHANGE the poem's rhyme from the original scheme to a completely new rhyme, while preserving meaning and meter.

CURRENT RHYME: {rhyme}
NEW RHYME: Choose ANY rhyme sound different from {rhyme} and apply it consistently.

REQUIREMENTS:
1. Replace ONLY the ending words so that all lines end with the new rhyme.
2. The new rhyme must be completely different from the original: {rhyme}.
3. Preserve the exact meaning of each line as much as possible.
4. Maintain the original Arabic meter (البحر الشعري).
5. Keep correct grammar and poetic flow.
6. Apply the new rhyme consistently to ALL lines.

CRITICAL:
Provide ONLY the rewritten poem with the new rhyme applied.
No explanations, analysis, or commentary.

Original Poem:
{poem_text}

Rewritten Poem:"""
    
    return prompt


def create_era_corruption_prompt(poem_title, poet_name, poet_era, poem_text, **kwargs):
    """FIXED: Era corruption that transforms poem text to different era"""
    import random
    
    # Arabic literary eras
    eras = [
        "العصر الجاهلي", "العصر الإسلامي", "عصر صدر الإسلام",
        "العصر الأموي", "العصر العباسي", "العصر الأندلسي", 
        "عصر المماليك", "العصر العثماني", "عصر النهضة",
        "العصر الحديث", "العصر المعاصر"
    ]
    
    # Select different era
    available_eras = [era for era in eras if era != poet_era]
    target_era = random.choice(available_eras) if available_eras else "العصر العباسي"
    
    prompt = f"""You are an expert Arabic poetry editor. Transform this poem from {poet_era} to {target_era} style.

TASK: Rewrite the poem content to reflect {target_era} themes and vocabulary while preserving rhyme and meter.

CURRENT ERA: {poet_era}
TARGET ERA: {target_era}

REQUIREMENTS:
1. Change vocabulary and references to match {target_era}
2. Preserve original rhyme scheme and meter
3. Keep same number of lines
4. Make content feel authentically from {target_era}

CRITICAL: Provide ONLY the rewritten poem. No explanations.

Poet: {poet_name}
Original Era: {poet_era} → Target Era: {target_era}

Original Poem:
{poem_text}

Rewritten Poem:"""
    
    return prompt


def create_meter_destruction_prompt(poet_name, poet_era, poem_text, meter, **kwargs):
    """FIXED: Accepts meter parameter from dataset"""
    prompt = f"""You are an expert Arabic poetry editor. Your task is to destroy the poetic meter completely while preserving the exact meaning and rhyming patterns.

TASK: Rewrite the Arabic poem to break all rhythmic patterns but keep the same meaning and rhymes.

CURRENT METER: {meter}
TARGET: No consistent meter (irregular rhythm)

REQUIREMENTS:
1. Preserve the exact meaning of each line
2. Keep ALL original rhyming words and patterns intact
3. Break the rhythmic meter by changing syllable counts and stress patterns
4. Add or remove syllables to create irregular rhythm
5. Mix different rhythmic patterns within the poem
6. Create jarring rhythmic inconsistencies

CRITICAL: Provide ONLY the rewritten poem. No explanations, analysis, or commentary.

Poet: {poet_name} ({poet_era})
Current Meter: {meter}

Original Poem:
{poem_text}

Rewritten Poem (broken meter, same meaning and rhymes):"""
    
    return prompt


def create_meter_transformation_prompt(poet_name, poet_era, poem_text, meter, target_meter, **kwargs):
    """FIXED: Uses meter parameter from dataset"""
    prompt = f"""You are an expert Arabic poetry editor. Your task is to transform the poem's meter from one بحر to a completely different بحر while preserving meaning and rhyme.

TASK: Change the poetic meter from the current بحر to the target بحر.

CURRENT METER: {meter}
TARGET METER: {target_meter}

TRANSFORMATION REQUIREMENTS:
1. Change the rhythmic pattern from {meter} to {target_meter}
2. Preserve the exact meaning of each line
3. Keep ALL original rhyming words and patterns intact
4. Adapt syllable counts and stress patterns to match {target_meter}
5. Ensure consistent application of the new meter throughout
6. Maintain natural flow in the new rhythmic pattern

CRITICAL: Provide ONLY the rewritten poem in the new meter. No explanations, analysis, or commentary.

Poet: {poet_name} ({poet_era})

Original Poem:
{poem_text}

Rewritten Poem (in {target_meter}):"""
    
    return prompt


def create_meter_inconsistency_prompt(poet_name, poet_era, poem_text, meter, **kwargs):
    """FIXED: Uses meter parameter from dataset"""
    prompt = f"""You are an expert Arabic poetry editor. Your task is to create meter chaos by making each line follow a different rhythmic pattern while preserving meaning and rhyme.

TASK: Rewrite the poem so each line has a different meter, creating maximum rhythmic inconsistency.

CURRENT METER: {meter} (consistent throughout)
TARGET: Different meter for each line (maximum inconsistency)

REQUIREMENTS:
1. Preserve the exact meaning of each line
2. Keep ALL original rhyming words and patterns intact
3. Use a different Arabic meter for each line
4. Create jarring rhythmic transitions between lines
5. Mix various بحور العروض randomly throughout the poem

CRITICAL: Provide ONLY the rewritten poem. No explanations, analysis, or commentary.

Poet: {poet_name} ({poet_era})

Original Poem:
{poem_text}

Rewritten Poem (different meter per line):"""
    
    return prompt


def get_available_corruption_types():
    """Get list of available corruption types"""
    return [
        "rhyme_structure_destruction",
        "rhyme_content_corruption", 
        "rhyme_substitution_corruption",
        "era_corruption",
        "meter_destruction",
        "meter_transformation",
        "meter_inconsistency"
    ]


class WholeDatasetTester:
    """Test all corruption types on the entire dataset with 200 workers"""
    
    def __init__(self):
        self.setup_output_directory()
        self.setup_logging()
        self.setup_gemini()
        
    def setup_output_directory(self):
        """Create output directory"""
        os.makedirs(WholeDatasetConfig.OUTPUT_DIR, exist_ok=True)
        print(f"Output directory: {WholeDatasetConfig.OUTPUT_DIR}")
    
    def setup_logging(self):
        """Setup logging"""
        log_file = os.path.join(WholeDatasetConfig.OUTPUT_DIR, 'whole_dataset_corruption.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_gemini(self):
        """Setup Gemini API"""
        genai.configure(api_key=WholeDatasetConfig.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(WholeDatasetConfig.GEMINI_MODEL)
    
    def load_dataset(self) -> Optional[pd.DataFrame]:
        """Load poetry dataset"""
        try:
            df = pd.read_csv(WholeDatasetConfig.DATASET_PATH)
            self.logger.info(f"Successfully loaded {len(df)} poems from dataset")
            
            # Filter for poems with complete data
            complete_poems = df.dropna(subset=['poem_text', 'poet_name', 'poet_era', 'meter', 'rhyme_letter'])
            self.logger.info(f"Found {len(complete_poems)} poems with complete data")
            
            # Apply test mode limitations if enabled
            if WholeDatasetConfig.TEST_MODE:
                complete_poems = complete_poems.head(WholeDatasetConfig.TEST_POEMS_COUNT)
                self.logger.info(f" TEST MODE: Limited to {len(complete_poems)} poems for testing")
            else:
                self.logger.info(f"FULL MODE: Processing all {len(complete_poems)} poems")
            
            return complete_poems
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            return None
    
    def prepare_all_poems(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Prepare all poems for processing - FIXED: Added poem_id"""
        poems = []
        
        for idx, row in df.iterrows():
            poem_data = {
                "poet_name": str(row.get('poet_name', 'Unknown Poet')),
                "poet_era": str(row.get('poet_era', 'Unknown Era')),
                "poem_title": str(row.get('poem_title', 'Untitled Poem')),
                "poem_text": str(row.get('poem_text', '')),
                "rhyme": str(row.get('rhyme_letter', 'Unknown')),
                "meter": str(row.get('meter', 'Unknown')),
                "row_index": idx
            }
            poems.append(poem_data)
        
        self.logger.info(f"Prepared {len(poems)} poems for corruption testing")
        return poems
    
    def call_gemini_api(self, prompt: str) -> str:
        """Call Gemini API with retry logic and debugging - FIXED: Better error handling"""
        for attempt in range(WholeDatasetConfig.MAX_RETRIES):
            try:
                if WholeDatasetConfig.DEBUG_MODE and attempt == 0:
                    self.logger.info(f"API call attempt {attempt + 1}, prompt length: {len(prompt)} chars")
                
                response = self.model.generate_content(prompt)
                time.sleep(WholeDatasetConfig.DELAY_BETWEEN_REQUESTS)
                
                if hasattr(response, 'text') and response.text:
                    if WholeDatasetConfig.DEBUG_MODE:
                        self.logger.info(f"API call successful, response length: {len(response.text)} chars")
                    return response.text
                else:
                    error_msg = f"Empty response from API (attempt {attempt + 1})"
                    self.logger.warning(error_msg)
                    if attempt == WholeDatasetConfig.MAX_RETRIES - 1:
                        return f"ERROR: {error_msg}"
                    continue
            except Exception as e:
                error_msg = str(e)
                self.logger.warning(f"API call failed (attempt {attempt + 1}): {error_msg}")
                
                if WholeDatasetConfig.DEBUG_MODE:
                    self.logger.warning(f"Full error details: {type(e).__name__}: {error_msg}")
                
                if attempt < WholeDatasetConfig.MAX_RETRIES - 1:
                    wait_time = 2 ** attempt
                    if WholeDatasetConfig.DEBUG_MODE:
                        self.logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)  # Exponential backoff
                else:
                    final_error = f"ERROR: Failed after {WholeDatasetConfig.MAX_RETRIES} attempts - {error_msg}"
                    if WholeDatasetConfig.DEBUG_MODE:
                        self.logger.error(f"All retry attempts failed: {final_error}")
                    return final_error
    
    def generate_prompt(self, poem_data: Dict[str, Any], corruption_type: str) -> str:
        """Generate prompt for specific corruption type"""
        kwargs = {
            "poem_text": poem_data["poem_text"],
            "poet_name": poem_data["poet_name"],
            "poet_era": poem_data["poet_era"],
            "rhyme": poem_data["rhyme"],
            "meter": poem_data["meter"]  
        }
        
       
        if corruption_type == "era_corruption":
            kwargs = {
                "poem_title": poem_data["poem_title"],
                "poet_name": poem_data["poet_name"],
                "poet_era": poem_data["poet_era"],
                "poem_text": poem_data["poem_text"]
            }
        elif corruption_type == "meter_transformation":
            meters = ["الكامل", "الطويل", "البسيط", "الوافر", "الرجز", "المتقارب"]
            available_meters = [m for m in meters if m != poem_data["meter"]]
            kwargs["target_meter"] = random.choice(available_meters) if available_meters else "الكامل"
        
        return get_prompt(corruption_type, **kwargs)
    
    def analyze_corruption_result(self, poem_data: Dict[str, Any], response: str, corruption_type: str) -> Dict[str, Any]:
        """Basic analysis of corruption result"""
        if response.startswith("ERROR:"):
            return {
                "success": False,
                "error": response,
                "analysis_type": "error"
            }
        
        # Basic success metrics
        analysis = {
            "success": True,
            "analysis_type": corruption_type,
            "response_length": len(response),
            "original_length": len(poem_data["poem_text"]),
            "length_change": len(response) - len(poem_data["poem_text"]),
            "has_arabic_text": any('\u0600' <= c <= '\u06FF' for c in response)
        }
        
        if "rhyme" in corruption_type:
            analysis.update(self.analyze_rhyme_changes(poem_data["poem_text"], response))
        elif corruption_type == "era_corruption":
            analysis.update(self.analyze_era_response(response))
        elif "meter" in corruption_type:
            analysis.update(self.analyze_meter_changes(poem_data["poem_text"], response))
        
        return analysis
    
    def analyze_rhyme_changes(self, original: str, corrupted: str) -> Dict[str, Any]:
        """Analyze rhyme changes"""
        try:
            original_lines = [line.strip() for line in original.split('\n') if line.strip()]
            corrupted_lines = [line.strip() for line in corrupted.split('\n') if line.strip()]
            
            return {
                "original_line_count": len(original_lines),
                "corrupted_line_count": len(corrupted_lines),
                "line_count_preserved": len(original_lines) == len(corrupted_lines),
                "rhyme_analysis": "basic_line_comparison"
            }
        except:
            return {"rhyme_analysis": "failed"}
    
    def analyze_era_response(self, response: str) -> Dict[str, Any]:
        """Analyze era corruption response"""
        try:
            if response.strip().startswith('{'):
                data = json.loads(response)
                return {
                    "json_valid": True,
                    "has_corrupted_era": "corrupted_era" in data,
                    "era_value": data.get("corrupted_era", "")
                }
            else:
                return {
                    "json_valid": False,
                    "response_preview": response[:100]
                }
        except:
            return {"json_valid": False, "json_parse_error": True}
    
    def analyze_meter_changes(self, original: str, corrupted: str) -> Dict[str, Any]:
        """Analyze meter changes"""
        try:
            orig_words = len(original.split())
            corr_words = len(corrupted.split())
            
            return {
                "original_word_count": orig_words,
                "corrupted_word_count": corr_words,
                "word_count_change": corr_words - orig_words,
                "meter_analysis": "basic_word_comparison"
            }
        except:
            return {"meter_analysis": "failed"}
    
    def process_single_corruption(self, corruption_task):
        """Process a single corruption task - for parallel execution - FIXED: Better error handling"""
        poem_data, corruption_type = corruption_task
        
        try:
            prompt = self.generate_prompt(poem_data, corruption_type)
            
            response = self.call_gemini_api(prompt)
            
            analysis = self.analyze_corruption_result(poem_data, response, corruption_type)
            
            result = {
                "poem_id": poem_data.get("poem_id", f"poem_{poem_data.get('row_index', 'unknown')}"),
                "poet_name": poem_data["poet_name"],
                "poet_era": poem_data["poet_era"],
                "poem_title": poem_data["poem_title"],
                "original_poem": poem_data["poem_text"],
                "original_rhyme": poem_data["rhyme"],
                "original_meter": poem_data["meter"],
                
                "corruption_type": corruption_type,
                "corrupted_poem": response,
                "success": analysis.get("success", False),
                "timestamp": datetime.now().isoformat(),
                
                **{f"analysis_{k}": self.convert_numpy_types(v) for k, v in analysis.items() if k != "success"}
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing {corruption_type} for poem {poem_data.get('poem_id', 'unknown')}: {str(e)}")
            
            error_result = {
                "poem_id": poem_data.get("poem_id", f"poem_{poem_data.get('row_index', 'unknown')}"),
                "poet_name": poem_data["poet_name"],
                "corruption_type": corruption_type,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "original_poem": poem_data["poem_text"],
                "corrupted_poem": f"ERROR: {str(e)}"
            }
            return error_result
    
    def convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):  
            return obj.item()
        else:
            return obj
    
    def test_all_corruptions_on_whole_dataset(self, poems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Test all corruption types on the entire dataset in parallel"""
        if WholeDatasetConfig.TEST_MODE and WholeDatasetConfig.TEST_CORRUPTION_TYPES:
            corruption_types = WholeDatasetConfig.TEST_CORRUPTION_TYPES
            self.logger.info(f" TEST MODE: Testing specific corruption types: {corruption_types}")
        else:
            corruption_types = get_available_corruption_types()
            if WholeDatasetConfig.TEST_MODE:
                self.logger.info(f" TEST MODE: Testing all {len(corruption_types)} corruption types")
        
        all_tasks = []
        for poem_data in poems:
            for corruption_type in corruption_types:
                all_tasks.append((poem_data, corruption_type))
        
        total_tasks = len(all_tasks)
        mode = "TEST MODE" if WholeDatasetConfig.TEST_MODE else "FULL MODE"
        self.logger.info(f"🎯 {mode}: Created {total_tasks} tasks ({len(poems)} poems × {len(corruption_types)} corruption types)")
        self.logger.info(f"Processing with {WholeDatasetConfig.MAX_WORKERS} workers")
        
        results = []
        completed_count = 0
        
        with ThreadPoolExecutor(max_workers=WholeDatasetConfig.MAX_WORKERS) as executor:
            future_to_task = {
                executor.submit(self.process_single_corruption, task): task 
                for task in all_tasks
            }
            
            for future in tqdm(as_completed(future_to_task), total=total_tasks, desc="Processing corruptions"):
                try:
                    result = future.result()
                    results.append(result)
                    completed_count += 1
                    
                    if (not WholeDatasetConfig.TEST_MODE and completed_count % WholeDatasetConfig.SAVE_INTERMEDIATE_EVERY == 0) or \
                       (WholeDatasetConfig.TEST_MODE and total_tasks > 50 and completed_count % 25 == 0):
                        self.save_intermediate_results(results, completed_count)
                    
                    log_interval = 10 if WholeDatasetConfig.TEST_MODE else 50
                    if completed_count % log_interval == 0:
                        success_rate = sum(1 for r in results if r.get("success", False)) / len(results) * 100
                        self.logger.info(f"Progress: {completed_count}/{total_tasks} ({success_rate:.1f}% success rate)")
                    
                except Exception as e:
                    self.logger.error(f"Task failed with exception: {e}")
        
        self.logger.info(f" Completed all {total_tasks} corruption tasks")
        return results
    
    def save_intermediate_results(self, results: List[Dict[str, Any]], count: int):
        """Save intermediate results to prevent data loss"""
        try:
            clean_results = []
            for result in results:
                clean_result = {}
                for key, value in result.items():
                    clean_result[key] = self.convert_numpy_types(value)
                clean_results.append(clean_result)
            
            df = pd.DataFrame(clean_results)
            filename = f"intermediate_results_{count}.csv"
            filepath = os.path.join(WholeDatasetConfig.OUTPUT_DIR, filename)
            df.to_csv(filepath, index=False, encoding='utf-8')
            
            self.logger.info(f"Saved intermediate results: {count} tasks completed")
        except Exception as e:
            self.logger.error(f"Error saving intermediate results: {e}")
    
    def save_results_to_csv(self, results: List[Dict[str, Any]]) -> str:
        """Save final results to CSV file - Simple filename with mode indicator"""
        clean_results = []
        for result in results:
            clean_result = {}
            for key, value in result.items():
                clean_result[key] = self.convert_numpy_types(value)
            clean_results.append(clean_result)
        
        df = pd.DataFrame(clean_results)
        
        if WholeDatasetConfig.TEST_MODE:
            filename = "test_corruption_results.csv"
        else:
            filename = "whole_dataset_corruption_results.csv"
            
        filepath = os.path.join(WholeDatasetConfig.OUTPUT_DIR, filename)
        
        df.to_csv(filepath, index=False, encoding='utf-8')
        self.logger.info(f"Final results saved to: {filepath}")
        
        return filepath
    
    def print_summary(self, results: List[Dict[str, Any]], total_poems: int):
        """Print test summary"""
        successful = sum(1 for r in results if r.get("success", False))
        total = len(results)
        
        print("\n" + "="*80)
        print("WHOLE DATASET CORRUPTION TEST SUMMARY")
        print("="*80)
        print(f"Total poems processed: {total_poems}")
        print(f"Total corruption tasks: {total}")
        print(f"Successful corruptions: {successful}")
        print(f"Failed corruptions: {total - successful}")
        print(f"Success rate: {(successful/total*100):.1f}%")
        print(f"Workers used: {WholeDatasetConfig.MAX_WORKERS}")
        
        corruption_stats = {}
        for result in results:
            corruption_type = result.get("corruption_type", "unknown")
            if corruption_type not in corruption_stats:
                corruption_stats[corruption_type] = {"total": 0, "successful": 0}
            
            corruption_stats[corruption_type]["total"] += 1
            if result.get("success", False):
                corruption_stats[corruption_type]["successful"] += 1
        
        print(f"\nResults by corruption type:")
        print("-" * 60)
        for corruption_type, stats in corruption_stats.items():
            success_rate = (stats["successful"] / stats["total"] * 100) if stats["total"] > 0 else 0
            print(f"{corruption_type}: {stats['successful']}/{stats['total']} ({success_rate:.1f}%)")
        
        print(f"\nResults saved in: {WholeDatasetConfig.OUTPUT_DIR}")
        
        if successful == 0:
            print("\n ALL TASKS FAILED! Checking for common issues...")
            error_samples = [r.get("error", r.get("ai_response", "")) for r in results[:3] if not r.get("success", False)]
            for i, error in enumerate(error_samples, 1):
                if error and error.startswith("ERROR:"):
                    print(f"Sample error {i}: {error[:100]}...")
    
    def run_test(self):
        """Run the complete whole dataset corruption test"""
        self.logger.info(f"Starting Whole Dataset Corruption Test with {WholeDatasetConfig.MAX_WORKERS} workers")
        
        df = self.load_dataset()
        if df is None:
            self.logger.error("Failed to load dataset. Exiting.")
            return
        
        poems = self.prepare_all_poems(df)
        if not poems:
            self.logger.error("No poems to process. Exiting.")
            return
        
        results = self.test_all_corruptions_on_whole_dataset(poems)
        
        if not results:
            self.logger.error("No results generated. Exiting.")
            return
        
        csv_path = self.save_results_to_csv(results)
        
        self.print_summary(results, len(poems))
        
        self.logger.info("Whole dataset corruption test completed successfully!")
        return results, csv_path


def main():
    """Main execution function"""
    tester = WholeDatasetTester()
    return tester.run_test()


if __name__ == "__main__":
    main()

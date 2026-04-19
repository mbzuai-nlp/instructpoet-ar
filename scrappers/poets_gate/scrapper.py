import os
import time
import logging
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

# Constants
POEM_URL = "https://poetsgate.com/poem.php?pm={}"
POET_URL = "https://poetsgate.com/poet.php?pt={}"
SAVE_DIR = "/path/to/data/raw/poets_gate"
POEMS_DIR = os.path.join(SAVE_DIR, "poems")
POETS_DIR = os.path.join(SAVE_DIR, "poets")
FAILED_POEMS_FILE = "failed_poems.txt"
FAILED_POETS_FILE = "failed_poets.txt"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/113.0.0.0 Safari/537.36"
}

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("scraper.log"),
        logging.StreamHandler()
    ]
)

# Ensure output directories exist
os.makedirs(POEMS_DIR, exist_ok=True)
os.makedirs(POETS_DIR, exist_ok=True)

# Create a requests session with retry strategy
def create_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        backoff_factor=1
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.headers.update(HEADERS)
    return session

def download_item(item_id, url_template, save_dir, item_type, session):
    file_path = os.path.join(save_dir, f"{item_type}_{item_id}.html")
    if os.path.exists(file_path):
        return None  # Already downloaded

    url = url_template.format(item_id)
    try:
        response = session.get(url, timeout=10)
        if response.status_code == 200:
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(response.text)
            return None
        elif response.status_code == 404:
            return None
        else:
            logging.warning(f"{item_type.capitalize()} {item_id} returned status code {response.status_code}")
            return item_id
    except requests.RequestException as e:
        logging.error(f"Error downloading {item_type} {item_id}: {e}")
        return item_id

def download_items(start_id, end_id, url_template, save_dir, item_type, failed_file, max_workers=10):
    session = create_session()
    failed_ids = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_item, item_id, url_template, save_dir, item_type, session): item_id
            for item_id in range(start_id, end_id)
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Downloading {item_type}s"):
            try:
                failed_id = future.result()
                if failed_id is not None:
                    failed_ids.append(failed_id)
            except Exception as e:
                item_id = futures[future]
                logging.error(f"Unhandled exception on {item_type} {item_id}: {e}")
                failed_ids.append(item_id)

    if failed_ids:
        with open(failed_file, "w") as f:
            for pid in failed_ids:
                f.write(f"{pid}\n")
        logging.info(f"Saved {len(failed_ids)} failed {item_type} IDs to {failed_file}")

if __name__ == "__main__":
    start = time.time()

    logging.info("Starting download of poems...")
    download_items(0, 221700, POEM_URL, POEMS_DIR, "poem", FAILED_POEMS_FILE, max_workers=24)

    logging.info("Starting download of poets...")
    download_items(0, 4400, POET_URL, POETS_DIR, "poet", FAILED_POETS_FILE, max_workers=24)

    logging.info(f"Completed all downloads in {time.time() - start:.2f} seconds")

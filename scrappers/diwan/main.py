import os
import time
import logging
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

# Constants
BASE_URL = "https://www.aldiwan.net/poem{}.html"
SAVE_DIR = "poems_html"
FAILED_FILE = "failed_poems.txt"
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

# Ensure output directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

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


def download_poem(poem_id, session):
    file_path = os.path.join(SAVE_DIR, f"poem_{poem_id}.html")
    if os.path.exists(file_path):
        return None  # Already downloaded

    url = BASE_URL.format(poem_id)
    try:
        response = session.get(url, timeout=10)
        if response.status_code == 200:
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(response.text)
            return None  # Success
        elif response.status_code == 404:
            return None  # Not found, but not a failure worth retrying
        else:
            logging.warning(f"Poem {poem_id} returned status code {response.status_code}")
            return poem_id
    except requests.RequestException as e:
        logging.error(f"Error downloading poem {poem_id}: {e}")
        return poem_id


def download_poems(start_id, end_id, max_workers=10):
    session = create_session()
    failed_ids = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_poem, poem_id, session): poem_id
            for poem_id in range(start_id, end_id)
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading poems"):
            try:
                failed_id = future.result()
                if failed_id is not None:
                    failed_ids.append(failed_id)
            except Exception as e:
                poem_id = futures[future]
                logging.error(f"Unhandled exception on poem {poem_id}: {e}")
                failed_ids.append(poem_id)

    # Save failed IDs
    if failed_ids:
        with open(FAILED_FILE, "w") as f:
            for pid in failed_ids:
                f.write(f"{pid}\n")
        logging.info(f"Saved {len(failed_ids)} failed IDs to {FAILED_FILE}")


if __name__ == "__main__":
    start = time.time()
    download_poems(0, 130000, max_workers=24)
    logging.info(f"Completed in {time.time() - start:.2f} seconds")

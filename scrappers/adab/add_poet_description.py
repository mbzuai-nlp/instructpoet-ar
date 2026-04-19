import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Load JSONL data
data_path = "/path/to/data/adab.jsonl"
with open(data_path, 'r', encoding='utf-8') as file:
    data = [json.loads(line) for line in file]

# Create DataFrame
df = pd.DataFrame(data)

# Ensure 'poet_description' column exists
if 'poet_description' not in df.columns:
    df['poet_description'] = ''
else:
    df['poet_description'] = df['poet_description'].fillna('')

# Function to fetch poet description from HTML
def fetch_description(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        p_tag = soup.find('p', class_='p-about p-about_biography')
        return url, p_tag.get_text(separator="\n", strip=True) if p_tag else ''
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return url, ''

# Get unique poet_page_urls
unique_urls = df['poet_page_url'].dropna().unique()

# Dictionary to store fetched descriptions
poet_descriptions = {}

# Use ThreadPoolExecutor to parallelize the requests
with ThreadPoolExecutor(max_workers=32) as executor:
    futures = {executor.submit(fetch_description, url): url for url in unique_urls}
    for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching poet descriptions"):
        url, desc = future.result()
        poet_descriptions[url] = desc

# Map the fetched descriptions back to the DataFrame
df['poet_description'] = df['poet_page_url'].map(poet_descriptions)

# Optionally save updated data
output_path = Path(__file__).resolve().parent / "adab_with_descriptions.jsonl"
with open(output_path, 'w', encoding='utf-8') as f:
    for record in df.to_dict(orient='records'):
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

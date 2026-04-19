from bs4 import BeautifulSoup
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import re
import json
from typing import List, Dict, Optional
from multiprocessing import Pool, cpu_count

def remove_arabic_diacritics(text: str) -> str:
    arabic_diacritics = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    return re.sub(arabic_diacritics, '', text)

def extract_poem_verses(soup: BeautifulSoup) -> str:
    poem_div = soup.find('div', id='DivfinalPoem')
    if not poem_div:
        return ""
    verse_divs = poem_div.find_all('div', class_='pmContainer')
    verses = [div.get_text(strip=True) for div in verse_divs if div.get_text(strip=True)]
    return '\n'.join(verses)

def extract_tags(soup: BeautifulSoup) -> List[str]:
    tips_div = soup.find('div', class_='tips')
    if not tips_div:
        return []
    return [a.get_text(strip=True) for a in tips_div.find_all('a')]

def extract_metadata_from_breadcrumbs(soup: BeautifulSoup) -> Dict[str, Optional[str]]:
    breadcrumb_div = soup.find('div', class_='col-lg-12 text-right')
    if not breadcrumb_div:
        return {"era": None, "country": None, "poet": None, "poem_title": None}
    
    elements = list(breadcrumb_div.stripped_strings)
    
    # Heuristically assign values by position
    era = elements[0] if len(elements) > 0 else None
    country = elements[2] if len(elements) > 2 else None
    poet = elements[4] if len(elements) > 4 else None
    poem_title = elements[6] if len(elements) > 6 else None

    return {
        "era": era,
        "country": country,
        "poet": poet,
        "poem_title": poem_title
    }

def extract_poet_description(soup: BeautifulSoup) -> Optional[str]:
    container = soup.find('div', class_='col-lg-5 col-md-6 col-12 float-left mosahmat_block_top')
    if container:
        h4_tag = container.find('h4')
        if h4_tag:
            return h4_tag.get_text(strip=True)
    return None

def extract_poet_english_id(soup: BeautifulSoup) -> str:
    tag = soup.find("p", class_="text-center mb-1 text-slug")
    if tag and tag.get_text(strip=True).endswith("@"):
        return tag.get_text(strip=True).rstrip("@")
    return "Unknown"

def extract_fallback_title_and_poet(title_tag: Optional[str]) -> Dict[str, str]:
    if not title_tag:
        return {"fallback_poet": "Unknown", "fallback_title": "Unknown"}

    parts = title_tag.split("-")
    if len(parts) >= 3:
        return {"fallback_title": parts[0], "fallback_poet": parts[1]}
    return {"fallback_title": "Unknown", "fallback_poet": "Unknown"}

def parse_html_file(filepath: Path) -> Dict:
    with filepath.open(encoding="utf-8") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.title.get_text(strip=True) if soup.title else None
    fallback = extract_fallback_title_and_poet(title_tag)

    metadata = extract_metadata_from_breadcrumbs(soup)
    poem_text = extract_poem_verses(soup)
    tags = extract_tags(soup)
    poet_description = extract_poet_description(soup)
    poet_english_id = extract_poet_english_id(soup)

    # Extract poem URL from canonical link
    canonical_link = soup.find("link", rel="canonical")
    poem_url = canonical_link["href"] if canonical_link and canonical_link.has_attr("href") else None


    result = {
        "title": metadata.get("poem_title") or fallback["fallback_title"],
        "poem_url": poem_url,
        "poet_page_url": None,
        "id": filepath.name,
        "poet": metadata.get("poet") or fallback["fallback_poet"],
        "poem": poem_text,
        "tags": tags,
        "poem_no_diacritics": remove_arabic_diacritics(poem_text),
        "era": metadata.get("era"),
        "country": metadata.get("country"),
        "poem_title": metadata.get("poem_title"),
        "poet_english_id": poet_english_id,
        "poet_description": poet_description
    }

    for k, v in result.items():
        if not isinstance(v, (str, list, dict, type(None))):
            raise ValueError(f"Key '{k}' contains a non-serializable object: {type(v)}")

    return result

def main():
    path = Path("../../data/raw/poets_gate/poems")  # Adjust as needed
    html_files = sorted(path.glob("*.html"))

    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(parse_html_file, html_files, chunksize=10), total=len(html_files), desc="Parsing poems"))

    df = pd.DataFrame(results)
    assert len(df) == len(html_files), "Mismatch between files and extracted data"

    df.to_json("../../data/poets_gate.jsonl", orient="records", lines=True, force_ascii=False)

if __name__ == "__main__":
    main()

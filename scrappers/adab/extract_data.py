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

def extract_tags(soup: BeautifulSoup) -> List[str]:
    tips_div = soup.find('div', class_='tips')
    if not tips_div:
        return []
    return [a.get_text(strip=True) for a in tips_div.find_all('a')]

def extract_poet_info(soup: BeautifulSoup) -> Dict[str, Optional[str]]:
    result = {"poet": None, "country": None, "poet_page_url": None}
    h4 = soup.find("h4", class_="profile-name")
    if h4:
        a_tag = h4.find("a", class_="profile_url")
        span_tag = h4.find("span", class_="span-contry-post")
        if a_tag:
            result["poet"] = a_tag.get_text(strip=True)
            result["poet_page_url"] = a_tag.get("href")
        if span_tag:
            result["country"] = span_tag.get_text(strip=True).replace("\u200f", "")
    return result

def extract_poem_title(soup: BeautifulSoup) -> Optional[str]:
    h2 = soup.find("h2", class_="post-title")
    return h2.get_text(strip=True) if h2 else None

def extract_poem_text_from_textarea(soup: BeautifulSoup) -> Optional[str]:
    textarea = soup.find("textarea", id="post_content")
    return textarea.get_text(strip=True) if textarea else ""

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

    poet_info = extract_poet_info(soup)
    poem_title = extract_poem_title(soup) or fallback["fallback_title"]
    poem_text = extract_poem_text_from_textarea(soup)
    tags = extract_tags(soup)
    poet_description = extract_poet_description(soup)
    poet_english_id = extract_poet_english_id(soup)

    canonical_link = soup.find("link", rel="canonical")
    poem_url = canonical_link["href"] if canonical_link and canonical_link.has_attr("href") else None

    result = {
        "title": poem_title,
        "poem_url": poem_url,
        "poet_page_url": poet_info.get("poet_page_url"),
        "id": filepath.name,
        "poet": poet_info.get("poet") or fallback["fallback_poet"],
        "poem": poem_text,
        "tags": tags,
        "poem_no_diacritics": remove_arabic_diacritics(poem_text),
        "era": None,
        "country": poet_info.get("country"),
        "poem_title": poem_title,
        "poet_english_id": poet_english_id,
        "poet_description": poet_description
    }

    for k, v in result.items():
        if not isinstance(v, (str, list, dict, type(None))):
            raise ValueError(f"Key '{k}' contains a non-serializable object: {type(v)}")

    return result

def main():
    path = Path("../../data/raw/adab/poems")  # Adjust as needed
    html_files = sorted(path.glob("*.html"))

    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(parse_html_file, html_files, chunksize=10), total=len(html_files), desc="Parsing poems"))

    df = pd.DataFrame(results)
    assert len(df) == len(html_files), "Mismatch between files and extracted data"

    df.to_json("../../data/adab.jsonl", orient="records", lines=True, force_ascii=False)

if __name__ == "__main__":
    main()

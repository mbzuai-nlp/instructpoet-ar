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


from bs4 import BeautifulSoup
from typing import Optional

from bs4 import BeautifulSoup
from typing import Optional

def extract_poem(soup: BeautifulSoup) -> Optional[str]:
    poem_div = soup.find('div', id='poem_content')
    if not poem_div:
        return None

    # Case 1: Poem with half verses in <h3>
    h3_lines = poem_div.find_all('h3')
    if h3_lines:
        half_verses = [line.get_text(strip=True) for line in h3_lines]
        verses = []
        for i in range(0, len(half_verses), 2):
            first_half = half_verses[i]
            second_half = half_verses[i+1] if i+1 < len(half_verses) else ''
            verses.append(f"{first_half}\t{second_half}")
        return '\n'.join(verses)

    # Case 2: Poem with full text in a single <h4> and <br> tags between lines
    h4_tag = poem_div.find('h4')
    if h4_tag:
        # Use .stripped_strings to correctly handle <br> separators
        lines = list(h4_tag.stripped_strings)
        return '\n'.join(lines)

    return None

html = '''
<div class="bet-1 row pt-0 px-5 pb-4 justify-content-center " id="poem_content">
                    <h4>إنا لجهال من الجهال<br>
حيث نحيي طلل الأطلال<br>
بالأرسط المثل من الأمثال<br>
باليةً في دمنٍ بوالِ<br>
محلةً من أنسٍ حلال<br>
تعرفُ فيها منزل النزال<br>
ومثلاً في خلدٍ مثال<br>
ورقاً تصلين بنار الصالي<br>
يحد سيل الأبطح السيال<br>
عنها وعن أطحل كالطحال<br>
أحوى القرا دونَ الصعيدِ العالي<br>
مثلُ الهلالِ ليلةَ الهلالِ<br>
وقد عرفنا بعرى الأبطالِ<br>
مراكز الخطيةِ الطوال<br>
ومربط الفحالِ والفحال<br>
ينحتن جلَّ الليل في الأجلالِ<br>
مراً ويصهلنَ إلى الصهال<br>
بنات ذي الطوقِ وذي العقالِ<br>
فاستبدلت والدهرُ ذو إبدالِ<br>
كل جفولٍ بالحصى مجفالِ<br>
تجرُّ أذيالاً على أذيال<br>
تتركُ حالَ التربِ كلَّ حالِ<br>
كأنما غربلَ بالغربالِ<br>
وصابهُ من لجبٍ جلجالِ<br>
بالوابلِ الراعِدِ والهَطَّالِ<br>
بديمٍ منهُ وباحتفالِ<br>
وهي الروايا مُرسِل العزالي<br>
فالربدُ منه بعشيبٍ خالي<br>
ترعى كهمالٍ من الهمال<br>
جرب طلاها بالكحيل الطالي<br>
منها رئالٌ وأبو رئال<br>
كالحبشي التقف في أسمال<br>
تبري له حرجاءُ كالخيال<br>
فهن بالروض والأقبال<br>
كالنعم الجلة والفِصال<br>
في خاذلات البقر الخذال<br>
يزجين أطفالاً إلى أطفال<br>
فالعين من نتجٍ ومن حيال<br>
يعلفن حولي لهق ذيال<br>
أعين يمشي مشية المختال<br>
ورد السراويل رخي البال<br>
لابس سربالٍ على سربال<br>
ثوبين من طر ومن إنسال<br>
يطيرُ عن ذاكَ الدخيل العالي<br>
ينطف روقاهُ من الطلالِ<br>
على جبين وعلى قذال<br>
وقد نرى من أهلها الأهال<br>
غوالياً في اليمند الغوالي<br>
برج العيون وعثة الأكفال<br>
كأن تحتَ الأزرفي الحجال<br>
منهن أنقاءً من الرمالِ<br>
نيطت بأحقي بدنٍ ثقال<br>
يخرس عنها جرسُ الخلخال<br>
بدنٌ جرى في أسؤقٍ خدالِ<br>
من خلقِ هيفٍ ألفِ الأظلال<br>
قطف السرى كاسيةٍ حوالي<br>
مغموسةٍ في الحسن والجَمالِ<br>
يضحكن عن أبيض كالسيالِ<br>
بثلجِ ماءِ البردِ الزلالِ<br>
لا يتنولون من النوالِ<br>
لمن تعرضن من الرجال<br>
إن لم يكن من نائِلٍ حَلالِ<br>
إلا بداءِ الخبلِ والسلالِ<br>
يعطين من صافحن بالدلال<br>
ملسا كأولادِ النقى المنهالِ<br>
تلوي به القربَ على ميالِ<br>
جعدٍ كوحفِ العنبِ المندال<br>
قد كان يهوى مثلها أمثالي<br>
حتى رأى الغالي وغيرُ الغالي<br>
شيباً حفافي صلعِ زُلالِ<br>
فانقطعَ الوصل من الوصالِ<br>
وزادني خبلاً من الخبالِ<br>
إني أبالي وهي لا تُبالي<br>
يا عجباً للأشمط البجالِ<br>
علامَ يُقلى وهو غيرُ قالِ<br>
لما اراحَ الجذبَ بالهُزالِ<br>
واختلَّ من لم يكُ ذا اختلالِ<br>
وصلدَ المسؤولُ بالسؤالِ<br>
واعتل من لم يَكُ ذا اعتِلالِ<br>
باتَت همومُ الصدرِ في بَلبالِ<br>
خصمينِ بينَ الصُلحِ والقِتالِ<br>
في ليلةٍ طالت من الليالي<br>
ثم علا هَمِّي وهمي عالِ<br>
فاخترتُ والمختارُ غيرُ آلِ<br>
خليفةَ اللَهِ الذي يُوالي<br>
إليكَ خُضنا الليلَ ذا الأهوالِ<br>
بالعِيسِ من مُنقَطِع الشمالِ<br>
يَرمُلنَ في الآلِ وغيرِ الآلِ<br>
مُعصَوصِياتٍ رَمَلَ السَّعالي<br>
لاحقةَ الآطالِ بالآطالِ<br>
يَرمينَ بالسخالِ والسخالِ<br>
للنسرِ أو للأطلسِ العَسال<br>
إن لم يكن للأسودِ الحجال<br>
كأن بين الأرض والرحالِ<br>
هنديةً جاءَت من الصقال<br>
لولا عصيرُ العرقِ الشَلشالِ<br>
يَرِدنَ مِن جَوزِ الفَلا الأَفلالِ<br>
بالمستقيمينَ وبالمُيّالِ<br>
مناهِلاً تُبذَلُ للنهالِ<br>
مِنَ الحَمامِ والقَطَا الأَرسالِ<br>
كأنَّ من أرياشِهِ النصالِ<br>
نِصال أقيانٍ على نِصالِ<br>
في آجنٍ أصفَرَ كالأبوالِ<br>
تشقُ منه الدلوَ عن مُحتالِ<br>
طامٍ كغسل الماشِطِ الغسال<br>
نجتازُهُ قفراً من السبال<br>
بيعملاتٍ بزلٍ عمال<br>
نوقٍ تُداني شبهَ الجمالِ<br>
يطوين بعد الأرضِ بالإِرقالِ<br>
إذا تسنمن مع الآصال<br>
دويةً غُولاً من الأغوال<br>
باتت على عُوج لها عجال<br>
لم تثنِ أوصالاً على أوصال<br>
حتى تقيلن مع القيال<br>
بمهمهٍ ليس بذي بلال<br>
تثير من تحتِ عروقِ الضالِ<br>
أم الغزالِ وأبا الغزال<br>
كأنها بين قوى الحبالِ<br>
إذ صارَ بطن البازِلِ الشملالِ<br>
في بطنها الداني إلى المحالِ<br>
كتابُ كافٍ أو كتابُ دالِ<br>
حتى ضيفنَ على المطال<br>
بعدَ الحفا منهنَّ والكَلال<br>
خليفةً سماهُ ذو الجلالِ<br>
أكرمَ من يمشي على النعال<br>
من كل جدٍّ وأبٍ وخال<br>
يا راعيَ الناسِ ارعَ لي عيالي<br>
وأكفهم الفقر إلى الموالي<br>
إنك تكفي بخلةَ البُخالِ<br>
بمفضلاتٍ من يدي مفضالِ<br>
إنهم كثروا وقل مالي<br>
فقلتُ لما أكسفوا لي بالي<br>
باللَه فيهم وبه اختيالي<br></h4>
                </div>
'''
soup = BeautifulSoup(html, 'html.parser')
print(extract_poem(soup))


def extract_tags(soup: BeautifulSoup) -> List[str]:
    tips_div = soup.find('div', class_='tips')
    if not tips_div:
        return []
    return [a.get_text(strip=True) for a in tips_div.find_all('a')]


def extract_poet(soup: BeautifulSoup, fallback_poet: Optional[str] = None) -> str:
    try:
        section_div = soup.find('div', class_='m-section-2')
        if not section_div:
            return fallback_poet or "Unknown"
        a_tags = section_div.find_all('a')
        return a_tags[2].get_text(strip=True) if len(a_tags) >= 3 else (fallback_poet or "Unknown")
    except Exception:
        return fallback_poet or "Unknown"


def extract_metadata(soup: BeautifulSoup) -> Dict[str, Optional[str]]:
    title_tag = soup.title.get_text(strip=True) if soup.title else None
    og_url = soup.find("meta", property="og:url")
    poem_url = og_url["content"] if og_url and og_url.has_attr("content") else None

    poet_page_url = None
    poet_english = None

    ld_json_script = soup.find("script", type="application/ld+json")
    if ld_json_script:
        try:
            data = json.loads(ld_json_script.get_text())
            items = data.get("itemListElement", [])
            for item in items:
                if "cat-poet" in item.get("item", ""):
                    poet_page_url = item["item"]
                    poet_english = poet_page_url.rstrip('/').split('-')[-1]
        except Exception:
            pass

    return {
        "title": title_tag,
        "poem_url": poem_url,
        "poet_page_url": poet_page_url,
        "poet_english": poet_english
    }


def extract_era_and_title(soup: BeautifulSoup, fallback_title: Optional[str] = None) -> Dict[str, str]:
    script_tag = soup.find("script", type="application/ld+json")
    if not script_tag:
        return {"era": "Unknown", "title": fallback_title or "Unknown"}

    try:
        data = json.loads(script_tag.get_text())
        items = data.get("itemListElement", [])
        era = items[0]["name"] if len(items) > 0 else "Unknown"
        title = items[2]["name"] if len(items) > 2 else (fallback_title or "Unknown")
        return {"era": era, "title": title}
    except Exception:
        return {"era": "Unknown", "title": fallback_title or "Unknown"}


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

def extract_poet_description(soup: BeautifulSoup) -> Optional[str]:
    container = soup.find('div', class_='col-lg-5 col-md-6 col-12 float-left mosahmat_block_top')
    if container:
        h4_tag = container.find('h4')
        if h4_tag:
            return h4_tag.get_text(strip=True)
    return None


def parse_html_file(filepath: Path) -> Dict:
    with filepath.open(encoding="utf-8") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")

    # Extract from <title> as fallback
    title_tag = soup.title.get_text(strip=True) if soup.title else None
    fallback = extract_fallback_title_and_poet(title_tag)

    poem = extract_poem(soup)
    poet = extract_poet(soup, fallback_poet=fallback["fallback_poet"])
    tags = extract_tags(soup)
    era_title = extract_era_and_title(soup, fallback_title=fallback["fallback_title"])
    metadata = extract_metadata(soup)
    poet_description = extract_poet_description(soup)
    poet_english_id = extract_poet_english_id(soup)

    result = {
        "title": metadata.get("title") or fallback["fallback_title"],
        "poem_url": metadata.get("poem_url"),
        "poet_page_url": metadata.get("poet_page_url"),
        "id": filepath.name,
        "poet": poet,
        "poem": poem or "",
        "tags": tags,
        "poem_no_diacritics": remove_arabic_diacritics(poem or ""),
        "era": era_title["era"],
        "poem_title": era_title["title"],
        "poet_english_id": poet_english_id,
        "poet_description": poet_description
    }

    # Sanity check: make sure no non-serializable objects (like BeautifulSoup Tag) are included
    for k, v in result.items():
        if not isinstance(v, (str, list, dict, type(None))):
            raise ValueError(f"Key '{k}' contains a non-serializable object: {type(v)}")

    return result


def main():
    path = Path("../../data/raw/diwan_html")
    html_files = sorted(path.glob("*.html"))

    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(parse_html_file, html_files, chunksize=10), total=len(html_files), desc="Parsing poems"))

    df = pd.DataFrame(results)
    assert len(df) == len(html_files), "Mismatch between files and extracted data"

    df.to_json("../../data/diwan.jsonl", orient="records", lines=True, force_ascii=False)


if __name__ == "__main__":
    main()

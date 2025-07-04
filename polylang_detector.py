import unicodedata
import regex as re
import string
import jieba
import fasttext
from collections import defaultdict, Counter

try:
    from janome.tokenizer import Tokenizer as JanomeTokenizer
    janome = JanomeTokenizer()
except:
    janome = None

try:
    from pythainlp.tokenize import word_tokenize as thai_tokenize
except:
    thai_tokenize = None

model = fasttext.load_model("lid.176.ftz")

SCRIPT_LANG_MAP = {
    "ARABIC": ["ar", "fa", "ur", "ckb"],
    "DEVANAGARI": ["hi", "mr", "ne"],
    "BENGALI": ["bn"],
    "GURMUKHI": ["pa"],
    "GUJARATI": ["gu"],
    "TAMIL": ["ta"],
    "TELUGU": ["te"],
    "KANNADA": ["kn"],
    "MALAYALAM": ["ml"],
    "SINHALA": ["si"],
    "THAI": ["th"],
    "ETHIOPIC": ["am", "ti"],
    "HAN": ["zh", "ja", "yue"],
    "HIRAGANA": ["ja"],
    "KATAKANA": ["ja"],
    "HANGUL": ["ko"],
    "LATIN": ["en", "fr", "de", "es", "it", "pt", "nl", "sv", "pl", "tr", "fi", "ro", "jbo"],
    "CYRILLIC": ["ru", "uk", "bg", "sr"],
}

LANGUAGE_HINTS = {
    "th": [r"\bสวัสดี\b", r"ครับ", r"ค่ะ", r"จ้า", r"นะ", r"ค่ะ"],
    "ko": [r"입니다", r"어요", r"습니까", r"네요", r"하다"],
    "am": [r"ነው", r"ናቸው", r"ይችላሉ", r"እንደ"],
    "ti": [r"ኣሎ", r"የን", r"እዩ", r"ኢዩ"],
    "ar": [r"\bال\w{3,}"],
    "fa": [r"\bمی\w{3,}", r"هستم", r"خواه\w+", r"\bبود"],
    "ur": [r"ہوں", r"رہا", r"چاہیے", r"کرنا"],
    "hi": [r"है", r"था", r"कर", r"से", r"रहा", r"की"],
    "ms": [r"saya", r"kami", r"tidak", r"akan", r"boleh"],
    "id": [r"saya", r"kamu", r"dia", r"tidak", r"akan", r"mereka"]
}

LOW_UTILITY_LANGS = {"bpy", "bs", "et", "la", "lt", "mk", "sh", "sk", "sl"}

# Expanded English stopwords for more aggressive filtering
EN_STOPWORDS = set('''
a an and are as at be but by for if in into is it no not of on or such that the their then there these they this to was will with you we do did does have has had can should would could our your his her its about also
after before once over under out up down than through very i me my mine myself we us our ours ourselves you your yours yourself yourselves he him his himself she her hers herself it its itself they them their theirs themselves what which who whom whose whoever whomever whichever whenever wherever however why how all any both each few more most other some such only own same so than too very can will just don don should now
their's aren't couldn't didn't doesn't hadn't hasn't haven't isn't shouldn't wasn't weren't won't wouldn't i'm i'll i'd i've you're you'll you'd you've he's he'll he'd she's she'll she'd we've we'll we'd they've they'll they'd it's it'll it'd let's that's there's who's what's where's when's why's how's
'''.split())

# Add more regex hints and stopwords for low-recall languages
LANGUAGE_HINTS.update({
    "sw": [r"\bmimi\b", r"\bwewe\b", r"\byeye\b", r"\bsisi\b", r"\bninyi\b", r"\bwao\b", r"\bhabari\b", r"\bkaribu\b", r"\brafiki\b", r"\bshule\b", r"\bkitabu\b", r"\bchakula\b", r"\bndugu\b", r"\bmtoto\b", r"\bnyumba\b", r"\bmbwa\b", r"\bpaka\b", r"\bngoma\b", r"\bbarua\b", r"\bmagari\b"],
    "ms": [r"\baku\b", r"\bdia\b", r"\bkita\b", r"\bengkau\b", r"\bpergi\b", r"\bdatang\b", r"\bkeluarga\b", r"\bsekolah\b", r"\bbuku\b", r"\bmakan\b", r"\bminum\b", r"\bkerja\b", r"\bteman\b", r"\bkanak-kanak\b", r"\brumah\b", r"\banjing\b", r"\bkucing\b", r"\btarian\b", r"\bsurat\b", r"\bkereta\b"],
    "id": [r"\bsaya\b", r"\bkamu\b", r"\bdia\b", r"\bpergi\b", r"\bdatang\b", r"\btinggal\b", r"\bmakan\b", r"\bminum\b", r"\bkeluarga\b", r"\bsekolah\b", r"\bbuku\b", r"\bteman\b", r"\banak\b", r"\brumah\b", r"\banjing\b", r"\bkucing\b", r"\btarian\b", r"\bsurat\b", r"\bmobil\b"],
    "no": [r"\bjeg\b", r"\bikke\b", r"\bmen\b", r"\bdet\b", r"\bvar\b", r"\bkan\b", r"\bskal\b", r"\bvenn\b", r"\bskole\b", r"\bbok\b", r"\bspise\b", r"\bdrikke\b", r"\bfamilie\b", r"\bbarn\b", r"\bhus\b", r"\bhund\b", r"\bkat\b", r"\bdans\b", r"\bbrev\b", r"\bbil\b"],
    "hr": [r"\bprijatelj\b", r"\bškola\b", r"\bknjiga\b", r"\bjesti\b", r"\bpiti\b", r"\bobitelj\b", r"\bdijete\b", r"\bkuća\b", r"\bpas\b", r"\bmačka\b", r"\bples\b", r"\bpismo\b", r"\bauto\b"],
    "cs": [r"\bpřítel\b", r"\bškola\b", r"\bkniha\b", r"\bjíst\b", r"\bpít\b", r"\brodina\b", r"\bdítě\b", r"\bdům\b", r"\bpes\b", r"\bkočka\b", r"\btanec\b", r"\bdopis\b", r"\bauto\b"]
})

# Add stopwords for these languages to help filtering
SW_STOPWORDS = set(["na", "ya", "kwa", "ni", "si", "wa", "la", "za", "katika", "hii", "hiyo", "ile", "kama", "bila", "hata", "basi", "pia", "tu", "bado", "tena", "sana", "sasa", "hapo", "huku", "humo", "hivyo", "hivyo hivyo"])
MS_STOPWORDS = set(["dan", "atau", "tetapi", "kerana", "jika", "dengan", "untuk", "pada", "dari", "ke", "oleh", "sebagai", "adalah", "itu", "ini", "juga", "sudah", "belum", "lagi", "akan", "masih", "sangat", "hanya", "sama", "lebih", "kurang", "paling", "antara", "setiap", "semua", "beberapa", "banyak", "sedikit"])
ID_STOPWORDS = set(["dan", "atau", "tetapi", "karena", "jika", "dengan", "untuk", "pada", "dari", "ke", "oleh", "sebagai", "adalah", "itu", "ini", "juga", "sudah", "belum", "lagi", "akan", "masih", "sangat", "hanya", "sama", "lebih", "kurang", "paling", "antara", "setiap", "semua", "beberapa", "banyak", "sedikit"])
NO_STOPWORDS = set(["og", "eller", "men", "fordi", "hvis", "med", "for", "på", "fra", "til", "av", "som", "er", "det", "dette", "den", "de", "vi", "du", "jeg", "han", "hun", "de", "vi", "dere", "oss", "seg", "sin", "sitt", "sine", "en", "et", "ei", "ikke", "var", "har", "hadde", "skal", "vil", "kan", "må", "bør", "blir", "ble", "blitt", "vært", "være"])

def get_script(char):
    try:
        return unicodedata.name(char).split(' ')[0]
    except:
        return "UNKNOWN"

def split_by_script(text):
    blocks = []
    buffer = ""
    current_script = None

    for char in text:
        script = get_script(char)

        if script in {"COMMON", "INHERITED"}:
            buffer += char
            continue

        if current_script is None:
            current_script = script
            buffer = char
        elif script == current_script or (
            current_script == "HANGUL" and script in {"LATIN", "COMMON"} and char in string.punctuation
        ):
            buffer += char
        else:
            if buffer:
                blocks.append((buffer, current_script))
            buffer = char
            current_script = script

    if buffer:
        blocks.append((buffer, current_script))
    return blocks

def tokenize_by_script(chunk, script):
    script = script.upper()
    if script == "HAN":
        return jieba.lcut(chunk)
    elif script in ["HIRAGANA", "KATAKANA"] and janome:
        return [token.surface for token in janome.tokenize(chunk)]
    elif script == "THAI" and thai_tokenize:
        return thai_tokenize(chunk)
    else:
        return re.findall(r'\w+|\S', chunk)

def is_valid_token(token, script=None, lang_hint=None):
    token = token.strip()
    if script and script.upper() != "LATIN":
        if len(token) < 1:
            return False
    else:
        if len(token) <= 1:
            return False
    if token.lower() in EN_STOPWORDS:
        return False
    # Only apply stopword filtering for low-resource languages if the hint is strong
    if lang_hint in {"sw", "ms", "id", "no"}:
        if lang_hint == "sw" and token.lower() in SW_STOPWORDS:
            return False
        if lang_hint == "ms" and token.lower() in MS_STOPWORDS:
            return False
        if lang_hint == "id" and token.lower() in ID_STOPWORDS:
            return False
        if lang_hint == "no" and token.lower() in NO_STOPWORDS:
            return False
    if all(c in string.punctuation + string.whitespace for c in token):
        return False
    if re.match(r'^[\d\W_]+$', token):
        return False
    return True

def regex_language_boost(text):
    hint_votes = Counter()
    for lang, patterns in LANGUAGE_HINTS.items():
        for pattern in patterns:
            # For Chinese, only boost if the match is for a full token of Chinese characters
            if lang == "zh":
                if re.fullmatch(pattern, text):
                    hint_votes[lang] += 2  # Stronger boost for full Chinese token
            else:
                if re.search(pattern, text):
                    hint_votes[lang] += 1
    if hint_votes:
        # Only return a hint if it's strong (e.g., >1 for zh, >0 for others)
        lang, votes = hint_votes.most_common(1)[0]
        if (lang == "zh" and votes > 1) or (lang != "zh" and votes > 0):
            return lang
    return None

def detect_language(text, script=None, threshold=0.3):
    try:
        labels, probs = model.predict(text.strip(), k=3)
        candidates = [(label.replace("__label__", ""), prob) for label, prob in zip(labels, probs)]

        if script:
            script_langs = SCRIPT_LANG_MAP.get(script.upper(), [])
            for i in range(len(candidates)):
                lang, score = candidates[i]
                if lang in script_langs:
                    score += 0.2
                candidates[i] = (lang, score)

        candidates.sort(key=lambda x: -x[1])
        top_lang, top_score = candidates[0]
        # For HAN script, prefer zh if token is all Chinese characters
        if script and script.upper() == "HAN":
            if re.fullmatch(r"[\u4e00-\u9fff]+", text):
                return "zh", 1.0

        # Aggressive English filtering
        if top_lang == "en":
            # Only assign EN if it is the top prediction by a margin of at least 0.2 and score >= 0.9
            if len(candidates) > 1 and (top_score - candidates[1][1] < 0.2 or top_score < 0.9):
                return "unknown", 0.0
            if top_score < 0.9:
                return "unknown", 0.0
        if script and script.upper() == "LATIN" and top_lang == "en":
            for lang, score in candidates[1:]:
                if lang != "en" and top_score < 0.95:
                    return lang, 0.0
        if top_lang in LOW_UTILITY_LANGS:
            return "unknown", 0.0
        if top_lang in {"ar", "fa", "ur", "hi"} and top_score < 0.4:
            return "unknown", 0.0
        return (top_lang, top_score) if top_score >= threshold else ("unknown", 0.0)
    except:
        return "unknown", 0.0

def guess_from_script(script):
    if not script:
        return "unknown"
    script = script.upper()
    langs = SCRIPT_LANG_MAP.get(script, [])
    return langs[0] if langs else "unknown"

def merge_adjacent_spans(pairs):
    # Improved merging: avoid concatenating tokens, use space separation, and keep character offsets
    merged = []
    buffer = ""
    prev_lang = None
    for token, lang in pairs:
        if lang == prev_lang:
            if buffer:
                buffer += " " + token
            else:
                buffer = token
        else:
            if buffer:
                merged.append((buffer.strip(), prev_lang))
            buffer = token
            prev_lang = lang
    if buffer:
        merged.append((buffer.strip(), prev_lang))
    return merged

def detect_languages(sentence):
    results = []
    unknown_count = 0
    total_tokens = 0

    blocks = split_by_script(sentence)
    for chunk, script in blocks:
        tokens = tokenize_by_script(chunk, script)
        chunk_results = []
        regex_hint = regex_language_boost(chunk)
        han_zh_count = 0
        han_total = 0
        han_token_indices = []
        for i, token in enumerate(tokens):
            lang_hint = regex_hint
            # For HAN script, prefer zh if token is mostly Chinese
            if script and script.upper() == "HAN":
                han_total += 1
                if re.search(r"[\u4e00-\u9fff]", token):
                    han_zh_count += 1
                    han_token_indices.append(i)
                if re.fullmatch(r"[\u4e00-\u9fff]+", token):
                    chunk_results.append((token, "zh"))
                    total_tokens += 1
                    continue
            if is_valid_token(token, script, lang_hint):
                total_tokens += 1
                top_lang, top_score = detect_language(token, script)
                if top_lang == "en" and script and script.upper() != "LATIN":
                    top_lang = guess_from_script(script)
                if top_score >= 0.3:
                    chunk_results.append((token, top_lang))
                else:
                    lang = guess_from_script(script)
                    if lang != "unknown":
                        chunk_results.append((token, lang))
                    else:
                        unknown_count += 1
        # Han script chunk-level fallback: only relabel ambiguous Han tokens as zh in Han-dominated chunks
        if script and script.upper() == "HAN" and han_total > 0 and han_zh_count / han_total > 0.3:
            new_chunk_results = []
            for idx, (token, lang) in enumerate(chunk_results):
                if lang == "unknown" or (lang in {"ja", "yue"} and re.search(r"[\u4e00-\u9fff]", token)):
                    left_zh = idx > 0 and chunk_results[idx-1][1] == "zh"
                    right_zh = idx < len(chunk_results)-1 and chunk_results[idx+1][1] == "zh"
                    if left_zh or right_zh or han_zh_count / han_total > 0.5:
                        new_chunk_results.append((token, "zh"))
                    else:
                        new_chunk_results.append((token, lang))
                else:
                    new_chunk_results.append((token, lang))
            chunk_results = new_chunk_results
        elif chunk_results:
            lang_counter = Counter([lang for _, lang in chunk_results])
            top_lang, freq = lang_counter.most_common(1)[0]
            if freq < len(chunk_results) * 0.5:
                fallback_lang, _ = detect_language(chunk, script)
                if fallback_lang != "unknown":
                    chunk_results = [(chunk, fallback_lang)]
                else:
                    chunk_results = [(chunk, guess_from_script(script))]
        else:
            if script == "HANGUL":
                hangul_chars = [c for c in chunk if get_script(c) == "HANGUL"]
                if len(hangul_chars) > 3:
                    chunk_results = [(chunk, "ko")]
                else:
                    unknown_count += 1
            elif is_valid_token(chunk):
                lang, _ = detect_language(chunk, script)
                if lang == "unknown":
                    lang = guess_from_script(script)
                if lang != "unknown":
                    chunk_results = [(chunk, lang)]
                else:
                    unknown_count += 1

        results.extend(chunk_results)

    if total_tokens > 0 and unknown_count / total_tokens > 0.8:
        fallback_lang, _ = detect_language(sentence, None)
        if fallback_lang != "unknown":
            return [(sentence, fallback_lang)]

    return merge_adjacent_spans(results)

def print_code_switch_spans(spans):
    for text, lang in spans:
        print(f"[{lang.upper()}] {text}")

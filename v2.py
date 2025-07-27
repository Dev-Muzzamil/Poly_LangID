import unicodedata
import regex as re
import string
import jieba
import time
import logging
from collections import defaultdict, Counter
from functools import lru_cache, wraps
from typing import List, Tuple, Optional, Union
import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import numpy as np
import fasttext

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HybridPolyLangID")

# Load optional tokenizers
try:
    from janome.tokenizer import Tokenizer as JanomeTokenizer
    janome = JanomeTokenizer()
except:
    janome = None

try:
    from pythainlp.tokenize import word_tokenize as thai_tokenize
except:
    thai_tokenize = None

def get_batch_size_from_user():
    """Get batch size from user"""
    print("\n" + "="*60)
    print("🚀 HYBRID POLYLANGID - BEST OF XLM-V + FASTTEXT")
    print("="*60)
    print("🔥 COMBINING:")
    print("   ✅ XLM-V transformer power")
    print("   ✅ FastText dead language revival")
    print("   ✅ Enhanced pattern matching")
    print("   ✅ Nuclear English detection")
    print("   ✅ Language-specific tokenization")
    print()
    print("📋 BATCH SIZE OPTIONS:")
    print("  1️⃣  Small (10 sentences)     - Safe for testing")
    print("  2️⃣  Medium (25 sentences)    - RECOMMENDED") 
    print("  3️⃣  Large (50 sentences)     - For good systems")
    print("  4️⃣  Extra (100 sentences)    - High performance")
    print()
    
    while True:
        try:
            choice = input("🔢 Enter choice (1-4, or Enter for recommended): ").strip()
            if choice == "" or choice == "2": return 25
            elif choice == "1": return 10
            elif choice == "3": return 50
            elif choice == "4": return 100
            else: print("❌ Please enter 1-4")
        except KeyboardInterrupt:
            print("\n👋 Using recommended: 25")
            return 25

class HybridPolyLangID:
    def __init__(self, model_batch_size: int = 16, sentence_batch_size: int = None, device: str = "auto", interactive: bool = True):
        """
        HYBRID SYSTEM: XLM-V + FastText + Enhanced Patterns
        """
        self.model_batch_size = model_batch_size
        
        if sentence_batch_size is None and interactive:
            self.sentence_batch_size = get_batch_size_from_user()
        else:
            self.sentence_batch_size = sentence_batch_size or 25
            
        self.setup_device(device)
        self.load_models()
        self.setup_hybrid_language_data()
        
        print(f"\n🚀 HYBRID SYSTEM INITIALIZED:")
        print(f"   Primary: XLM-V transformer")
        print(f"   Secondary: FastText patterns")
        print(f"   Enhanced: Dead language revival")
        print(f"   Batch size: {self.sentence_batch_size}")
        print("="*60)
    
    def setup_device(self, device: str):
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print(f"🔥 GPU: {torch.cuda.get_device_name()}")
            else:
                self.device = torch.device("cpu")
                print("💻 Using CPU")
        else:
            self.device = torch.device(device)
    
    def load_models(self):
        """Load both XLM-V and FastText models"""
        # Load XLM-V (primary)
        print("🔄 Loading XLM-V transformer...")
        model_name = "juliensimon/xlm-v-base-language-id"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        device_id = 0 if self.device.type == "cuda" else -1
        self.xlm_v_detector = pipeline(
            "text-classification", 
            model=self.model, 
            tokenizer=self.tokenizer, 
            device=device_id,
            batch_size=self.model_batch_size,
            truncation=True
        )
        
        # Load FastText (secondary)
        try:
            print("🔄 Loading FastText...")
            self.fasttext_model = fasttext.load_model('lid.176.ftz')
            print("✅ Both models loaded successfully")
        except:
            print("⚠️  FastText not available, using XLM-V only")
            self.fasttext_model = None
    
    def setup_hybrid_language_data(self):
        """Setup enhanced language data combining both systems"""
        
        self.TOP_40_LANGUAGES = {
            'en', 'zh', 'hi', 'es', 'fr', 'ar', 'bn', 'pt', 'ru', 'ur', 
            'id', 'de', 'ja', 'tr', 'vi', 'ko', 'it', 'th', 'gu', 'ta',
            'te', 'ml', 'kn', 'or', 'mr', 'pa', 'ne', 'si', 'my', 'km',
            'lo', 'ka', 'am', 'ti', 'sw', 'ha', 'yo', 'ig', 'zu', 'af'
        }
        
        # Perfect script detection (keep what works)
        self.PERFECT_SCRIPT_LANGUAGES = {
            "BENGALI": "bn", "HIRAGANA": "ja", "KATAKANA": "ja", 
            "HANGUL": "ko", "THAI": "th", "TAMIL": "ta",
            "TELUGU": "te", "MALAYALAM": "ml", "GREEK": "el", "HEBREW": "he"
        }
        
        # ENHANCED LANGUAGE HINTS - Dead Language Revival from FastText
        self.ENHANCED_LANGUAGE_HINTS = {
            # Major languages (enhanced)
            "en": [
                r'\b(the|and|that|have|you|this|with|from|they|been|which|their|will|would|could|should|must|can|are|is|was|were|be|do|does|did|has|had)\b',
                r'\b\w+ing\b', r'\b\w+ed\b', r'\b\w+ly\b', r'\b\w+tion\b'
            ],
            
            # DEAD LANGUAGE REVIVAL - From FastText system
            "no": [  # Norwegian revival
                r"\b(og|det|for|med|hvor|eller|som|ikke|skal|vil|kan|har|er|var|blir|ble|når|hvis|jeg|du|han|hun|vi|dere|de)\b", 
                r"[ø]", r"[å]", r"\bkj", r"\bskj"
            ],
            "da": [  # Danish revival
                r"\b(og|det|for|med|hvor|eller|som|ikke|skal|vil|kan|har|er|var|bliver|blev|når|hvis|jeg|du|han|hun|vi|I|de)\b", 
                r"[ø]", r"[å]", r"[æ]"
            ],
            "sv": [  # Swedish revival
                r"\b(och|det|för|med|var|eller|som|inte|ska|vill|kan|har|är|var|blir|blev|när|om|jag|du|han|hon|vi|ni|de)\b", 
                r"[ä]", r"[ö]", r"[å]"
            ],
            "fi": [  # Finnish revival
                r"\b(ja|se|että|on|ei|ole|tai|kun|jos|kuin|hän|minä|sinä|me|te|he|tämä|tuo|mikä|mitä|missä|milloin|miten|miksi)\b", 
                r"[ä]", r"[ö]", r"[y]", r"kk", r"ll", r"nn", r"tt"
            ],
            "hr": [  # Croatian revival
                r"\b(i|je|se|da|u|na|za|s|o|od|do|preko|između|poslije|prije|biti|imati|moći|htjeti|trebati)\b", 
                r"[čćžšđ]"
            ],
            "cs": [  # Czech revival
                r"\b(je|jsou|byl|byla|bylo|byli|byly|a|ale|nebo|protože|když|že|jak|co|kde|já|ty|on|ona|my|vy|oni|ony)\b", 
                r"[čř][aeiou]", r"[ěščřžýáíéúů]"
            ],
            "pl": [  # Polish revival
                r"\b(i|jest|są|był|była|było|byli|były|a|ale|lub|bo|gdy|że|jak|co|gdzie|ja|ty|on|ona|my|wy|oni|one)\b", 
                r"[ąćęłńóśźż]"
            ],
            "hu": [  # Hungarian revival
                r"\b(és|van|lesz|volt|egy|a|az|el|be|ki|fel|le|meg|át|vissza|hogy|mint|ami|aki|ahol|amikor|miért)\b", 
                r"[őűáéíóú]", r"gy", r"ly", r"ny", r"sz", r"ty", r"zs"
            ],
            "nl": [  # Dutch revival
                r"\b(en|de|het|een|van|in|op|voor|met|aan|bij|door|over|onder|naar|uit|tegen|tussen|zonder|binnen|buiten)\b", 
                r"[ij]", r"oe", r"ui", r"au", r"ou"
            ],
            "ro": [  # Romanian revival
                r"\b(și|de|la|în|cu|pe|pentru|ca|să|care|ce|când|unde|cum|de|ce|dacă|eu|tu|el|ea|noi|voi|ei|ele)\b", 
                r"[ăâîșț]"
            ],
            
            # Enhanced major European languages
            "fr": [r'\b(le|la|les|un|une|des|et|de|du|dans|pour|avec|qui|que|ce|cette|ces|mais|ou|où|sur|par)\b'],
            "de": [r'\b(der|die|das|und|in|von|zu|den|mit|sich|auf|für|ist|im|dem|nicht|ein|eine|als|auch|es|an)\b'],
            "es": [r'\b(el|la|los|las|un|una|y|de|en|un|una|es|con|por|que|no|se|le|lo|me|te|su|sus)\b'],
            "it": [r'\b(il|la|lo|gli|le|di|da|in|con|su|per|tra|fra|a|e|ma|o|se|che|chi|cui|dove|quando|come)\b'],
            "pt": [r'\b(o|a|os|as|um|uma|e|de|em|um|uma|é|com|por|que|não|se|do|da|dos|das|no|na)\b'],
            "ru": [r'\b(это|что|который|быть|иметь|мочь|сказать|говорить|знать|стать|видеть|хотеть)\b'],
            
            # Non-Latin scripts
            "hi": [r'है', r'था', r'कर', r'से', r'रहा', r'की', r'के', r'को', r'में', r'पर', r'और'],
            "ar": [r'\bال\w{3,}', r'في', r'من', r'إلى', r'على', r'مع', r'عن', r'كان', r'يكون'],
            "zh": [r'的', r'了', r'在', r'是', r'我', r'有', r'和', r'就', r'不', r'人', r'都', r'一']
        }
        
        # ENHANCED CHARACTER PATTERNS - From FastText system
        self.ENHANCED_CHARACTER_PATTERNS = {
            'fi': ['ä', 'ö', 'y'], 'sv': ['ä', 'ö', 'å'], 'no': ['ø', 'å', 'æ'], 'da': ['ø', 'å', 'æ'],
            'de': ['ä', 'ö', 'ü', 'ß'], 'fr': ['ç', 'é', 'è', 'ê', 'à', 'ù'],
            'es': ['ñ', 'í', 'ó', 'á', 'é', 'ú'], 'pt': ['ã', 'õ', 'ç'],
            'it': ['à', 'è', 'é', 'ì', 'ò', 'ù'], 'tr': ['ğ', 'ı', 'ş', 'ç'],
            'hr': ['č', 'ć', 'ž', 'š', 'đ'], 'cs': ['č', 'ř', 'ň', 'š', 'ž'], 'pl': ['ą', 'ć', 'ę', 'ł', 'ń'],
            'hu': ['ő', 'ű', 'á', 'é', 'í'], 'nl': ['ij'], 'ro': ['ă', 'â', 'î', 'ș', 'ț']
        }
        
        # Script mapping for boosting
        self.SCRIPT_LANG_MAP = {
            "LATIN": ["en", "fr", "de", "es", "it", "pt", "nl", "sv", "no", "da", "fi", "pl", "cs", "hr", "hu", "ro"],
            "CYRILLIC": ["ru", "bg", "uk", "sr"],
            "ARABIC": ["ar", "fa", "ur"], "DEVANAGARI": ["hi", "mr", "ne"],
            "HAN": ["zh", "ja"], "HANGUL": ["ko"], "HIRAGANA": ["ja"], "KATAKANA": ["ja"]
        }
        
        # Dead languages set for prioritization
        self.DEAD_LANGUAGES = {"no", "da", "sv", "fi", "hr", "cs", "pl", "hu", "nl", "ro"}
    
    @lru_cache(maxsize=2048)
    def get_script(self, char: str) -> str:
        try:
            return unicodedata.name(char).split(' ')[0]
        except:
            return "UNKNOWN"
    
    def detect_perfect_script(self, text: str, script: str) -> Tuple[Optional[str], float]:
        """Perfect script detection - keep what works perfectly"""
        if not script:
            return None, 0.0
            
        script = script.upper()
        
        if script in self.PERFECT_SCRIPT_LANGUAGES:
            lang = self.PERFECT_SCRIPT_LANGUAGES[script]
            
            if script == 'HAN':
                if re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text):
                    return 'ja', 0.95
                else:
                    return 'zh', 0.95
            elif script == 'ARABIC':
                if re.search(r'[پچژگ]', text):
                    return 'fa', 0.95
                elif re.search(r'[ٹڈڑںےہ]', text):
                    return 'ur', 0.95
                else:
                    return 'ar', 0.95
            else:
                return lang, 1.0
        
        return None, 0.0
    
    def enhanced_pattern_matching(self, text: str) -> Tuple[Optional[str], float]:
        """Enhanced pattern matching with dead language revival"""
        text_lower = text.lower()
        scores = Counter()
        
        for lang, patterns in self.ENHANCED_LANGUAGE_HINTS.items():
            if lang in self.TOP_40_LANGUAGES:
                total_matches = 0
                for pattern in patterns:
                    matches = len(re.findall(pattern, text_lower))
                    total_matches += matches
                
                if total_matches > 0:
                    # Boost dead languages for revival
                    multiplier = 1.5 if lang in self.DEAD_LANGUAGES else 1.0
                    scores[lang] = total_matches * multiplier
        
        if scores:
            best_lang = scores.most_common(1)[0][0]
            max_score = scores[best_lang]
            # Higher confidence for pattern matches, especially dead languages
            base_confidence = 0.85 if best_lang in self.DEAD_LANGUAGES else 0.75
            confidence = min(0.95, base_confidence + (max_score * 0.05))
            return best_lang, confidence
            
        return None, 0.0
    
    def enhanced_character_detection(self, text: str) -> Tuple[Optional[str], float]:
        """Enhanced character-level detection from FastText system"""
        if len(text) < 3:
            return None, 0.0
        
        text_lower = text.lower()
        scores = defaultdict(float)
        
        # Character pattern scoring with dead language boost
        for lang, chars in self.ENHANCED_CHARACTER_PATTERNS.items():
            if lang in self.TOP_40_LANGUAGES:
                char_score = sum(1 for char in chars if char in text_lower)
                if char_score > 0:
                    multiplier = 1.5 if lang in self.DEAD_LANGUAGES else 1.0
                    scores[lang] += char_score * 0.7 * multiplier
        
        # Enhanced frequency analysis
        char_freq = Counter(text_lower)
        text_len = len(text)
        
        if text_len > 0:
            # Nordic languages
            if char_freq.get('ø', 0) / text_len > 0.02: scores['no'] += 0.5; scores['da'] += 0.5
            if char_freq.get('å', 0) / text_len > 0.02: scores['no'] += 0.4; scores['sv'] += 0.4
            if char_freq.get('ä', 0) / text_len > 0.02: scores['sv'] += 0.4; scores['fi'] += 0.4
            if char_freq.get('ö', 0) / text_len > 0.02: scores['sv'] += 0.3; scores['de'] += 0.3
            
            # Slavic languages
            if char_freq.get('č', 0) / text_len > 0.01: scores['cs'] += 0.5; scores['hr'] += 0.4
            if char_freq.get('ł', 0) / text_len > 0.01: scores['pl'] += 0.6
            if char_freq.get('ř', 0) / text_len > 0.01: scores['cs'] += 0.7
            
            # Other patterns
            if char_freq.get('ß', 0) / text_len > 0.01: scores['de'] += 0.8
            if char_freq.get('ñ', 0) / text_len > 0.01: scores['es'] += 0.8
        
        if scores:
            best_lang = max(scores, key=scores.get)
            confidence = min(0.85, scores[best_lang])
            if confidence > 0.2:
                return best_lang, confidence
        
        return None, 0.0
    
    def nuclear_english_detection(self, text: str, detected_lang: str, confidence: float) -> Tuple[str, float]:
        """Nuclear English detection from FastText system"""
        if detected_lang != "en":
            return detected_lang, confidence
        
        # Less aggressive than FastText but still protective
        if confidence < 0.6:  # Reduced from 0.95
            return "unknown", 0.0
        
        # Pattern requirements
        english_patterns = [
            r'\b(the|and|that|have|you|this|with|from|they|been|which|their)\b',
            r'\b\w+ing\b', r'\b\w+ed\b', r'\b\w+ly\b'
        ]
        
        matches = sum(1 for pattern in english_patterns if re.search(pattern, text.lower()))
        if matches < 1:  # At least 1 pattern required
            return "unknown", 0.0
        
        # Check for foreign characters
        foreign_chars = r'[àáâäåæçèéêëìíîïñòóôöøùúûüÿčćžšđřň]'
        if re.search(foreign_chars, text.lower()):
            return "unknown", 0.0
        
        return detected_lang, confidence
    
    def fasttext_detection(self, text: str) -> Tuple[str, float]:
        """FastText detection as secondary model"""
        if not self.fasttext_model:
            return "unknown", 0.0
        
        try:
            predictions = self.fasttext_model.predict(text, k=3)
            labels, scores = predictions
            
            # Extract language from __label__xx format
            if labels:
                lang = labels[0].replace('__label__', '')
                confidence = float(scores[0])
                
                if lang in self.TOP_40_LANGUAGES and confidence > 0.3:
                    return lang, confidence
        
        except Exception as e:
            logger.debug(f"FastText error: {e}")
        
        return "unknown", 0.0
    
    def intelligent_fusion(self, xlm_result: Tuple[str, float], fasttext_result: Tuple[str, float], 
                          text: str, script: str = None) -> Tuple[str, float]:
        """Intelligent fusion of XLM-V and FastText predictions"""
        xlm_lang, xlm_conf = xlm_result
        ft_lang, ft_conf = fasttext_result
        
        # Priority 1: High-confidence dead language patterns (FastText strength)
        if ft_lang in self.DEAD_LANGUAGES and ft_conf > 0.8:
            return ft_lang, ft_conf
        
        # Priority 2: Perfect script matches
        if script:
            script_langs = self.SCRIPT_LANG_MAP.get(script.upper(), [])
            if xlm_lang in script_langs and xlm_conf > 0.5:
                return xlm_lang, min(0.95, xlm_conf + 0.1)  # Script boost
            if ft_lang in script_langs and ft_conf > 0.5:
                return ft_lang, min(0.95, ft_conf + 0.1)  # Script boost
        
        # Priority 3: High-confidence XLM-V for longer context
        if xlm_conf > 0.7 and len(text.split()) > 5:
            return xlm_lang, xlm_conf
        
        # Priority 4: Agreement between models
        if xlm_lang == ft_lang:
            combined_conf = min(0.95, (xlm_conf + ft_conf) / 2 + 0.1)
            return xlm_lang, combined_conf
        
        # Priority 5: Higher confidence wins
        if xlm_conf > ft_conf and xlm_conf > 0.4:
            return xlm_lang, xlm_conf
        elif ft_conf > 0.4:
            return ft_lang, ft_conf
        
        return "unknown", 0.0
    
    def language_specific_tokenization(self, text: str, script: str = None) -> List[Tuple[str, str]]:
        """Language-specific tokenization from FastText system"""
        tokens_with_scripts = []
        
        # Determine if we need special tokenization
        if script:
            if script.upper() == "HAN":
                # Chinese tokenization
                segments = jieba.lcut(text)
                for segment in segments:
                    if len(segment.strip()) > 0:
                        tokens_with_scripts.append((segment, script))
                return tokens_with_scripts
            
            elif script.upper() in ["HIRAGANA", "KATAKANA"] and janome:
                # Japanese tokenization
                tokens = [token.surface for token in janome.tokenize(text)]
                for token in tokens:
                    if len(token.strip()) > 0:
                        tokens_with_scripts.append((token, script))
                return tokens_with_scripts
            
            elif script.upper() == "THAI" and thai_tokenize:
                # Thai tokenization
                tokens = thai_tokenize(text)
                for token in tokens:
                    if len(token.strip()) > 0:
                        tokens_with_scripts.append((token, script))
                return tokens_with_scripts
        
        # Default word-based tokenization
        segments = re.findall(r'\b\w+\b', text)
        for segment in segments:
            if len(segment) > 1:
                token_script = self.get_script(segment[0]) if segment else "UNKNOWN"
                tokens_with_scripts.append((segment, token_script))
        
        return tokens_with_scripts
    
    def detect_single_token(self, text: str, script: str = None) -> Tuple[str, float]:
        """Hybrid token detection combining all approaches"""
        
        # Stage 1: Perfect script detection
        if script:
            perfect_result = self.detect_perfect_script(text, script)
            if perfect_result[0]:
                return perfect_result
        
        # Stage 2: Enhanced pattern matching (dead language revival)
        pattern_result = self.enhanced_pattern_matching(text)
        if pattern_result[0]:
            return pattern_result
        
        # Stage 3: Dual model detection
        try:
            # XLM-V detection
            xlm_results = self.xlm_v_detector(text, top_k=3)
            if xlm_results:
                xlm_lang = xlm_results[0]['label']
                xlm_conf = xlm_results[0]['score']
                
                # Filter to top 40
                if xlm_lang not in self.TOP_40_LANGUAGES:
                    for result in xlm_results[1:]:
                        if result['label'] in self.TOP_40_LANGUAGES:
                            xlm_lang = result['label']
                            xlm_conf = result['score']
                            break
                    else:
                        xlm_lang, xlm_conf = "unknown", 0.0
            else:
                xlm_lang, xlm_conf = "unknown", 0.0
            
            # FastText detection
            ft_lang, ft_conf = self.fasttext_detection(text)
            
            # Apply nuclear English detection to XLM-V result
            xlm_lang, xlm_conf = self.nuclear_english_detection(text, xlm_lang, xlm_conf)
            
            # Intelligent fusion
            fused_result = self.intelligent_fusion((xlm_lang, xlm_conf), (ft_lang, ft_conf), text, script)
            
            if fused_result[0] != "unknown" and fused_result[1] > 0.05:  # Very low threshold
                return fused_result
            
        except Exception as e:
            logger.debug(f"Model detection error: {e}")
        
        # Stage 4: Character-level fallback
        char_result = self.enhanced_character_detection(text)
        if char_result[0]:
            return char_result
        
        return "unknown", 0.0
    
    def merge_adjacent_languages(self, results: List[Tuple[str, str, float]]) -> List[Tuple[str, str, float]]:
        """Enhanced merging with confidence preservation"""
        if not results:
            return []
        
        merged = []
        current_text = ""
        current_lang = None
        current_conf = 0.0
        
        for text, lang, confidence in results:
            if lang == current_lang and lang != "unknown":
                current_text += " " + text if current_text else text
                current_conf = max(current_conf, confidence)  # Take max confidence
            else:
                if current_text:
                    merged.append((current_text.strip(), current_lang, current_conf))
                current_text = text
                current_lang = lang
                current_conf = confidence
        
        if current_text:
            merged.append((current_text.strip(), current_lang, current_conf))
        
        return merged
    
    def detect_sentence(self, sentence: str) -> List[Tuple[str, str, float]]:
        """Hybrid sentence detection"""
        if not sentence or not sentence.strip():
            return []
        
        # Language-specific tokenization
        script = self.get_script(sentence[0]) if sentence else None
        tokens_with_scripts = self.language_specific_tokenization(sentence, script)
        
        if not tokens_with_scripts:
            return [(sentence.strip(), "unknown", 0.0)]
        
        # Detect each token
        results = []
        for token, token_script in tokens_with_scripts:
            lang, confidence = self.detect_single_token(token, token_script)
            results.append((token, lang, confidence))
        
        # Merge adjacent same languages
        merged_results = self.merge_adjacent_languages(results)
        return merged_results
    
    def detect_batch(self, sentences: List[str], progress_callback=None) -> List[List[Tuple[str, str, float]]]:
        """Hybrid batch processing"""
        batch_results = []
        total_sentences = len(sentences)
        
        print(f"\n🚀 HYBRID PROCESSING: {total_sentences} sentences")
        print(f"🔥 XLM-V + FastText + Enhanced Patterns")
        print(f"📦 Batch size: {self.sentence_batch_size}")
        print("="*60)
        
        for i in range(0, len(sentences), self.sentence_batch_size):
            batch = sentences[i:i + self.sentence_batch_size]
            batch_num = i // self.sentence_batch_size + 1
            total_batches = (total_sentences + self.sentence_batch_size - 1) // self.sentence_batch_size
            
            print(f"🔄 Processing hybrid batch {batch_num}/{total_batches}: {len(batch)} sentences")
            
            batch_start_time = time.time()
            
            batch_batch_results = []
            for j, sentence in enumerate(batch):
                try:
                    result = self.detect_sentence(sentence)
                    batch_batch_results.append(result)
                    
                    if progress_callback:
                        processed = i + j + 1
                        progress_callback(processed, total_sentences, batch_num, total_batches)
                    
                    if (j + 1) % max(1, len(batch) // 4) == 0:
                        print(f"  ⏳ Progress: {j+1}/{len(batch)} sentences")
                        
                except Exception as e:
                    print(f"⚠️  Error processing sentence {i+j+1}: {e}")
                    batch_batch_results.append([(sentence, "unknown", 0.0)])
            
            batch_results.extend(batch_batch_results)
            
            batch_time = time.time() - batch_start_time
            sentences_per_sec = len(batch) / batch_time if batch_time > 0 else 0
            print(f"✅ Hybrid batch {batch_num} completed in {batch_time:.2f}s ({sentences_per_sec:.1f} sent/sec)")
            
            # Memory cleanup
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        
        print(f"\n🎉 HYBRID PROCESSING COMPLETED!")
        print("="*60)
        return batch_results
    
    def get_hybrid_stats(self, results: List[List[Tuple[str, str, float]]]) -> dict:
        """Get comprehensive statistics"""
        stats = {
            'total_sentences': len(results),
            'language_counts': Counter(),
            'dead_language_revivals': 0,
            'script_based_detections': 0,
            'pattern_based_detections': 0,
            'model_fusion_detections': 0,
            'unknown_rate': 0.0
        }
        
        for sentence_results in results:
            for text, lang, confidence in sentence_results:
                if lang != "unknown":
                    stats['language_counts'][lang] += 1
                    
                    if lang in self.DEAD_LANGUAGES:
                        stats['dead_language_revivals'] += 1
                    
                    # Estimate detection source based on confidence patterns
                    if confidence >= 0.95:
                        stats['script_based_detections'] += 1
                    elif confidence >= 0.8:
                        stats['pattern_based_detections'] += 1
                    else:
                        stats['model_fusion_detections'] += 1
        
        total_detections = sum(stats['language_counts'].values())
        unknown_count = sum(len([1 for _, lang, _ in sentence if lang == "unknown"]) for sentence in results)
        
        if total_detections + unknown_count > 0:
            stats['unknown_rate'] = unknown_count / (total_detections + unknown_count)
        
        return stats

# Global compatibility functions
_global_detector = None

def _get_global_detector():
    global _global_detector
    if _global_detector is None:
        logger.info("Initializing hybrid global detector...")
        _global_detector = HybridPolyLangID(
            model_batch_size=8,
            sentence_batch_size=1,
            interactive=False
        )
    return _global_detector

def detect_languages(sentence: str) -> List[Tuple[str, str]]:
    """Hybrid compatibility function"""
    try:
        detector = _get_global_detector()
        results = detector.detect_sentence(sentence)
        return [(text, lang) for text, lang, _ in results]
    except Exception as e:
        logger.error(f"Error in detect_languages: {e}")
        return [(sentence, "unknown")]

def detect_languages_with_confidence(sentence: str) -> List[Tuple[str, str, float]]:
    """Hybrid compatibility function with confidence"""
    try:
        detector = _get_global_detector()
        results = detector.detect_sentence(sentence)
        return results
    except Exception as e:
        logger.error(f"Error in detect_languages_with_confidence: {e}")
        return [(sentence, "unknown", 0.0)]

# Main API
def create_hybrid_detector(sentence_batch_size: int = None) -> HybridPolyLangID:
    """Create hybrid detector combining XLM-V + FastText"""
    return HybridPolyLangID(
        model_batch_size=8,
        sentence_batch_size=sentence_batch_size or 25,
        interactive=True
    )

# Test the hybrid system
if __name__ == "__main__":
    print("🚀 HYBRID POLYLANGID - XLM-V + FASTTEXT + ENHANCED PATTERNS")
    print("="*80)
    
    # Create hybrid detector
    detector = create_hybrid_detector()
    
    # Test cases covering all major improvements
    hybrid_test_sentences = [
        # English (was broken - 0.14 F1)
        "Hello world this is English text",
        "The quick brown fox jumps over the lazy dog",
        
        # Dead language revival tests
        "Dette er norsk tekst og det fungerer",  # Norwegian
        "Det här är svenska och det borde fungera",  # Swedish  
        "Tämä on suomalaista tekstiä",  # Finnish
        "To je český text a měl by fungovat",  # Czech
        "Ovo je hrvatski tekst",  # Croatian
        "To jest polski tekst",  # Polish
        "Ez magyar szöveg",  # Hungarian
        
        # Major European languages
        "Bonjour le monde comment allez-vous",  # French
        "Hola mundo como estas muy bien",  # Spanish
        "Ich bin sehr glücklich heute",  # German
        "Mi piace molto l'italiano",  # Italian
        
        # Script-based (should remain perfect)
        "এটি বাংলা টেক্সট",  # Bengali
        "これは日本語です",      # Japanese  
        "안녕하세요 여러분",     # Korean
        
        # Code-switching test
        "Hello world 这是中文 bonjour le monde",
        
        # Single problematic words
        "sinfonia", "familie", "mot"  # These were causing issues
    ]
    
    print(f"\n🧪 HYBRID SYSTEM TEST: {len(hybrid_test_sentences)} sentences")
    
    start_time = time.time()
    results = detector.detect_batch(hybrid_test_sentences)
    total_time = time.time() - start_time
    
    print(f"\n⚡ HYBRID TEST RESULTS:")
    print(f"   Time: {total_time:.2f} seconds")
    print(f"   Speed: {len(hybrid_test_sentences)/total_time:.1f} sentences/sec")
    
    # Show detailed results
    print(f"\n🔍 DETAILED HYBRID RESULTS:")
    english_count = dead_lang_count = script_count = unknown_count = 0
    
    for i, (sentence, result_list) in enumerate(zip(hybrid_test_sentences, results)):
        print(f"\n{i+1}. Input: '{sentence}'")
        for text, lang, conf in result_list:
            print(f"   [{lang.upper()}:{conf:.3f}] '{text}'")
            
            if lang == "en": english_count += 1
            elif lang in {"no", "sv", "fi", "cs", "hr", "pl", "hu"}: dead_lang_count += 1
            elif lang in {"bn", "ja", "ko"}: script_count += 1
            elif lang == "unknown": unknown_count += 1
    
    # Get hybrid statistics
    stats = detector.get_hybrid_stats(results)
    
    print(f"\n📊 HYBRID SYSTEM PERFORMANCE:")
    print(f"   English detections: {english_count} (was broken in XLM-V)")
    print(f"   Dead language revivals: {dead_lang_count} (Norwegian, Czech, etc.)")
    print(f"   Script-based (perfect): {script_count} (Bengali, Japanese, Korean)")
    print(f"   Unknown count: {unknown_count} (should be minimal)")
    print(f"   Success rate: {((sum(len(r) for r in results) - unknown_count) / sum(len(r) for r in results) * 100):.1f}%")
    print(f"   Languages detected: {list(stats['language_counts'].keys())}")
    
    if english_count > 0 and dead_lang_count > 0 and unknown_count < 3:
        print(f"\n🎉 HYBRID SYSTEM SUCCESS!")
        print(f"✅ English detection: RESTORED")
        print(f"✅ Dead languages: REVIVED")
        print(f"✅ Script detection: MAINTAINED") 
        print(f"✅ Ready for full evaluation")
    else:
        print(f"\n⚠️  Some hybrid components need tuning")
    
    print("="*80)

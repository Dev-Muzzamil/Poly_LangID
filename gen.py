import random
import json

# For each language, 20 random words (2 per level, levels 1-10)
lang_words = {
    "en": {
        1: ["cat", "dog"],
        2: ["apple", "book"],
        3: ["family", "river"],
        4: ["journey", "window"],
        5: ["freedom", "teacher"],
        6: ["mountain", "courage"],
        7: ["elegance", "disaster"],
        8: ["abundance", "strategy"],
        9: ["philosophy", "symphony"],
        10: ["metamorphosis", "quintessence"]
    },
    "zh": {
        1: ["猫", "狗"],
        2: ["苹果", "书"],
        3: ["家庭", "河流"],
        4: ["旅程", "窗口"],
        5: ["自由", "老师"],
        6: ["山", "勇气"],
        7: ["优雅", "灾难"],
        8: ["丰富", "策略"],
        9: ["哲学", "交响曲"],
        10: ["变形", "精髓"]
    },
    "es": {
        1: ["gato", "perro"],
        2: ["manzana", "libro"],
        3: ["familia", "río"],
        4: ["viaje", "ventana"],
        5: ["libertad", "maestro"],
        6: ["montaña", "valor"],
        7: ["elegancia", "desastre"],
        8: ["abundancia", "estrategia"],
        9: ["filosofía", "sinfonía"],
        10: ["metamorfosis", "quintaesencia"]
    },
    "hi": {
        1: ["बिल्ली", "कुत्ता"],
        2: ["सेब", "किताब"],
        3: ["परिवार", "नदी"],
        4: ["यात्रा", "खिड़की"],
        5: ["स्वतंत्रता", "शिक्षक"],
        6: ["पहाड़", "साहस"],
        7: ["शिष्टता", "आपदा"],
        8: ["समृद्धि", "रणनीति"],
        9: ["दर्शन", "सिम्फनी"],
        10: ["परिवर्तन", "सार"]
    },
    "ar": {
        1: ["قطة", "كلب"],
        2: ["تفاحة", "كتاب"],
        3: ["عائلة", "نهر"],
        4: ["رحلة", "نافذة"],
        5: ["حرية", "معلم"],
        6: ["جبل", "شجاعة"],
        7: ["أناقة", "كارثة"],
        8: ["وفرة", "استراتيجية"],
        9: ["فلسفة", "سيمفونية"],
        10: ["تحول", "جوهر"]
    },
    "bn": {
        1: ["বিড়াল", "কুকুর"],
        2: ["আপেল", "বই"],
        3: ["পরিবার", "নদী"],
        4: ["ভ্রমণ", "জানালা"],
        5: ["স্বাধীনতা", "শিক্ষক"],
        6: ["পর্বত", "সাহস"],
        7: ["নিখুঁত", "দুর্যোগ"],
        8: ["প্রাচুর্য", "কৌশল"],
        9: ["দর্শন", "সিনফনি"],
        10: ["রূপান্তর", "সারমর্ম"]
    },
    "pt": {
        1: ["gato", "cão"],
        2: ["maçã", "livro"],
        3: ["família", "rio"],
        4: ["viagem", "janela"],
        5: ["liberdade", "professor"],
        6: ["montanha", "coragem"],
        7: ["elegância", "desastre"],
        8: ["abundância", "estratégia"],
        9: ["filosofia", "sinfonia"],
        10: ["metamorfose", "quintessência"]
    },
    "ru": {
        1: ["кот", "собака"],
        2: ["яблоко", "книга"],
        3: ["семья", "река"],
        4: ["путешествие", "окно"],
        5: ["свобода", "учитель"],
        6: ["гора", "мужество"],
        7: ["элегантность", "катастрофа"],
        8: ["изобилие", "стратегия"],
        9: ["философия", "симфония"],
        10: ["метаморфоза", "квинтэссенция"]
    },
    "ja": {
        1: ["ねこ", "いぬ"],
        2: ["りんご", "ほん"],
        3: ["かぞく", "かわ"],
        4: ["たび", "まど"],
        5: ["じゆう", "せんせい"],
        6: ["やま", "ゆうき"],
        7: ["ゆうが", "さいがい"],
        8: ["ほうふ", "せんりゃく"],
        9: ["てつがく", "こうきょうきょく"],
        10: ["へんしん", "しんずい"]
    },
    "de": {
        1: ["Katze", "Hund"],
        2: ["Apfel", "Buch"],
        3: ["Familie", "Fluss"],
        4: ["Reise", "Fenster"],
        5: ["Freiheit", "Lehrer"],
        6: ["Berg", "Mut"],
        7: ["Eleganz", "Katastrophe"],
        8: ["Überfluss", "Strategie"],
        9: ["Philosophie", "Sinfonie"],
        10: ["Metamorphose", "Quintessenz"]
    },
    "fr": {
        1: ["chat", "chien"],
        2: ["pomme", "livre"],
        3: ["famille", "rivière"],
        4: ["voyage", "fenêtre"],
        5: ["liberté", "professeur"],
        6: ["montagne", "courage"],
        7: ["élégance", "désastre"],
        8: ["abondance", "stratégie"],
        9: ["philosophie", "symphonie"],
        10: ["métamorphose", "quintessence"]
    },
    "it": {
        1: ["gatto", "cane"],
        2: ["mela", "libro"],
        3: ["famiglia", "fiume"],
        4: ["viaggio", "finestra"],
        5: ["libertà", "insegnante"],
        6: ["montagna", "coraggio"],
        7: ["eleganza", "disastro"],
        8: ["abbondanza", "strategia"],
        9: ["filosofia", "sinfonia"],
        10: ["metamorfosi", "quintessenza"]
    },
    "tr": {
        1: ["kedi", "köpek"],
        2: ["elma", "kitap"],
        3: ["aile", "nehir"],
        4: ["yolculuk", "pencere"],
        5: ["özgürlük", "öğretmen"],
        6: ["dağ", "cesaret"],
        7: ["zarafet", "felaket"],
        8: ["bolluk", "strateji"],
        9: ["felsefe", "senfoni"],
        10: ["metamorfoz", "öz"]
    },
    "ko": {
        1: ["고양이", "개"],
        2: ["사과", "책"],
        3: ["가족", "강"],
        4: ["여행", "창문"],
        5: ["자유", "선생님"],
        6: ["산", "용기"],
        7: ["우아함", "재난"],
        8: ["풍부", "전략"],
        9: ["철학", "교향곡"],
        10: ["변신", "정수"]
    },
    "vi": {
        1: ["mèo", "chó"],
        2: ["táo", "sách"],
        3: ["gia đình", "sông"],
        4: ["hành trình", "cửa sổ"],
        5: ["tự do", "giáo viên"],
        6: ["núi", "dũng cảm"],
        7: ["thanh lịch", "thảm họa"],
        8: ["dồi dào", "chiến lược"],
        9: ["triết học", "giao hưởng"],
        10: ["biến hình", "tinh túy"]
    },
    "pl": {
        1: ["kot", "pies"],
        2: ["jabłko", "książka"],
        3: ["rodzina", "rzeka"],
        4: ["podróż", "okno"],
        5: ["wolność", "nauczyciel"],
        6: ["góra", "odwaga"],
        7: ["elegancja", "katastrofa"],
        8: ["obfitość", "strategia"],
        9: ["filozofia", "symfonia"],
        10: ["metamorfoza", "kwintesencja"]
    },
    "nl": {
        1: ["kat", "hond"],
        2: ["appel", "boek"],
        3: ["familie", "rivier"],
        4: ["reis", "raam"],
        5: ["vrijheid", "leraar"],
        6: ["berg", "moed"],
        7: ["elegantie", "ramp"],
        8: ["overvloed", "strategie"],
        9: ["filosofie", "symfonie"],
        10: ["metamorfose", "kwintessens"]
    },
    "th": {
        1: ["แมว", "สุนัข"],
        2: ["แอปเปิ้ล", "หนังสือ"],
        3: ["ครอบครัว", "แม่น้ำ"],
        4: ["การเดินทาง", "หน้าต่าง"],
        5: ["เสรีภาพ", "ครู"],
        6: ["ภูเขา", "ความกล้าหาญ"],
        7: ["ความสง่างาม", "ภัยพิบัติ"],
        8: ["ความอุดมสมบูรณ์", "กลยุทธ์"],
        9: ["ปรัชญา", "ซิมโฟนี"],
        10: ["การเปลี่ยนแปลง", "แก่นแท้"]
    },
    "fa": {
        1: ["گربه", "سگ"],
        2: ["سیب", "کتاب"],
        3: ["خانواده", "رودخانه"],
        4: ["سفر", "پنجره"],
        5: ["آزادی", "معلم"],
        6: ["کوه", "شجاعت"],
        7: ["ظرافت", "فاجعه"],
        8: ["وفور", "استراتژی"],
        9: ["فلسفه", "سمفونی"],
        10: ["دگرگونی", "جوهر"]
    },
    "id": {
        1: ["kucing", "anjing"],
        2: ["apel", "buku"],
        3: ["keluarga", "sungai"],
        4: ["perjalanan", "jendela"],
        5: ["kebebasan", "guru"],
        6: ["gunung", "keberanian"],
        7: ["keanggunan", "bencana"],
        8: ["kelimpahan", "strategi"],
        9: ["filsafat", "simfoni"],
        10: ["metamorfosis", "intisari"]
    },
    "uk": {
        1: ["кіт", "собака"],
        2: ["яблуко", "книга"],
        3: ["родина", "річка"],
        4: ["подорож", "вікно"],
        5: ["свобода", "вчитель"],
        6: ["гора", "сміливість"],
        7: ["елегантність", "катастрофа"],
        8: ["достаток", "стратегія"],
        9: ["філософія", "симфонія"],
        10: ["метаморфоза", "квінтесенція"]
    },
    "el": {
        1: ["γάτα", "σκύλος"],
        2: ["μήλο", "βιβλίο"],
        3: ["οικογένεια", "ποτάμι"],
        4: ["ταξίδι", "παράθυρο"],
        5: ["ελευθερία", "δασκάλα"],
        6: ["βουνό", "θάρρος"],
        7: ["κομψότητα", "καταστροφή"],
        8: ["αφθονία", "στρατηγική"],
        9: ["φιλοσοφία", "συμφωνία"],
        10: ["μεταμόρφωση", "πεμπτουσία"]
    },
    "sv": {
        1: ["katt", "hund"],
        2: ["äpple", "bok"],
        3: ["familj", "flod"],
        4: ["resa", "fönster"],
        5: ["frihet", "lärare"],
        6: ["berg", "mod"],
        7: ["elegans", "katastrof"],
        8: ["överflöd", "strategi"],
        9: ["filosofi", "symfoni"],
        10: ["metamorfos", "kvintessens"]
    },
    "ro": {
        1: ["pisică", "câine"],
        2: ["măr", "carte"],
        3: ["familie", "râu"],
        4: ["călătorie", "fereastră"],
        5: ["libertate", "profesor"],
        6: ["munte", "curaj"],
        7: ["eleganță", "dezastru"],
        8: ["abundență", "strategie"],
        9: ["filosofie", "simfonie"],
        10: ["metamorfoză", "esență"]
    },
    "hu": {
        1: ["macska", "kutya"],
        2: ["alma", "könyv"],
        3: ["család", "folyó"],
        4: ["utazás", "ablak"],
        5: ["szabadság", "tanár"],
        6: ["hegy", "bátorság"],
        7: ["elegancia", "katasztrófa"],
        8: ["bőség", "stratégia"],
        9: ["filozófia", "szimfónia"],
        10: ["metamorfózis", "kvintesszencia"]
    },
    "cs": {
        1: ["kočka", "pes"],
        2: ["jablko", "kniha"],
        3: ["rodina", "řeka"],
        4: ["cesta", "okno"],
        5: ["svoboda", "učitel"],
        6: ["hora", "odvaha"],
        7: ["elegance", "katastrofa"],
        8: ["hojnost", "strategie"],
        9: ["filosofie", "symfonie"],
        10: ["metamorfóza", "kvintesence"]
    },
    "fi": {
        1: ["kissa", "koira"],
        2: ["omena", "kirja"],
        3: ["perhe", "joki"],
        4: ["matka", "ikkuna"],
        5: ["vapaus", "opettaja"],
        6: ["vuori", "rohkeus"],
        7: ["eleganssi", "katastrofi"],
        8: ["runsaus", "strategia"],
        9: ["filosofia", "sinfonia"],
        10: ["muodonmuutos", "olemus"]
    },
    "da": {
        1: ["kat", "hund"],
        2: ["æble", "bog"],
        3: ["familie", "flod"],
        4: ["rejse", "vindue"],
        5: ["frihed", "lærer"],
        6: ["bjerg", "mod"],
        7: ["elegance", "katastrofe"],
        8: ["overflod", "strategi"],
        9: ["filosofi", "symfoni"],
        10: ["metamorfose", "kvintessens"]
    },
    "he": {
        1: ["חתול", "כלב"],
        2: ["תפוח", "ספר"],
        3: ["משפחה", "נהר"],
        4: ["מסע", "חלון"],
        5: ["חירות", "מורה"],
        6: ["הר", "אומץ"],
        7: ["אלגנטיות", "אסון"],
        8: ["שפע", "אסטרטגיה"],
        9: ["פילוסופיה", "סימפוניה"],
        10: ["מטמורפוזה", "מהות"]
    },
    "no": {
        1: ["katt", "hund"],
        2: ["eple", "bok"],
        3: ["familie", "elv"],
        4: ["reise", "vindu"],
        5: ["frihet", "lærer"],
        6: ["fjell", "mot"],
        7: ["eleganse", "katastrofe"],
        8: ["overflod", "strategi"],
        9: ["filosofi", "symfoni"],
        10: ["metamorfose", "kvintessens"]
    },
    "ms": {
        1: ["kucing", "anjing"],
        2: ["epal", "buku"],
        3: ["keluarga", "sungai"],
        4: ["perjalanan", "tingkap"],
        5: ["kebebasan", "guru"],
        6: ["gunung", "keberanian"],
        7: ["keanggunan", "bencana"],
        8: ["kelimpahan", "strategi"],
        9: ["falsafah", "simfoni"],
        10: ["metamorfosis", "intipati"]
    },
    "ta": {
        1: ["பூனை", "நாய்"],
        2: ["ஆப்பிள்", "புத்தகம்"],
        3: ["குடும்பம்", "ஆறு"],
        4: ["பயணம்", "ஜன்னல்"],
        5: ["சுதந்திரம்", "ஆசிரியர்"],
        6: ["மலை", "தைரியம்"],
        7: ["நயமிகு", "பேரழிவு"],
        8: ["வசதி", "ยุทธศาสตร์"],
        9: ["தத்துவம்", "சிம்பொனி"],
        10: ["மாற்றம்", "சாரம்"]
    },
    "mr": {
        1: ["मांजर", "कुत्रा"],
        2: ["सफरचंद", "पुस्तक"],
        3: ["कुटुंब", "नदी"],
        4: ["प्रवास", "खिडकी"],
        5: ["स्वातंत्र्य", "शिक्षक"],
        6: ["डोंगर", "धैर्य"],
        7: ["शालीनता", "आपत्ती"],
        8: ["समृद्धी", "रणनीती"],
        9: ["तत्त्वज्ञान", "सिंफनी"],
        10: ["रूपांतरण", "सार"]
    },
    "ur": {
        1: ["بلی", "کتا"],
        2: ["سیب", "کتاب"],
        3: ["خاندان", "دریا"],
        4: ["سفر", "کھڑکی"],
        5: ["آزادی", "استاد"],
        6: ["پہاڑ", "حوصلہ"],
        7: ["نفاست", "آفت"],
        8: ["کثرت", "حکمت عملی"],
        9: ["فلسفہ", "سمفنی"],
        10: ["تبدیلی", "جوہر"]
    },
    "te": {
        1: ["పిల్లి", "కుక్క"],
        2: ["ఆపిల్", "పుస్తకం"],
        3: ["కుటుంబం", "నది"],
        4: ["ప్రయాణం", "జనాలా"],
        5: ["స్వేచ్ఛ", "అధ్యాపకుడు"],
        6: ["పర్వతం", "ధైర్యం"],
        7: ["అందం", "విపత్తు"],
        8: ["సమృద్ధి", "విధానం"],
        9: ["తత్వశాస్త్రం", "సింఫనీ"],
        10: ["రూపాంతరం", "సారం"]
    },
    "ml": {
        1: ["പൂച്ച", "നായ"],
        2: ["ആപ്പിൾ", "പുസ്തകം"],
        3: ["കുടുംബം", "നദി"],
        4: ["യാത്ര", "ജാലകം"],
        5: ["സ്വാതന്ത്ര്യം", "അധ്യാപകന്‍"],
        6: ["പർവ്വതം", "ധൈര്യം"],
        7: ["സൗന്ദര്യം", "ദുരന്തം"],
        8: ["സമൃദ്ധി", "യുക്തി"],
        9: ["തത്ത്വചിന്ത", "സിംഫോണി"],
        10: ["പരിണാമം", "സാരം"]
    },
    "sw": {
        1: ["paka", "mbwa"],
        2: ["tofaa", "kitabu"],
        3: ["familia", "mto"],
        4: ["safari", "dirisha"],
        5: ["uhuru", "mwalimu"],
        6: ["mlima", "ujuzi"],
        7: ["haiba", "janga"],
        8: ["wingi", "mbinu"],
        9: ["falsafa", "simfoni"],
        10: ["mabadiliko", "kiini"]
    },
    "bg": {
        1: ["котка", "куче"],
        2: ["ябълка", "книга"],
        3: ["семейство", "река"],
        4: ["пътуване", "прозорец"],
        5: ["свобода", "учител"],
        6: ["планина", "смелост"],
        7: ["елегантност", "бедствие"],
        8: ["изобилие", "стратегия"],
        9: ["философия", "симфония"],
        10: ["метаморфоза", "квинтесенция"]
    },
    "hr": {
        1: ["mačka", "pas"],
        2: ["jabuka", "knjiga"],
        3: ["obitelj", "rijeka"],
        4: ["putovanje", "prozor"],
        5: ["sloboda", "učitelj"],
        6: ["planina", "hrabrost"],
        7: ["elegancija", "katastrofa"],
        8: ["obilje", "strategija"],
        9: ["filozofija", "simfonija"],
        10: ["metamorfoza", "kvintesencija"]
    }
}

all_langs = list(lang_words.keys())
num_sentences = 1000
min_langs = 5
max_langs = 10

output = []

for _ in range(num_sentences):
    n_langs = random.randint(min_langs, max_langs)
    chosen_langs = random.sample(all_langs, n_langs)
    spans = []
    text_fragments = []
    for lang in chosen_langs:
        level = random.randint(1, 10)
        word = random.choice(lang_words[lang][level])
        spans.append({"text": word, "lang": lang})
        text_fragments.append(word)
    mixed = " ".join(text_fragments)
    output.append({
        "text": mixed,
        "spans": spans
    })

with open("multilang_sentences1.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"Generated {num_sentences} sentences in multilang_sentences1.json")
import json
from polylang_detector import detect_languages
from collections import Counter, defaultdict
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

MAX_BYTES = 480 * 1024  # 480 KB per file

def normalize(text):
    return text.strip().replace(" ", "").lower()

def write_split_txt_report(lines, base_name="evaluation_report_part", max_bytes=MAX_BYTES, encoding="utf-8"):
    part = 1
    current_bytes = 0
    fname = f"{base_name}{part}.txt"
    current_file = open(fname, "w", encoding=encoding)
    for line in lines:
        if not line.endswith("\n"):
            line = line + "\n"
        line_bytes = len(line.encode(encoding))
        if current_bytes + line_bytes > max_bytes:
            current_file.close()
            part += 1
            fname = f"{base_name}{part}.txt"
            current_file = open(fname, "w", encoding=encoding)
            current_bytes = 0
        current_file.write(line)
        current_bytes += line_bytes
    current_file.close()
    print(f"Wrote {part} part files.")

def analyze_alignment(gold_spans, pred_spans):
    gold_set = set(gold_spans)
    pred_set = set(pred_spans)
    errors = []
    gold_texts = {normalize(t): l for t, l in gold_spans}
    pred_texts = {normalize(t): l for t, l in pred_spans}
    for text, lang in gold_spans:
        norm_text = normalize(text)
        if (norm_text, lang) not in pred_set:
            if norm_text in [normalize(t) for t, _ in pred_spans]:
                pred_lang = pred_texts[norm_text]
                errors.append(('wrong_lang', lang, pred_lang, norm_text))
            else:
                errors.append(('missing', lang, '', norm_text))
    for text, lang in pred_spans:
        norm_text = normalize(text)
        if (norm_text, lang) not in gold_set:
            if norm_text not in [normalize(t) for t, _ in gold_spans]:
                errors.append(('extra', '', lang, norm_text))
    return errors

def main():
    with open("multilang_sentences1.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)

    y_true = []
    y_pred = []
    mismatches = []
    error_patterns = defaultdict(list)
    lang_error_counter = defaultdict(Counter)

    exact = 0
    partial = 0
    total = len(dataset)
    partial_count = 0
    none_count = 0

    all_gold_langs = []
    all_pred_langs = []

    for sample in tqdm(dataset):
        sentence = sample["text"]
        gold_spans = [(normalize(span["text"]), span["lang"]) for span in sample["spans"]]

        pred_spans = detect_languages(sentence)
        pred_spans = [(normalize(text), lang) for text, lang in pred_spans]

        gold_set = set(gold_spans)
        pred_set = set(pred_spans)

        gold_langs = [lang for _, lang in gold_spans]
        pred_langs = [lang for _, lang in pred_spans]
        gold_texts = [text for text, _ in gold_spans]
        pred_texts = [text for text, _ in pred_spans]
        alignment = []
        min_len = min(len(gold_spans), len(pred_spans))
        for i in range(min_len):
            alignment.append({
                "gold_text": gold_texts[i],
                "gold_lang": gold_langs[i],
                "pred_text": pred_texts[i],
                "pred_lang": pred_langs[i]
            })
        for i in range(min_len, len(gold_spans)):
            alignment.append({
                "gold_text": gold_texts[i],
                "gold_lang": gold_langs[i],
                "pred_text": "",
                "pred_lang": ""
            })
        for i in range(min_len, len(pred_spans)):
            alignment.append({
                "gold_text": "",
                "gold_lang": "",
                "pred_text": pred_texts[i],
                "pred_lang": pred_langs[i]
            })

        alignment_errors = analyze_alignment(gold_spans, pred_spans)
        for err_type, gold_l, pred_l, token in alignment_errors:
            lang_error_counter[gold_l or pred_l][err_type] += 1
            error_patterns[err_type].append((sentence, gold_l, pred_l, token))

        if pred_set == gold_set:
            exact += 1
        elif pred_set & gold_set:
            partial += 1
            partial_count += 1
            mismatches.append({
                "text": sentence,
                "type": "partial",
                "alignment": alignment,
                "gold_spans": sample["spans"],
                "predicted_spans": [{"text": t, "lang": l} for t, l in pred_spans],
                "alignment_errors": alignment_errors
            })
        else:
            none_count += 1
            mismatches.append({
                "text": sentence,
                "type": "none",
                "alignment": alignment,
                "gold_spans": sample["spans"],
                "predicted_spans": [{"text": t, "lang": l} for t, l in pred_spans],
                "alignment_errors": alignment_errors
            })

        if len(gold_spans) == len(pred_spans):
            y_true.extend(gold_langs)
            y_pred.extend(pred_langs)
        all_gold_langs.extend(gold_langs)
        all_pred_langs.extend(pred_langs)

    lines = []
    lines.append("==== EVALUATION RESULTS (Summary) ====\n\n")
    if y_true and y_pred:
        labels = sorted(set(y_true + y_pred))
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
        macro_f1 = sum(f1) / len(f1)

        summary = [
            f"Total Sentences   : {total}",
            f"Exact Matches     : {exact} ({exact/total:.2%})",
            f"Partial Matches   : {partial} ({partial/total:.2%})",
            f"No Matches        : {total - exact - partial} ({(total - exact - partial)/total:.2%})",
            "\n==== Per-Language Metrics ===="
        ]
        for i, label in enumerate(labels):
            summary.append(f"{label.upper():<6} | P: {precision[i]:.2f} | R: {recall[i]:.2f} | F1: {f1[i]:.2f}")
        summary.append(f"\nMacro F1 Score: {macro_f1:.2f}\n")
        lines.extend([s + "\n" for s in summary])
        print("\n".join(summary))
    else:
        summary = [
            f"Total Sentences   : {total}",
            f"Exact Matches     : {exact} ({exact/total:.2%})",
            f"Partial Matches   : {partial} ({partial/total:.2%})",
            f"No Matches        : {total - exact - partial} ({(total - exact - partial)/total:.2%})",
            "\nNo valid span-level matches found for precision/recall/f1 calculation."
        ]
        lines.extend([s + "\n" for s in summary])
        print("\n".join(summary))

    gold_counter = Counter(all_gold_langs)
    pred_counter = Counter(all_pred_langs)
    langs = sorted(set(all_gold_langs) | set(all_pred_langs))
    lines.append("==== Per-Language Aggregate Metrics ====\n")
    lines.append("Lang | Precision | Recall | F1 | Gold Count | Pred Count | Correct\n")
    for lang in langs:
        gold = gold_counter[lang]
        pred = pred_counter[lang]
        correct = min(gold, pred)
        precision_value = correct / pred if pred > 0 else 0
        recall_value = correct / gold if gold > 0 else 0
        f1_value = 2 * precision_value * recall_value / (precision_value + recall_value) if (precision_value + recall_value) > 0 else 0
        lines.append(f"{lang.upper():<6} | {precision_value:.2f} | {recall_value:.2f} | {f1_value:.2f} | {gold} | {pred} | {correct}\n")
    lines.append("\n")

    lines.append("==== ERROR ANALYSIS SUMMARY ====\n")
    lines.append("ErrorType | Language | Count\n")
    for lang in sorted(lang_error_counter):
        for err_type, count in lang_error_counter[lang].items():
            lines.append(f"{err_type:<10} | {lang.upper():<6} | {count}\n")
    lines.append("\n")

    lines.append("Most common wrong_lang errors (Top 10):\n")
    for s, gold_l, pred_l, token in error_patterns['wrong_lang'][:10]:
        lines.append(f"Sentence: {s}\n  Token: '{token}'  Gold: {gold_l}  Pred: {pred_l}\n")
    lines.append("\n")

    lines.append("Most common missing errors (Top 10):\n")
    for s, gold_l, _, token in error_patterns['missing'][:10]:
        lines.append(f"Sentence: {s}\n  Missing Token: '{token}'  Gold: {gold_l}\n")
    lines.append("\n")

    lines.append("Most common extra errors (Top 10):\n")
    for s, _, pred_l, token in error_patterns['extra'][:10]:
        lines.append(f"Sentence: {s}\n  Extra Token: '{token}'  Pred: {pred_l}\n")
    lines.append("\n")

    lines.append("==== MATCHED SENTENCES ====\n")
    for sample in dataset:
        sentence = sample["text"]
        in_mismatch = next((m for m in mismatches if m["text"] == sentence), None)
        if not in_mismatch:
            spans = "; ".join([f'{span["lang"]}: {span["text"]}' for span in sample["spans"]])
            lines.append(f"- {sentence}\n  Spans: {spans}\n")
    lines.append("\n")

    lines.append("==== UNMATCHED SENTENCES ====\n")
    for m in mismatches:
        align_txt = "\n    ".join([
            f'Gold ({a["gold_lang"]}): {a["gold_text"]} | Pred ({a["pred_lang"]}): {a["pred_text"]}'
            for a in m.get("alignment", [])
        ])
        gold = "; ".join([f'{span["lang"]}: {span["text"]}' for span in m["gold_spans"]])
        pred = "; ".join([f'{span["lang"]}: {span["text"]}' for span in m.get("predicted_spans", [])])
        lines.append(
            f"- {m['text']}\n  Gold: {gold}\n  Pred: {pred}\n  Alignment:\n    {align_txt}\n  Type: {m.get('type','unknown')}\n"
        )
    lines.append("\n")

    # DETAILED ERROR ANALYSIS (Space-separated version, no tabs)
    lines.append("==== DETAILED ERROR ANALYSIS (Space-separated) ====\n")
    lines.append("ErrorType GoldLang PredLang Token Sentence\n")
    for err_type, err_list in error_patterns.items():
        for sentence, gold_l, pred_l, token in err_list:
            sentence_clean = sentence.replace('\n', ' ').replace('\t', ' ')
            token_clean = token.replace('\n', ' ').replace('\t', ' ')
            # All fields separated by single space, no tabs
            lines.append(f"{err_type} {gold_l} {pred_l} {token_clean} {sentence_clean}\n")
    lines.append("\n")

    write_split_txt_report(lines)

    print("\nText report generated and split as needed (evaluation_report_partN.txt, with detailed error analysis included at the end)")

if __name__ == "__main__":
    main()
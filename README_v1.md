# poly-langid v1

A multilingual language identification system using FastText and script-aware heuristics.

## Features
- Token-level language identification for over 40 languages
- Script-aware tokenization (Han, Hangul, Devanagari, etc.)
- FastText-based language prediction with custom thresholds
- Regex and stopword hints for low-resource languages
- Chunk-level fallback for Han script (Chinese/Japanese)
- Handles code-switching and mixed-script sentences

## Usage
1. Place your FastText model (`lid.176.ftz`) in the project directory.
2. Run `evaluate.py` to evaluate the system and generate reports.
3. Main logic is in `polylang_detector.py`.

## Evaluation
- Reports are generated as `evaluation_report_partN.txt`.
- Summary metrics: Macro F1, per-language F1, recall, precision, and error analysis.

## Project Structure
- `polylang_detector.py`: Main language detection logic
- `evaluate.py`: Evaluation script
- `lid.176.ftz`: FastText language ID model
- `requirements.txt`: Python dependencies
- `rig.txt`: Summarized evaluation results
- `doc/`: Documentation

## Version
This is the v1 baseline for further development and comparison.

---

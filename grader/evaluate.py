"""
Evaluate all keyword detectors on cached Whisper transcriptions.

Run after grader/tune.py:
    python -m grader.evaluate

Output
------
    Console: macro-averaged comparison table + per-keyword F1 grid
    grader/data/results.json: full confusion matrices and per-keyword metrics

Detectors evaluated
-------------------
    LinearSearch               — exact string match
    Phonetic(k)                — ARPAbet Levenshtein, integer threshold
    NormalizedPhonetic(ratio)  — Levenshtein / len(keyword) threshold
    MSPhonetic(k)              — Microsoft PhoneticMatching continuous metric
    ConfusionWeighted(k)       — data-driven substitution costs (requires
                                 grader/data/phoneme_confusion.json)

Thresholds are loaded from grader/data/thresholds.json (produced by tune.py).
Falls back to config.py defaults if thresholds.json is missing.
"""

import json
import os
import re
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import PHONETIC_K, PHONETIC_K_RATIO, MS_PHONETIC_K
from classes.detector import (
    KeywordDetection, ProcessedKeywords,
    PhoneticKeywordDetector,
    NormalizedPhoneticDetector,
    MicrosoftPhoneticDetector,
    ConfusionWeightedPhoneticDetector,
)

TRANSCRIPTIONS_PATH   = os.path.join(os.path.dirname(__file__), "data", "transcriptions_validation.jsonl")
CONFUSION_MATRIX_PATH = os.path.join(os.path.dirname(__file__), "data", "phoneme_confusion.json")
THRESHOLDS_PATH       = os.path.join(os.path.dirname(__file__), "data", "thresholds.json")
RESULTS_PATH          = os.path.join(os.path.dirname(__file__), "data", "results.json")

KEYWORD_WORDS = [
    "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
]


# ---------------------------------------------------------------------------
# Thin wrapper so LinearSearch accepts an explicit keyword list at eval time
# (the real LinearSearch reads from config.KEYWORDS which is the live system's
# personal keyword list, not the full Speech Commands vocabulary).
# ---------------------------------------------------------------------------
class _EvalLinearSearch:
    def __init__(self, keywords: list[str]) -> None:
        self._kw_set = {kw.lower() for kw in keywords}

    def detect_keyword(self, phrase: str) -> ProcessedKeywords:
        words = re.findall(r'\b\w+\b', phrase)
        hits = [KeywordDetection(w, 1.0) for w in words if w.lower() in self._kw_set]
        return ProcessedKeywords(hits)


def _load_thresholds() -> dict[str, float]:
    if os.path.exists(THRESHOLDS_PATH):
        with open(THRESHOLDS_PATH) as f:
            thresholds = json.load(f)
        print(f"Loaded tuned thresholds from {THRESHOLDS_PATH}")
        return thresholds
    print(f"  Note: {THRESHOLDS_PATH} not found — using config.py defaults.")
    print("         Run `python -m grader.tune` after transcribing the train split.\n")
    return {}


def _make_detectors(keywords: list[str], thresholds: dict[str, float]) -> dict[str, Any]:
    k_phonetic   = int(thresholds.get("Phonetic",          PHONETIC_K))
    r_norm       = thresholds.get("NormPhonetic",           PHONETIC_K_RATIO)
    k_ms         = thresholds.get("MSPhonetic",             MS_PHONETIC_K)
    k_confusion  = thresholds.get("ConfusionWeighted",      float(PHONETIC_K))

    detectors: dict[str, Any] = {
        "LinearSearch":              _EvalLinearSearch(keywords),
        f"Phonetic(k={k_phonetic})": PhoneticKeywordDetector(keywords, k_phonetic),
        f"NormPhonetic(r={r_norm})": NormalizedPhoneticDetector(keywords, r_norm),
        f"MSPhonetic(k={k_ms})":     MicrosoftPhoneticDetector(keywords, k_ms),
    }

    if os.path.exists(CONFUSION_MATRIX_PATH):
        detectors[f"ConfusionWeighted(k={k_confusion})"] = ConfusionWeightedPhoneticDetector(
            keywords, k_confusion, CONFUSION_MATRIX_PATH
        )
    else:
        print(f"  Note: {CONFUSION_MATRIX_PATH} not found — ConfusionWeighted skipped.")
        print("         Run `python -m grader.build_confusion` to enable it.\n")

    return detectors


def _load_transcriptions(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def _evaluate_detector(
    name: str,
    detector: Any,
    records: list[dict],
    keywords: list[str],
) -> dict:
    """
    Run detector on every record.  Returns confusion matrix + per-keyword metrics.

    Confusion matrix rows = true_word, columns = detected_word | "none".
    When multiple keywords are detected, the highest-confidence one is chosen.
    """
    all_labels = keywords + ["none"]
    kw_set = set(keywords)
    # rows include "none" so unknown-clip false positives are counted
    confusion: dict[str, dict[str, int]] = {w: {l: 0 for l in all_labels} for w in all_labels}

    for rec in records:
        true_word = rec.get("true_word", "")
        if true_word not in confusion:
            continue

        transcription = rec.get("transcription", "")
        result = detector.detect_keyword(transcription)

        hits = result.keywords
        if hits:
            best = max(hits, key=lambda d: d.confidence)
            detected = best.pharse.lower()
            if detected not in kw_set:
                detected = "none"
        else:
            detected = "none"

        confusion[true_word][detected] += 1

    # Per-keyword precision / recall / F1
    per_keyword: dict[str, dict] = {}
    for kw in keywords:
        tp = confusion[kw][kw]
        # FP: other keywords OR unknown clips mislabelled as kw
        fp = sum(confusion[other][kw] for other in all_labels if other != kw)
        fn = sum(confusion[kw][l] for l in all_labels if l != kw)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec_ = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec_ / (prec + rec_) if (prec + rec_) > 0 else 0.0
        per_keyword[kw] = {"precision": prec, "recall": rec_, "f1": f1,
                           "tp": tp, "fp": fp, "fn": fn}

    macro_p  = sum(m["precision"] for m in per_keyword.values()) / len(per_keyword)
    macro_r  = sum(m["recall"]    for m in per_keyword.values()) / len(per_keyword)
    macro_f1 = sum(m["f1"]        for m in per_keyword.values()) / len(per_keyword)

    return {
        "confusion_matrix": confusion,
        "per_keyword": per_keyword,
        "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
    }


def _print_comparison_table(results: dict[str, dict]) -> None:
    w = 42
    print("\n" + "=" * (w + 33))
    print("DETECTOR COMPARISON  (macro-averaged over all keywords)")
    print("=" * (w + 33))
    print(f"{'Detector':<{w}} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * (w + 33))
    # Sort by macro F1 descending
    for name, result in sorted(results.items(), key=lambda x: -x[1]["macro"]["f1"]):
        m = result["macro"]
        print(f"{name:<{w}} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>10.3f}")
    print("=" * (w + 33))


def _print_per_keyword_table(results: dict[str, dict], keywords: list[str]) -> None:
    names = list(results.keys())
    col_w = 10
    row_w = 12
    total_w = row_w + (col_w + 1) * len(names)

    print("\n" + "=" * total_w)
    print("PER-KEYWORD F1 SCORES")
    print("=" * total_w)

    header = f"{'Keyword':<{row_w}}"
    for n in names:
        abbrev = n.split("(")[0][:col_w]
        header += f" {abbrev:>{col_w}}"
    print(header)
    print("-" * total_w)

    for kw in keywords:
        row = f"{kw:<{row_w}}"
        for n in names:
            f1 = results[n]["per_keyword"][kw]["f1"]
            row += f" {f1:>{col_w}.3f}"
        print(row)

    print("=" * total_w)


def main() -> None:
    if not os.path.exists(TRANSCRIPTIONS_PATH):
        print(f"ERROR: {TRANSCRIPTIONS_PATH} not found.")
        print("Run: python -m grader.transcribe --split validation")
        sys.exit(1)

    records = _load_transcriptions(TRANSCRIPTIONS_PATH)
    print(f"Loaded {len(records)} transcription records.")

    records = [r for r in records if r.get("true_word") in KEYWORD_WORDS or r.get("true_word") == "none"]
    kw_count  = sum(1 for r in records if r["true_word"] != "none")
    neg_count = sum(1 for r in records if r["true_word"] == "none")
    print(f"Kept   {len(records)} records  ({kw_count} keyword, {neg_count} negative).")

    word_counts = {}
    for r in records:
        word_counts[r["true_word"]] = word_counts.get(r["true_word"], 0) + 1
    print(f"Keywords covered: {len([w for w in word_counts if w != 'none'])} / {len(KEYWORD_WORDS)}")
    print()

    thresholds = _load_thresholds()
    print("Initialising detectors...")
    detectors = _make_detectors(KEYWORD_WORDS, thresholds)
    print()

    results: dict[str, dict] = {}
    for name, detector in detectors.items():
        print(f"  Evaluating {name}...")
        results[name] = _evaluate_detector(name, detector, records, KEYWORD_WORDS)

    _print_comparison_table(results)
    _print_per_keyword_table(results, KEYWORD_WORDS)

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results (confusion matrices + per-keyword metrics) → {RESULTS_PATH}")


if __name__ == "__main__":
    main()

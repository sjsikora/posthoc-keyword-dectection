"""
Tune detector thresholds on train-split Whisper transcriptions.

Run after:
    python -m grader.transcribe --split train --samples 50

Output
------
    grader/data/thresholds.json       — best threshold per detector
    grader/data/threshold_curves.json — macro F1 at each candidate (for plots)
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from classes.detector import (
    KeywordDetection, ProcessedKeywords,
    PhoneticKeywordDetector,
    NormalizedPhoneticDetector,
    MicrosoftPhoneticDetector,
    ConfusionWeightedPhoneticDetector,
)

TRAIN_PATH        = os.path.join(os.path.dirname(__file__), "data", "transcriptions_train.jsonl")
CONFUSION_PATH    = os.path.join(os.path.dirname(__file__), "data", "phoneme_confusion.json")
THRESHOLDS_PATH   = os.path.join(os.path.dirname(__file__), "data", "thresholds.json")
CURVES_PATH       = os.path.join(os.path.dirname(__file__), "data", "threshold_curves.json")

KEYWORD_WORDS = [
    "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
]

CANDIDATES: dict[str, list[float]] = {
    "Phonetic":          [0, 1, 2, 3],
    "NormPhonetic":      [0.10, 0.20, 0.25, 0.33, 0.40, 0.50],
    "MSPhonetic":        [0.25, 0.50, 0.75, 1.00, 1.25, 1.50],
    "ConfusionWeighted": [0.5, 1.0, 1.5, 2.0, 2.5],
}


def _load_records(path: str) -> list[dict]:
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
    return [r for r in records if r.get("true_word") in KEYWORD_WORDS or r.get("true_word") == "none"]


def _macro_f1(detector, records: list[dict], keywords: list[str]) -> float:
    kw_set = set(keywords)
    all_labels = keywords + ["none"]
    # rows include "none" so unknown-clip false positives are counted
    confusion = {w: {l: 0 for l in all_labels} for w in all_labels}

    for rec in records:
        true_word = rec.get("true_word", "")
        if true_word not in confusion:
            continue
        result = detector.detect_keyword(rec.get("transcription", ""))
        hits = result.keywords
        if hits:
            best = max(hits, key=lambda d: d.confidence)
            detected = best.pharse.lower()
            if detected not in kw_set:
                detected = "none"
        else:
            detected = "none"
        confusion[true_word][detected] += 1

    f1s = []
    for kw in keywords:
        tp = confusion[kw][kw]
        # FP: other keywords mislabelled as kw, AND unknown clips mislabelled as kw
        fp = sum(confusion[o][kw] for o in all_labels if o != kw)
        fn = sum(confusion[kw][l] for l in all_labels if l != kw)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
    return sum(f1s) / len(f1s)


def tune(records: list[dict], keywords: list[str]) -> tuple[dict, dict]:
    has_confusion = os.path.exists(CONFUSION_PATH)

    builders = {
        "Phonetic":     lambda k: PhoneticKeywordDetector(keywords, int(k)),
        "NormPhonetic": lambda k: NormalizedPhoneticDetector(keywords, k),
        "MSPhonetic":   lambda k: MicrosoftPhoneticDetector(keywords, k),
    }
    if has_confusion:
        builders["ConfusionWeighted"] = lambda k: ConfusionWeightedPhoneticDetector(
            keywords, k, CONFUSION_PATH
        )
    else:
        print("  Note: confusion matrix not found — skipping ConfusionWeighted.\n"
              "         Run `python -m grader.build_confusion` first.")

    best_thresholds: dict[str, float] = {}
    curves: dict[str, dict[str, float]] = {}

    for name, builder in builders.items():
        candidates = CANDIDATES[name]
        curve: dict[str, float] = {}

        print(f"\n  {name}")
        print(f"  {'Threshold':>12}  {'Macro F1':>10}")
        print(f"  {'-'*12}  {'-'*10}")

        for k in candidates:
            f1 = _macro_f1(builder(k), records, keywords)
            curve[str(k)] = round(f1, 6)
            print(f"  {k:>12}  {f1:>10.4f}")

        best_k = max(curve, key=lambda x: curve[x])
        best_thresholds[name] = float(best_k)
        curves[name] = curve
        print(f"  → best threshold: {best_k}  (F1 = {curve[best_k]:.4f})")

    return best_thresholds, curves


def main() -> None:
    if not os.path.exists(TRAIN_PATH):
        print(f"ERROR: {TRAIN_PATH} not found.")
        print("Run: python -m grader.transcribe --split train --samples 50")
        sys.exit(1)

    records = _load_records(TRAIN_PATH)
    print(f"Loaded {len(records)} train transcriptions "
          f"({len({r['true_word'] for r in records})} words covered).\n")
    print("Sweeping thresholds on train split...")

    best, curves = tune(records, KEYWORD_WORDS)

    os.makedirs(os.path.dirname(THRESHOLDS_PATH), exist_ok=True)
    with open(THRESHOLDS_PATH, "w") as f:
        json.dump(best, f, indent=2)
    with open(CURVES_PATH, "w") as f:
        json.dump(curves, f, indent=2)

    print(f"\nBest thresholds → {THRESHOLDS_PATH}")
    print(f"Threshold curves → {CURVES_PATH}")
    print("\nSummary:")
    for name, k in best.items():
        print(f"  {name:<22} k = {k}")



if __name__ == "__main__":
    main()

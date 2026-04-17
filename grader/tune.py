"""
Tune detector thresholds on train-split Whisper transcriptions.

Run after:
    python -m grader.transcribe --split train --samples 50

Output
------
    grader/data/thresholds.json       — best threshold per detector
    grader/data/threshold_curves.json — macro F1 at each candidate (for plots)
"""

import argparse
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
    "Phonetic":          [1, 2, 3],
    "NormPhonetic":      [0.10, 0.20, 0.25, 0.30, 0.40, 0.50],
    "MSPhonetic":        [0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00],
    "ConfusionWeighted": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
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


def _macro_metrics(detector, records: list[dict], keywords: list[str]) -> tuple[float, float, float]:
    """Returns (macro_precision, macro_recall, macro_f1)."""
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

    precisions, recalls, f1s = [], [], []
    for kw in keywords:
        tp = confusion[kw][kw]
        fp = sum(confusion[o][kw] for o in all_labels if o != kw)
        fn = sum(confusion[kw][l] for l in all_labels if l != kw)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
    n = len(keywords)
    return sum(precisions) / n, sum(recalls) / n, sum(f1s) / n


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
        print(f"  {'Threshold':>12}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}")
        print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}")

        for k in candidates:
            p, r, f1 = _macro_metrics(builder(k), records, keywords)
            curve[str(k)] = {"precision": round(p, 6), "recall": round(r, 6), "f1": round(f1, 6)}
            print(f"  {k:>12}  {p:>10.4f}  {r:>10.4f}  {f1:>10.4f}")

        best_k = max(curve, key=lambda x: curve[x]["f1"])
        best_thresholds[name] = float(best_k)
        curves[name] = curve
        print(f"  → best threshold: {best_k}  (F1 = {curve[best_k]['f1']:.4f})")

    return best_thresholds, curves


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep detector thresholds on cached transcriptions.")
    parser.add_argument("--transcriptions", default=TRAIN_PATH,
                        help="Path to transcriptions JSONL (default: train split).")
    parser.add_argument("--curves-out", default=CURVES_PATH,
                        help="Where to write threshold_curves JSON (default: threshold_curves.json).")
    parser.add_argument("--no-save-thresholds", action="store_true",
                        help="Skip writing thresholds.json (useful when sweeping validation data).")
    args = parser.parse_args()

    if not os.path.exists(args.transcriptions):
        print(f"ERROR: {args.transcriptions} not found.")
        print("Run: python -m grader.transcribe --split train --samples 50")
        sys.exit(1)

    records = _load_records(args.transcriptions)
    print(f"Loaded {len(records)} transcriptions "
          f"({len({r['true_word'] for r in records})} words covered).\n")
    print(f"Sweeping thresholds on {args.transcriptions}...")

    best, curves = tune(records, KEYWORD_WORDS)

    os.makedirs(os.path.dirname(args.curves_out), exist_ok=True)
    with open(args.curves_out, "w") as f:
        json.dump(curves, f, indent=2)
    print(f"Threshold curves → {args.curves_out}")

    if not args.no_save_thresholds:
        with open(THRESHOLDS_PATH, "w") as f:
            json.dump(best, f, indent=2)
        print(f"Best thresholds  → {THRESHOLDS_PATH}")

    print("\nSummary:")
    for name, k in best.items():
        print(f"  {name:<22} k = {k}")



if __name__ == "__main__":
    main()

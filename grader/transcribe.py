"""
Transcribe Google Speech Commands clips with Whisper and cache results to disk.

Run once before tune.py / evaluate.py:
    python -m grader.transcribe --split train --samples 50
    python -m grader.transcribe --split validation --samples 50

Output
------
    grader/data/transcriptions_{split}.jsonl
    One JSON object per line:
        {
            "true_word":     "yes",      # keyword, or "none" for unknown clips
            "transcription": "yes",
            "confidence":    -0.12,
            "split":         "validation"
        }

Unknown clips (true_word="none") serve as hard negatives during threshold tuning
and evaluation — any detection on these counts as a false positive.
"""

import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from classes.stt import WhisperModel
from config import WINDOW_SIZE, CHANNELS

KEYWORD_WORDS = [
    "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
]

NEGATIVE_WORDS = [
    "bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila", "tree", "wow",
]

NEGATIVE_KEY = "none"


def _output_path(split: str) -> str:
    return os.path.join(os.path.dirname(__file__), "data", f"transcriptions_{split}.jsonl")


def _load_already_done(path: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not os.path.exists(path):
        return counts
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                word = rec.get("true_word", "")
                counts[word] = counts.get(word, 0) + 1
            except json.JSONDecodeError:
                pass
    return counts


def _pad_to_window(audio_array: np.ndarray) -> np.ndarray:
    audio_2d = audio_array.reshape(-1, CHANNELS).astype(np.float32)
    if len(audio_2d) >= WINDOW_SIZE:
        return audio_2d[:WINDOW_SIZE]
    pad = np.zeros((WINDOW_SIZE - len(audio_2d), CHANNELS), dtype=np.float32)
    return np.vstack([audio_2d, pad])


def _write_record(out_file, true_word: str, result, split: str) -> str:
    clean = re.sub(r"[^\w\s]", "", result.pharse).strip().lower()
    record = {
        "true_word":     true_word,
        "transcription": clean,
        "confidence":    result.confidence,
        "split":         split,
    }
    out_file.write(json.dumps(record) + "\n")
    out_file.flush()
    return clean


def transcribe(samples_per_word: int, unknown_samples: int, split: str) -> None:
    print(f"Loading Speech Commands v0.01 ({split} split)...")
    from datasets import load_dataset

    ds = load_dataset(
        "google/speech_commands", "v0.01",
        split=split,
        trust_remote_code=True,
    )
    label_names: list[str] = ds.features["label"].names

    output_path = _output_path(split)
    already_done = _load_already_done(output_path)

    kw_target  = len(KEYWORD_WORDS) * samples_per_word
    kw_already = sum(already_done.get(w, 0) for w in KEYWORD_WORDS)
    unk_already = already_done.get(NEGATIVE_KEY, 0)

    if kw_already >= kw_target and unk_already >= unknown_samples:
        print(f"Cache already complete ({kw_already} keyword + {unk_already} negative clips). Nothing to do.")
        print(f"Delete {output_path} and rerun to rebuild from scratch.")
        return

    print(f"Keyword clips:  {kw_already} / {kw_target}")
    print(f"Negative clips: {unk_already} / {unknown_samples}")
    print("Loading Whisper...")
    whisper = WhisperModel()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_file = open(output_path, "a")

    word_counts   = dict(already_done)
    skipped_whisper = 0
    written = 0

    try:
        for example in ds:
            label_str: str = label_names[example["label"]]

            is_keyword  = label_str in KEYWORD_WORDS
            is_negative = label_str in NEGATIVE_WORDS

            if not is_keyword and not is_negative:
                continue

            if is_keyword  and word_counts.get(label_str,    0) >= samples_per_word:
                continue
            if is_negative and word_counts.get(NEGATIVE_KEY, 0) >= unknown_samples:
                continue

            audio  = _pad_to_window(np.array(example["audio"]["array"]))
            result = whisper.transcribe_audio_chunk(audio)

            if result is None or not result.pharse.strip():
                skipped_whisper += 1
                continue

            true_word = label_str if is_keyword else NEGATIVE_KEY

            _write_record(out_file, true_word, result, split)

            word_counts[true_word] = word_counts.get(true_word, 0) + 1
            written += 1

            kw_done  = sum(word_counts.get(w, 0) for w in KEYWORD_WORDS)
            unk_done = word_counts.get(NEGATIVE_KEY, 0)
            total    = kw_done + unk_done
            if total % 25 == 0:
                print(f"  {kw_done}/{kw_target} keyword   {unk_done}/{unknown_samples} unknown   "
                      f"({skipped_whisper} skips)")

            kw_done_flag  = all(word_counts.get(w, 0) >= samples_per_word for w in KEYWORD_WORDS)
            unk_done_flag = word_counts.get(NEGATIVE_KEY, 0) >= unknown_samples
            if kw_done_flag and unk_done_flag:
                break

    finally:
        out_file.close()

    print(f"\nDone. Wrote {written} new records to {output_path}")
    print(f"Whisper skips: {skipped_whisper}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache Whisper transcriptions of Speech Commands for evaluation."
    )
    parser.add_argument("--samples", type=int, default=50,
                        help="Keyword clips per word (default: 50).")
    parser.add_argument("--unknown", type=int, default=None,
                        help="Unknown (negative) clips to collect (default: same as --samples).")
    parser.add_argument("--split", default="validation",
                        choices=["train", "validation", "test"],
                        help="Dataset split (default: validation).")
    args = parser.parse_args()
    unknown_samples = args.unknown if args.unknown is not None else args.samples
    transcribe(args.samples, unknown_samples, args.split)


if __name__ == "__main__":
    main()

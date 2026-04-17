"""
Transcribe Google Speech Commands clips with Whisper and cache results to disk.

Run once before tune.py / evaluate.py:
    python -m grader.transcribe --split train --samples 50 --unknown 600
    python -m grader.transcribe --split validation --samples 50 --unknown 600

Noise experiment — transcribe validation at multiple SNR levels:
    python -m grader.transcribe --split validation --samples 50 --unknown 600 --snr 20
    python -m grader.transcribe --split validation --samples 50 --unknown 600 --snr 10
    python -m grader.transcribe --split validation --samples 50 --unknown 600 --snr 0

Output
------
    grader/data/transcriptions_{split}.jsonl          (clean)
    grader/data/transcriptions_{split}_snr{N}.jsonl   (noisy)

Unknown clips (true_word="none") serve as hard negatives — any detection on
these counts as a false positive during threshold tuning and evaluation.
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

_DIGIT_TO_WORD = {
    "0": "zero", "1": "one",   "2": "two",   "3": "three", "4": "four",
    "5": "five", "6": "six",   "7": "seven", "8": "eight", "9": "nine",
}


def _output_path(split: str, snr_db: float | None) -> str:
    suffix = f"_snr{int(snr_db)}" if snr_db is not None else ""
    return os.path.join(os.path.dirname(__file__), "data", f"transcriptions_{split}{suffix}.jsonl")


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


def _add_noise(audio: np.ndarray, snr_db: float) -> np.ndarray:
    """Add Gaussian white noise at the given SNR (dB) relative to signal power."""
    signal_power = np.mean(audio ** 2)
    if signal_power == 0:
        return audio
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), audio.shape).astype(np.float32)
    return np.clip(audio + noise, -1.0, 1.0)


def _write_record(out_file, true_word: str, result, split: str, snr_db: float | None) -> None:
    clean = re.sub(r"[^\w\s]", "", result.pharse).strip().lower()
    clean = re.sub(r"\b(\d)\b", lambda m: _DIGIT_TO_WORD.get(m.group(1), m.group(1)), clean)
    record = {
        "true_word":     true_word,
        "transcription": clean,
        "confidence":    result.confidence,
        "split":         split,
        "snr_db":        snr_db,
    }
    out_file.write(json.dumps(record) + "\n")
    out_file.flush()


def transcribe(samples_per_word: int, unknown_samples: int, split: str, snr_db: float | None) -> None:
    snr_label = f"SNR={snr_db}dB" if snr_db is not None else "clean"
    print(f"Loading Speech Commands v0.01 ({split} split, {snr_label})...")
    from datasets import load_dataset

    ds = load_dataset(
        "google/speech_commands", "v0.01",
        split=split,
        trust_remote_code=True,
    )
    label_names: list[str] = ds.features["label"].names

    output_path = _output_path(split, snr_db)
    already_done = _load_already_done(output_path)

    kw_target   = len(KEYWORD_WORDS) * samples_per_word
    kw_already  = sum(already_done.get(w, 0) for w in KEYWORD_WORDS)
    unk_already = already_done.get(NEGATIVE_KEY, 0)

    if kw_already >= kw_target and unk_already >= unknown_samples:
        print(f"Cache already complete ({kw_already} keyword + {unk_already} negative clips). Nothing to do.")
        print(f"Delete {output_path} and rerun to rebuild from scratch.")
        return

    print(f"Keyword clips:  {kw_already} / {kw_target}")
    print(f"Negative clips: {unk_already} / {unknown_samples}")
    print("Loading Whisper...")
    whisper = WhisperModel(log_prob_threshold=-float("inf"))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_file = open(output_path, "a")

    word_counts     = dict(already_done)
    skipped_whisper = 0
    written         = 0

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

            audio = _pad_to_window(np.array(example["audio"]["array"]))
            if snr_db is not None:
                audio = _add_noise(audio, snr_db)

            result = whisper.transcribe_audio_chunk(audio)
            if result is None or not result.pharse.strip():
                skipped_whisper += 1
                continue

            true_word = label_str if is_keyword else NEGATIVE_KEY
            _write_record(out_file, true_word, result, split, snr_db)

            word_counts[true_word] = word_counts.get(true_word, 0) + 1
            written += 1

            kw_done  = sum(word_counts.get(w, 0) for w in KEYWORD_WORDS)
            unk_done = word_counts.get(NEGATIVE_KEY, 0)
            if (kw_done + unk_done) % 25 == 0:
                print(f"  {kw_done}/{kw_target} keyword   {unk_done}/{unknown_samples} negative   "
                      f"({skipped_whisper} skips)")

            if all(word_counts.get(w, 0) >= samples_per_word for w in KEYWORD_WORDS) \
                    and word_counts.get(NEGATIVE_KEY, 0) >= unknown_samples:
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
                        help="Negative clips to collect (default: same as --samples).")
    parser.add_argument("--split", default="validation",
                        choices=["train", "validation", "test"],
                        help="Dataset split (default: validation).")
    parser.add_argument("--snr", type=float, default=None,
                        help="Add Gaussian noise at this SNR in dB (default: no noise).")
    args = parser.parse_args()
    unknown_samples = args.unknown if args.unknown is not None else args.samples
    transcribe(args.samples, unknown_samples, args.split, args.snr)


if __name__ == "__main__":
    main()

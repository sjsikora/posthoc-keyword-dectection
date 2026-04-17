"""
Build a data-driven phoneme confusion matrix from Whisper's errors on the
Google Speech Commands dataset (v0.01 — the same corpus used by TensorFlow
Datasets: https://www.tensorflow.org/datasets/catalog/speech_commands).

Loaded via HuggingFace `datasets` (already a project dependency) which
wraps the identical audio files and labels.

Usage
-----
    python -m grader.build_confusion               # default: 100 samples/word
    python -m grader.build_confusion --samples 50  # faster, noisier matrix
    python -m grader.build_confusion --samples 500 # slower, more reliable

Output
------
    grader/data/phoneme_confusion.json
        Nested dict  { ref_phone: { hyp_phone: count, ... }, ... }
        where ref_phone is the ground-truth ARPAbet phoneme and hyp_phone is
        what Whisper transcribed.

Algorithm
---------
For each labeled audio clip:
  1. Run Whisper → get transcription string.
  2. Look up ARPAbet phonemes for ground-truth word and transcription word.
  3. Compute optimal Levenshtein alignment (phoneme_alignment).
  4. For every (ref, hyp) pair in the alignment:
       - matches    → increment matrix[ref][ref]
       - substitutions → increment matrix[ref][hyp]
       - deletions  → increment matrix[ref]["_DEL_"]
       - insertions → increment matrix["_INS_"][hyp]
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

# Ensure project root is on the path when run as `python -m grader.build_confusion`
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from classes.stt import WhisperModel
from classes.phonetic import word_to_phonemes, phoneme_alignment

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "data", "phoneme_confusion.json")

# All word labels in Speech Commands v0.01
SPEECH_COMMANDS_WORDS = [
    "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila", "tree", "wow",
]


def build(samples_per_word: int) -> dict[str, dict[str, int]]:
    print("Loading Google Speech Commands v0.01 via HuggingFace datasets...")
    from datasets import load_dataset  # import here so it's optional at module level

    ds = load_dataset(
        "google/speech_commands",
        "v0.01",
        split="train",          # train split — keeps validation clean for evaluate.py
        trust_remote_code=True,
    )

    # Build label → word lookup from the dataset's ClassLabel feature
    label_names: list[str] = ds.features["label"].names

    print("Loading Whisper...")
    whisper = WhisperModel()

    # matrix[ref_phone][hyp_phone] = count
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    # Track how many samples we've processed per word
    word_counts: dict[str, int] = defaultdict(int)

    skipped_no_cmu     = 0
    skipped_whisper    = 0
    skipped_limit      = 0
    total_aligned_pairs = 0

    print(f"Processing up to {samples_per_word} samples per word "
          f"({len(SPEECH_COMMANDS_WORDS)} words x {samples_per_word} = "
          f"~{len(SPEECH_COMMANDS_WORDS) * samples_per_word} clips)...\n")

    for example in ds:
        label_str: str = label_names[example["label"]]

        # Skip words not in our target list (e.g. "unknown", "silence")
        if label_str not in SPEECH_COMMANDS_WORDS:
            continue

        # Stop collecting once we have enough samples for this word
        if word_counts[label_str] >= samples_per_word:
            skipped_limit += 1
            continue

        # Ground-truth ARPAbet phonemes
        ref_phones = word_to_phonemes(label_str)
        if ref_phones is None:
            skipped_no_cmu += 1
            continue

        # Prepare audio for Whisper — Speech Commands is 16 kHz mono int16
        audio_array = np.array(example["audio"]["array"], dtype=np.float32)
        audio_2d    = audio_array.reshape(-1, 1)   # (samples, channels=1)

        # Whisper needs at least ~0.5 s; pad short clips with silence
        min_samples = 8000
        if len(audio_2d) < min_samples:
            pad = np.zeros((min_samples - len(audio_2d), 1), dtype=np.float32)
            audio_2d = np.vstack([audio_2d, pad])

        result = whisper.transcribe_audio_chunk(audio_2d)
        if result is None or not result.pharse.strip():
            skipped_whisper += 1
            continue

        # Use the first word Whisper produced (clips contain one word)
        transcribed_word = result.pharse.strip().split()[0].lower()
        hyp_phones = word_to_phonemes(transcribed_word)
        if hyp_phones is None:
            skipped_whisper += 1
            continue

        # Align and accumulate
        alignment = phoneme_alignment(ref_phones, hyp_phones)
        for ref_p, hyp_p in alignment:
            if ref_p is None:
                matrix["_INS_"][hyp_p] += 1
            elif hyp_p is None:
                matrix[ref_p]["_DEL_"] += 1
            else:
                matrix[ref_p][hyp_p] += 1
            total_aligned_pairs += 1

        word_counts[label_str] += 1

        # Progress indicator
        total_done = sum(word_counts.values())
        if total_done % 50 == 0:
            print(f"  {total_done} clips processed  "
                  f"({total_aligned_pairs} aligned phoneme pairs so far)")

    print(f"\nDone.")
    print(f"  Clips processed:        {sum(word_counts.values())}")
    print(f"  Aligned phoneme pairs:  {total_aligned_pairs}")
    print(f"  Skipped (no CMU entry): {skipped_no_cmu}")
    print(f"  Skipped (Whisper fail): {skipped_whisper}")
    print(f"  Skipped (quota full):   {skipped_limit}")

    # Convert defaultdict → plain dict for JSON serialisation
    return {ref: dict(hyps) for ref, hyps in matrix.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a phoneme confusion matrix from Whisper on Speech Commands."
    )
    parser.add_argument(
        "--samples", type=int, default=100,
        help="Number of audio clips to process per word (default: 100).",
    )
    args = parser.parse_args()

    matrix = build(args.samples)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(matrix, f, indent=2)

    print(f"\nSaved confusion matrix → {OUTPUT_PATH}")
    print(f"Unique ref phonemes recorded: {len(matrix)}")


if __name__ == "__main__":
    main()

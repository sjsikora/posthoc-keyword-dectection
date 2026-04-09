SAMPLE_RATE = 16000 # Whisper strictly requires 16kHz audio
CHANNELS = 1 # Mono audio
BLOCK_SIZE = SAMPLE_RATE * 3 # Process the audio in three second chunks

"""
Keywords should only be one single word. It is un defined behavior (and unwise in general)
to have contractions (like I'm or don't) in the keywords.
"""
KEYWORDS = [
    "pineapple",
    "waffle iron",
    "lavender",
    "beam",
    "rock"
]

# Maximum phoneme edit distance for a transcribed word to be considered a keyword match.
# k=0: exact phoneme match only. k=1: one phoneme off. k=2: two phonemes off.
#
# NOTE: Unnormalized edit distance is sensitive to keyword length. A short keyword
# like "beam" (3 phonemes) at k=2 matches ~any CVC word (name, Sam, am, Dean...).
# k=1 is a safer default; for research, consider PHONETIC_K_RATIO below instead.
PHONETIC_K = 1

# Normalized threshold: dist / len(keyword_phonemes) <= PHONETIC_K_RATIO
# 0.33 = at most 1/3 of keyword phonemes wrong, regardless of word length.
# This is length-invariant and a more principled research baseline.
PHONETIC_K_RATIO = 0.33

# Threshold for MicrosoftPhoneticDetector. Units are continuous (Euclidean distance
# in 3D phoneme feature space), not integer phoneme counts.
# k=0.5 ≈ one close consonant substitution (e.g. P↔B costs 0.25)
# k=1.0 ≈ several near-phoneme differences or one vowel insertion (cost 0.5)
MS_PHONETIC_K = 0.5

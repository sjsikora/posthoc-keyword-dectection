from dataclasses import dataclass
from abc import abstractmethod
from config import KEYWORDS
from classes.phonetic import word_to_phonemes, phoneme_distance
from classes.ms_phonetic.vectors import arpabet_to_vectors, PhonemeVector
from classes.ms_phonetic.distance import phonetic_distance as ms_phonetic_distance
import math
import re

@dataclass
class KeywordDetection():
    pharse: str
    confidence : float

@dataclass
class ProcessedKeywords():
    keywords : list[KeywordDetection]

class KeywordDetector():
    @abstractmethod
    def detect_keyword(self, phrase: str) -> ProcessedKeywords:
        pass

class LinearSearch():
    def detect_keyword(self, phrase: str) -> ProcessedKeywords:
        keywords_detected = []

        # Split sentence into words, keeping them clean of punctuation
        words = re.findall(r'\b\w+\b', phrase)

        for word in words:
            if word.lower() in KEYWORDS:
                keywords_detected.append(KeywordDetection(word, 1))

        return ProcessedKeywords(keywords_detected)


class PhoneticKeywordDetector(KeywordDetector):
    """
    Detects keywords by comparing words in phonetic space rather than by exact string match.
    Each transcribed word is converted to its ARPAbet phoneme sequence and compared against
    pre-computed keyword phoneme sequences. A word is considered a match if its phoneme
    edit distance to any keyword is within the threshold k.

    k=0 requires an identical phoneme sequence (catches spelling variants of the same sound).
    k=1 allows one phoneme substitution/insertion/deletion (catches close homophones).
    k=2+ catches increasingly approximate matches at the cost of more false positives.
    """

    def __init__(self, keywords: list[str], k: int) -> None:
        self.k = k
        self.keyword_phonemes: dict[str, list[str]] = {}
        for kw in keywords:
            phones = word_to_phonemes(kw)
            if phones is not None:
                self.keyword_phonemes[kw] = phones
            else:
                print(f"Warning: '{kw}' not found in CMU pronouncing dictionary — skipping phonetic matching for it.")

    def detect_keyword(self, phrase: str) -> ProcessedKeywords:
        words = re.findall(r'\b\w+\b', phrase)
        detections = []

        for word in words:
            word_phones = word_to_phonemes(word)
            if word_phones is None:
                continue

            for kw, kw_phones in self.keyword_phonemes.items():
                dist = phoneme_distance(word_phones, kw_phones)
                if dist <= self.k:
                    # confidence=1.0 at dist=0, approaches 0 as dist approaches k+1
                    confidence = 1.0 - dist / (self.k + 1)
                    detections.append(KeywordDetection(kw, confidence))

        return ProcessedKeywords(detections)


class NormalizedPhoneticDetector(KeywordDetector):
    """
    Variant of PhoneticKeywordDetector that uses *normalized* edit distance:

        dist / len(keyword_phonemes) <= ratio

    This makes the threshold length-invariant: a ratio of 0.33 means "at most
    1 in 3 keyword phonemes may differ", whether the keyword is 3 or 10 phonemes.

    Without normalization, short keywords (e.g. "beam" = 3 phonemes) at k=2
    match most CVC words in the language ("name", "Sam", "am", "Dean" all pass).
    With ratio=0.33 those false positives are eliminated because dist/3 > 0.33.
    """

    def __init__(self, keywords: list[str], ratio: float) -> None:
        self.ratio = ratio
        self.keyword_phonemes: dict[str, list[str]] = {}
        for kw in keywords:
            phones = word_to_phonemes(kw)
            if phones is not None:
                self.keyword_phonemes[kw] = phones
            else:
                print(f"Warning: '{kw}' not found in CMU pronouncing dictionary — skipping.")

    def detect_keyword(self, phrase: str) -> ProcessedKeywords:
        words = re.findall(r'\b\w+\b', phrase)
        detections = []

        for word in words:
            word_phones = word_to_phonemes(word)
            if word_phones is None:
                continue

            for kw, kw_phones in self.keyword_phonemes.items():
                dist = phoneme_distance(word_phones, kw_phones)
                normalized = dist / len(kw_phones)
                if normalized <= self.ratio:
                    confidence = 1.0 - normalized / (self.ratio + (1 / len(kw_phones)))
                    detections.append(KeywordDetection(kw, confidence))

        return ProcessedKeywords(detections)



def _word_to_ms_vectors(word: str) -> list[PhonemeVector] | None:
    """CMU dict → stress-stripped ARPAbet → list[PhonemeVector] for MS distance."""
    arpabet = word_to_phonemes(word)
    if arpabet is None:
        return None
    vectors: list[PhonemeVector] = []
    for phone in arpabet:
        expanded = arpabet_to_vectors(phone)
        if not expanded:
            return None
        vectors.extend(expanded)
    return vectors


class MicrosoftPhoneticDetector(KeywordDetector):
    """
    Keyword detector using the phonetic distance metric from:

        Microsoft PhoneticMatching (2021)
        https://github.com/microsoft/PhoneticMatching

    Algorithm described in:
        Li & MacWhinney (2002). PatPho: A phonological pattern generator for neural networks.
        http://blclab.org/wp-content/uploads/2013/02/patpho.pdf

    Each ARPAbet phoneme is mapped to a 3D feature vector encoding articulatory
    properties (see classes/ms_phonetic/vectors.py). Distance is a weighted
    Levenshtein where substitution cost = Euclidean distance between vectors, and
    insertion/deletion cost = 0.5 (syllabic) or 0.25 (non-syllabic).

    Unlike PhoneticKeywordDetector, phonetically similar substitutions (e.g. P↔B,
    voicing-only difference, cost ≈ 0.25) are penalised less than dissimilar ones
    (e.g. P↔Z, cost ≈ 0.35). The threshold k is continuous, not integer.
    """

    def __init__(self, keywords: list[str], k: float) -> None:
        self.k = k
        self.keyword_vectors: dict[str, list[PhonemeVector]] = {}
        for kw in keywords:
            vecs = _word_to_ms_vectors(kw)
            if vecs is not None:
                self.keyword_vectors[kw] = vecs
            else:
                print(f"Warning: '{kw}' could not be vectorised — skipping.")

    def detect_keyword(self, phrase: str) -> ProcessedKeywords:
        words = re.findall(r'\b\w+\b', phrase)
        detections = []

        for word in words:
            word_vecs = _word_to_ms_vectors(word)
            if not word_vecs:
                continue

            for kw, kw_vecs in self.keyword_vectors.items():
                dist = ms_phonetic_distance(word_vecs, kw_vecs)
                if dist <= self.k:
                    confidence = max(0.0, 1.0 - dist / (self.k + 1))
                    detections.append(KeywordDetection(kw, confidence))

        return ProcessedKeywords(detections)


class ConfusionWeightedPhoneticDetector(KeywordDetector):
    """
    Phonetic keyword detector whose substitution costs come from a data-driven
    phoneme confusion matrix built by grader/build_confusion.py.

    Instead of treating every phoneme substitution as cost-1 (uniform Levenshtein),
    this detector uses empirically measured Whisper error rates:

        cost(ref_phone, hyp_phone) = 1 - P(hyp | ref)
                                   = 1 - count(ref→hyp) / total_ref_occurrences

    If Whisper commonly transcribes keyword phoneme S as Z (e.g. in noisy audio),
    then cost(S, Z) ≈ 0, so a word containing Z still matches an S-keyword cheaply.

    Falls back gracefully to uniform cost = 1 if the matrix file does not yet exist
    (run `python -m grader.build_confusion` first to generate it).

    The threshold k is in the same units as PhoneticKeywordDetector (cumulative
    weighted edit distance, where each substitution contributes in [0, 1]).
    """

    def __init__(self, keywords: list[str], k: float, matrix_path: str) -> None:
        self.k = k
        self.keyword_phonemes: dict[str, list[str]] = {}

        from grader.confusion_matrix import load_matrix, build_cost_table
        raw = load_matrix(matrix_path)
        self._cost_table = build_cost_table(raw)

        if not raw:
            print(f"Warning: confusion matrix not found at '{matrix_path}'. "
                  "Run `python -m grader.build_confusion` to generate it. "
                  "Falling back to uniform substitution costs.")

        for kw in keywords:
            phones = word_to_phonemes(kw)
            if phones is not None:
                self.keyword_phonemes[kw] = phones
            else:
                print(f"Warning: '{kw}' not found in CMU dict — skipping.")

    def _weighted_distance(self, a: list[str], b: list[str]) -> float:
        """
        Levenshtein with data-driven substitution costs.
        Insertion/deletion cost is uniform = 1 (same as PhoneticKeywordDetector).
        """
        from grader.confusion_matrix import substitution_cost
        m, n = len(a), len(b)
        dp = [float(j) for j in range(n + 1)]
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = float(i)
            for j in range(1, n + 1):
                temp = dp[j]
                sub  = prev + substitution_cost(a[i - 1], b[j - 1], self._cost_table)
                dp[j] = min(sub, dp[j] + 1.0, dp[j - 1] + 1.0)
                prev = temp
        return dp[n]

    def detect_keyword(self, phrase: str) -> ProcessedKeywords:
        words = re.findall(r'\b\w+\b', phrase)
        detections = []

        for word in words:
            word_phones = word_to_phonemes(word)
            if word_phones is None:
                continue
            for kw, kw_phones in self.keyword_phonemes.items():
                dist = self._weighted_distance(word_phones, kw_phones)
                if dist <= self.k:
                    confidence = max(0.0, 1.0 - dist / (self.k + 1))
                    detections.append(KeywordDetection(kw, confidence))

        return ProcessedKeywords(detections)

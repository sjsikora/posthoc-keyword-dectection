import cmudict
import re

_CMU_DICT: dict[str, list[list[str]]] = cmudict.dict()


def word_to_phonemes(word: str) -> list[str] | None:
    """
    Convert a word to its ARPAbet phoneme sequence using the CMU Pronouncing Dictionary.
    Stress markers are stripped (AH0, AH1, AH2 all become AH) so distance is purely
    about phoneme identity, not prosody.
    Returns None if the word is not in the dictionary.
    """
    entries = _CMU_DICT.get(word.lower())
    if not entries:
        return None
    # Use the first (most common) pronunciation, strip numeric stress markers
    return [re.sub(r'\d+$', '', p) for p in entries[0]]


def phoneme_distance(a: list[str], b: list[str]) -> int:
    """
    Levenshtein edit distance between two phoneme sequences.
    Each phoneme is treated as an atomic unit (not character-by-character).
    """
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]

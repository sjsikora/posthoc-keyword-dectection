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


def phoneme_alignment(ref: list[str], hyp: list[str]) -> list[tuple[str | None, str | None]]:
    """
    Compute the optimal Levenshtein alignment between two phoneme sequences and
    return the explicit edit path as a list of (ref_phone, hyp_phone) pairs:

        ('AH', 'AH') — match (same phoneme)
        ('AH', 'IH') — substitution (ref was heard as hyp)
        ('AH', None) — deletion  (ref phoneme absent in hyp)
        (None, 'IH') — insertion (hyp phoneme not in ref)

    Used by grader/build_confusion.py to accumulate the data-driven phoneme
    confusion matrix from Whisper's transcription errors on labeled audio.
    """
    m, n = len(ref), len(hyp)

    # Build the full DP table (space-inefficient but needed for traceback)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])

    # Traceback from dp[m][n] to dp[0][0]
    alignment: list[tuple[str | None, str | None]] = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            alignment.append((ref[i - 1], hyp[j - 1]))
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            alignment.append((ref[i - 1], hyp[j - 1]))   # substitution
            i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            alignment.append((ref[i - 1], None))           # deletion
            i -= 1
        else:
            alignment.append((None, hyp[j - 1]))           # insertion
            j -= 1

    alignment.reverse()
    return alignment


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

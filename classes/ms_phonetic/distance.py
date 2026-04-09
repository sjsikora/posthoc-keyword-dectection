# Faithful Python translation of:
#   microsoft/PhoneticMatching — src/maluuba/speech/phoneticdistance/metric.cpp
#   https://github.com/microsoft/PhoneticMatching/blob/master/src/maluuba/speech/phoneticdistance/metric.cpp
#
# Implements PhonemeDistance (substitution cost) and PhonemeCost (indel cost)
# exactly as defined in that file, then applies them via a weighted Levenshtein,
# corresponding to PhoneticDistance::operator().

import math
from .vectors import PhonemeVector


def substitution_cost(a: PhonemeVector, b: PhonemeVector) -> float:
    """
    Euclidean (L2) distance between two PhonemeVectors.

    Source: metric.cpp — struct PhonemeDistance::operator()
        auto sum_sq = 0.0;
        for (int i = 0; i < 3; ++i) {
            float diff = a[i] - b[i];
            sum_sq += diff * diff;
        }
        return std::sqrt(sum_sq);
    """
    return math.sqrt((a.v0 - b.v0) ** 2 + (a.v1 - b.v1) ** 2 + (a.v2 - b.v2) ** 2)


def indel_cost(phone: PhonemeVector) -> float:
    """
    Insertion / deletion cost for a single phoneme.

    Source: metric.cpp — struct PhonemeCost::operator()
        if (phoneme.is_syllabic()) return 0.5;
        else                       return 0.25;

    Syllabic phonemes (vowels, syllabic consonants) are penalised more heavily
    because they are perceptually more salient than non-syllabic consonants.
    """
    return 0.5 if phone.is_syllabic else 0.25


def phonetic_distance(a: list[PhonemeVector], b: list[PhonemeVector]) -> float:
    """
    Weighted Levenshtein distance over sequences of PhonemeVectors.

    Source: metric.cpp — PhoneticDistance::operator() delegates to
        LevenshteinDistance<PhonemeDistance, PhonemeCost>

    The standard Levenshtein DP recurrence, using real-valued costs:
        dp[i][j] = min(
            dp[i-1][j-1] + substitution_cost(a[i], b[j]),  # substitute
            dp[i-1][j]   + indel_cost(a[i]),               # delete from a
            dp[i][j-1]   + indel_cost(b[j]),               # insert from b
        )

    Unlike integer Levenshtein, substitution of phonetically similar phones
    (e.g. P↔B, which differ only in voicing: |1.0-0.75|=0.25) costs much less
    than substituting dissimilar phones (e.g. P↔Z, cost≈0.354).
    """
    m, n = len(a), len(b)

    # Initialise first row: cost of inserting all of b[:j]
    dp = [0.0] * (n + 1)
    for j in range(1, n + 1):
        dp[j] = dp[j - 1] + indel_cost(b[j - 1])

    for i in range(1, m + 1):
        prev_row_prev = dp[0]           # dp[i-1][j-1] before any update
        dp[0] = dp[0] + indel_cost(a[i - 1])   # cost of deleting a[:i]

        for j in range(1, n + 1):
            temp = dp[j]               # save dp[i-1][j] before overwriting

            substitute = prev_row_prev + substitution_cost(a[i - 1], b[j - 1])
            delete     = temp          + indel_cost(a[i - 1])
            insert     = dp[j - 1]    + indel_cost(b[j - 1])

            dp[j] = min(substitute, delete, insert)
            prev_row_prev = temp

    return dp[n]

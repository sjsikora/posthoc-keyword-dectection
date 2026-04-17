"""
Load and query the data-driven phoneme confusion matrix built by build_confusion.py.

The matrix is stored as a nested JSON dict:
    { "AH": { "AH": 142, "IH": 3, "AE": 1, ... }, "IH": { ... }, ... }

where matrix[ref][hyp] = number of times the ground-truth phoneme `ref` was
transcribed by Whisper as `hyp`.  A high count means Whisper commonly makes
this substitution, so it should cost LESS to align a transcribed `hyp` against
a keyword phoneme `ref`.

Substitution cost formula:
    cost(ref, hyp) = 1 - P(hyp | ref)
                   = 1 - count(ref→hyp) / total_ref_count

    cost(ref, ref) = 0  always (exact match)
    cost(ref, hyp) = 1  when the substitution was never observed (default)
"""

import json
import os
from functools import lru_cache


_DEFAULT_COST = 1.0   # cost for any (ref, hyp) pair not in the matrix


def load_matrix(path: str) -> dict[str, dict[str, int]]:
    """Load a saved confusion matrix JSON.  Returns an empty dict if missing."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def build_cost_table(
    matrix: dict[str, dict[str, int]],
) -> dict[tuple[str, str], float]:
    """
    Pre-compute substitution costs from raw confusion counts.
    Returns a dict keyed by (ref_phone, hyp_phone) → float cost in [0, 1].
    """
    costs: dict[tuple[str, str], float] = {}
    for ref, hyp_counts in matrix.items():
        total = sum(hyp_counts.values())
        if total == 0:
            continue
        for hyp, count in hyp_counts.items():
            costs[(ref, hyp)] = 1.0 - count / total
    return costs


def substitution_cost(
    ref: str,
    hyp: str,
    cost_table: dict[tuple[str, str], float],
) -> float:
    """
    Return the data-driven substitution cost for replacing keyword phoneme `ref`
    with transcribed phoneme `hyp`.  Falls back to 1.0 for unseen pairs.
    """
    if ref == hyp:
        return 0.0
    return cost_table.get((ref, hyp), _DEFAULT_COST)

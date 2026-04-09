# Faithful Python translation of:
#   microsoft/PhoneticMatching — src/maluuba/speech/phoneticdistance/metric.cpp
#   https://github.com/microsoft/PhoneticMatching/blob/master/src/maluuba/speech/phoneticdistance/metric.cpp
#
# Phoneme representation described in:
#   Li & MacWhinney (2002). PatPho: A phonological pattern generator for neural networks.
#   http://blclab.org/wp-content/uploads/2013/02/patpho.pdf
#
# Each ARPAbet phoneme maps to a PhonemeVector: three floats encoding articulatory
# features, plus an is_syllabic flag.  The three dimensions differ by phoneme type:
#
#   Consonants: (phonation, place_of_articulation, manner_of_articulation)
#   Vowels:     (0.1,       vowel_backness,        vowel_height)
#
# Diphthongs (AW, AY, EY, OW, OY) and affricates (CH, JH) expand to two PhonemeVectors,
# matching the Microsoft IPA decomposition in arpabet.cpp (AW → aʊ, CH → tʃ, etc.).

from typing import NamedTuple


class PhonemeVector(NamedTuple):
    v0: float           # phonation  (consonant) | 0.1       (vowel)
    v1: float           # place      (consonant) | backness  (vowel)
    v2: float           # manner     (consonant) | height    (vowel)
    is_syllabic: bool


# ---------------------------------------------------------------------------
# Scalar constants — metric.cpp consonant_to_vector() and vowel_to_vector()
# ---------------------------------------------------------------------------

# Phonation (consonant dimension 0)
_VOICELESS = 1.000   # Phonation::VOICELESS and GLOTTAL_CLOSURE
_VOICED    = 0.750   # all other phonation values

# Place of articulation (consonant dimension 1)
_BILABIAL        = 0.450
_LABIODENTAL     = 0.528
_DENTAL          = 0.606
_ALVEOLAR        = 0.684
_PALATO_ALVEOLAR = 0.762   # also RETROFLEX, ALVEOLO_PALATAL
_PALATAL         = 0.841   # also LABIAL_PALATAL, PALATAL_VELAR
_VELAR           = 0.921   # also LABIAL_VELAR, UVULAR
_GLOTTAL         = 1.000   # also PHARYNGEAL, EPIGLOTTAL

# Manner of articulation (consonant dimension 2)
_NASAL       = 0.644
_PLOSIVE     = 0.733   # also CLICK, IMPLOSIVE, EJECTIVE
_FRICATIVE   = 0.822   # SIBILANT_FRICATIVE and NON_SIBILANT_FRICATIVE share this value
_APPROXIMANT = 0.911   # also FLAP, TRILL
_LATERAL     = 1.000   # LATERAL_FRICATIVE, LATERAL_APPROXIMANT, LATERAL_FLAP

# Vowel dimension 0 is always fixed
_VOW_V0 = 0.100

# Vowel backness (dimension 1)
_FRONT   = 0.100   # FRONT and NEAR_FRONT
_CENTRAL = 0.175
_BACK    = 0.250   # BACK and NEAR_BACK

# Vowel height (dimension 2)
_CLOSE    = 0.100   # CLOSE and NEAR_CLOSE
_CLOSE_MID = 0.185
_MID      = 0.270
_OPEN_MID = 0.355
_OPEN     = 0.444   # OPEN and NEAR_OPEN


# ---------------------------------------------------------------------------
# Helper constructors
# ---------------------------------------------------------------------------

def _c(phonation: float, place: float, manner: float, syllabic: bool = False) -> PhonemeVector:
    """Consonant PhonemeVector."""
    return PhonemeVector(phonation, place, manner, syllabic)


def _v(backness: float, height: float) -> PhonemeVector:
    """Vowel PhonemeVector (v0 always 0.1, always syllabic)."""
    return PhonemeVector(_VOW_V0, backness, height, True)


# ---------------------------------------------------------------------------
# ARPAbet → PhonemeVector(s) table
#
# Source: arpabet.cpp (ARPAbet → IPA) + ipa.cpp (IPA → Phone features)
#         + metric.cpp (Phone features → PhonemeVector)
#
# Each entry is either a single PhonemeVector or a list[PhonemeVector] for
# phonemes that decompose into multiple phones (diphthongs, affricates).
# ---------------------------------------------------------------------------

ARPABET_VECTORS: dict[str, PhonemeVector | list[PhonemeVector]] = {

    # --- Vowels ---
    # arpabet.cpp: AA→ɑ  (back, open)
    "AA": _v(_BACK, _OPEN),
    # arpabet.cpp: AE→æ  (front, near-open → open)
    "AE": _v(_FRONT, _OPEN),
    # arpabet.cpp: AH→ʌ  (back, open-mid)
    "AH": _v(_BACK, _OPEN_MID),
    # arpabet.cpp: AO→ɔ  (back, open-mid)
    "AO": _v(_BACK, _OPEN_MID),
    # arpabet.cpp: AW→aʊ  (front-open + near-back near-close)
    "AW": [_v(_FRONT, _OPEN), _v(_BACK, _CLOSE)],
    # arpabet.cpp: AX→ə  (central, mid)
    "AX": _v(_CENTRAL, _MID),
    # arpabet.cpp: AY→aɪ  (front-open + near-front near-close)
    "AY": [_v(_FRONT, _OPEN), _v(_FRONT, _CLOSE)],
    # arpabet.cpp: EH→ɛ  (front, open-mid)
    "EH": _v(_FRONT, _OPEN_MID),
    # arpabet.cpp: ER→ɝ  (central rhotic, mid)
    "ER": _v(_CENTRAL, _MID),
    # arpabet.cpp: EY→eɪ  (front close-mid + near-front near-close)
    "EY": [_v(_FRONT, _CLOSE_MID), _v(_FRONT, _CLOSE)],
    # arpabet.cpp: IH→ɪ  (near-front, near-close)
    "IH": _v(_FRONT, _CLOSE),
    # arpabet.cpp: IY→i  (front, close)
    "IY": _v(_FRONT, _CLOSE),
    # arpabet.cpp: OW→oʊ  (back close-mid + near-back near-close)
    "OW": [_v(_BACK, _CLOSE_MID), _v(_BACK, _CLOSE)],
    # arpabet.cpp: OY→ɔɪ  (back open-mid + near-front near-close)
    "OY": [_v(_BACK, _OPEN_MID), _v(_FRONT, _CLOSE)],
    # arpabet.cpp: UH→ʊ  (near-back, near-close)
    "UH": _v(_BACK, _CLOSE),
    # arpabet.cpp: UW→u  (back, close)
    "UW": _v(_BACK, _CLOSE),

    # --- Syllabic consonants ---
    # arpabet.cpp: EL→l̩  ENG→ŋ̩  EM→m̩  EN→n̩
    "EL":  _c(_VOICED, _ALVEOLAR,  _LATERAL, syllabic=True),
    "EM":  _c(_VOICED, _BILABIAL,  _NASAL,   syllabic=True),
    "EN":  _c(_VOICED, _ALVEOLAR,  _NASAL,   syllabic=True),
    "ENG": _c(_VOICED, _VELAR,     _NASAL,   syllabic=True),

    # --- Stops / Plosives ---
    "B":  _c(_VOICED,    _BILABIAL, _PLOSIVE),
    "D":  _c(_VOICED,    _ALVEOLAR, _PLOSIVE),
    "G":  _c(_VOICED,    _VELAR,    _PLOSIVE),
    "K":  _c(_VOICELESS, _VELAR,    _PLOSIVE),
    "P":  _c(_VOICELESS, _BILABIAL, _PLOSIVE),
    "T":  _c(_VOICELESS, _ALVEOLAR, _PLOSIVE),
    # arpabet.cpp: Q→ʔ  (glottal stop; phonation=GLOTTAL_CLOSURE→1.0)
    "Q":  _c(_VOICELESS, _GLOTTAL,  _PLOSIVE),

    # --- Affricates — decomposed per arpabet.cpp: CH→tʃ, JH→dʒ ---
    "CH": [_c(_VOICELESS, _ALVEOLAR,        _PLOSIVE),
           _c(_VOICELESS, _PALATO_ALVEOLAR, _FRICATIVE)],
    "JH": [_c(_VOICED,    _ALVEOLAR,        _PLOSIVE),
           _c(_VOICED,    _PALATO_ALVEOLAR, _FRICATIVE)],

    # --- Fricatives ---
    "DH": _c(_VOICED,    _DENTAL,          _FRICATIVE),
    "F":  _c(_VOICELESS, _LABIODENTAL,     _FRICATIVE),
    "HH": _c(_VOICELESS, _GLOTTAL,         _FRICATIVE),
    "S":  _c(_VOICELESS, _ALVEOLAR,        _FRICATIVE),
    "SH": _c(_VOICELESS, _PALATO_ALVEOLAR, _FRICATIVE),
    "TH": _c(_VOICELESS, _DENTAL,          _FRICATIVE),
    "V":  _c(_VOICED,    _LABIODENTAL,     _FRICATIVE),
    "Z":  _c(_VOICED,    _ALVEOLAR,        _FRICATIVE),
    "ZH": _c(_VOICED,    _PALATO_ALVEOLAR, _FRICATIVE),

    # --- Nasals ---
    "M":  _c(_VOICED, _BILABIAL, _NASAL),
    "N":  _c(_VOICED, _ALVEOLAR, _NASAL),
    "NG": _c(_VOICED, _VELAR,    _NASAL),
    # arpabet.cpp: NX→ɾ̃  (nasal flap — approximated as alveolar nasal)
    "NX": _c(_VOICED, _ALVEOLAR, _NASAL),

    # --- Liquids ---
    "L":  _c(_VOICED, _ALVEOLAR,        _LATERAL),
    # arpabet.cpp: R→ɹ  (voiced alveolar approximant)
    "R":  _c(_VOICED, _ALVEOLAR,        _APPROXIMANT),
    # arpabet.cpp: DX→ɾ  (voiced alveolar flap)
    "DX": _c(_VOICED, _ALVEOLAR,        _APPROXIMANT),

    # --- Semivowels ---
    # arpabet.cpp: W→w  (labial-velar approximant)
    "W":  _c(_VOICED, _VELAR,   _APPROXIMANT),
    # arpabet.cpp: Y→j  (palatal approximant)
    "Y":  _c(_VOICED, _PALATAL, _APPROXIMANT),
}


def arpabet_to_vectors(phoneme: str) -> list[PhonemeVector]:
    """
    Expand a single stress-stripped ARPAbet phoneme into one or more PhonemeVectors.
    Returns an empty list for unknown phonemes (with no crash — caller decides how to handle).
    """
    entry = ARPABET_VECTORS.get(phoneme)
    if entry is None:
        return []
    if isinstance(entry, list):
        return entry
    return [entry]

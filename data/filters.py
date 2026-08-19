"""Cheap inline text filters — no offline cleaning pass (constraint #5)."""


def hangul_ratio(text: str) -> float:
    """Fraction of characters that are Hangul syllables/jamo.

    Used to keep Korean-dominant documents while streaming, so we never
    materialize a separate cleaned corpus.
    """
    if not text:
        return 0.0
    n_hangul = 0
    n_letters = 0
    for ch in text:
        if ch.isspace():
            continue
        n_letters += 1
        o = ord(ch)
        if 0xAC00 <= o <= 0xD7A3 or 0x1100 <= o <= 0x11FF or 0x3130 <= o <= 0x318F:
            n_hangul += 1
    return n_hangul / max(n_letters, 1)


def keep_document(text: str, min_hangul_ratio: float, min_chars: int = 200) -> bool:
    if len(text) < min_chars:
        return False
    if min_hangul_ratio > 0 and hangul_ratio(text) < min_hangul_ratio:
        return False
    return True

"""Tokenization untuk Indonesian text.

Indonesian morphology relatif simpel — whitespace + punctuation tokenization
biasanya cukup untuk IR. Untuk advanced case (compound word handling), tim
bisa eksplorasi `nltk.word_tokenize` atau `spaCy Indonesian`.
"""

from __future__ import annotations

import re


# Token = word characters (a-z, 0-9, underscore). Punctuation auto-excluded.
_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")


def simple_tokenize(text: str) -> list[str]:
    """Whitespace + punctuation tokenization."""
    if not text:
        return []
    return _TOKEN_PATTERN.findall(text)

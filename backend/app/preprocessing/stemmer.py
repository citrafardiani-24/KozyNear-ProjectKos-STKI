"""Sastrawi stemmer wrapper dengan LRU cache.

Sastrawi stemming bisa lambat untuk corpus besar (3000+ dokumen × N tokens) —
LRU cache di-method `stem()` mempercepat repeated tokens.

Factory di-init sekali (heavy, ~100ms), instance shared antar dokumen.
"""

from __future__ import annotations

from functools import lru_cache

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


class SastrawiStemmer:
    """Wrapper Sastrawi `StemmerFactory().create_stemmer()` dengan LRU cache.

    Cache size 10k tokens cukup untuk corpus 3000 listings (typical
    vocabulary unique tokens ~5-8k setelah preprocessing).
    """

    def __init__(self, cache_size: int = 10_000):
        factory = StemmerFactory()
        self._stemmer = factory.create_stemmer()
        # Wrap stem method dengan LRU cache
        self.stem = lru_cache(maxsize=cache_size)(self._stem_uncached)

    def _stem_uncached(self, word: str) -> str:
        """Stem single word tanpa cache (called via cached wrapper)."""
        if not word:
            return word
        return self._stemmer.stem(word)

    def stem_tokens(self, tokens: list[str]) -> list[str]:
        """Stem list of tokens (gunakan cache per-token)."""
        return [self.stem(tok) for tok in tokens]

    def cache_info(self):
        """Debug: LRU cache hit/miss stats."""
        return self.stem.cache_info()  # type: ignore[attr-defined]

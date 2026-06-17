"""Indonesian stopword removal: Sastrawi default + custom domain stopwords.

Custom stopwords penting untuk domain kos — kata yang terlalu sering muncul
turun jadi noise (low IR signal). Tim Anggota B: setelah Anggota A scrape,
jalanin frequency analysis di `notebooks/01_eda.ipynb`:

    from collections import Counter
    all_tokens = [tok for listing in corpus for tok in listing['deskripsi'].split()]
    Counter(all_tokens).most_common(50)

Top-50 paling sering biasanya stopword domain — tambah ke
`DEFAULT_CUSTOM_STOPWORDS` kalau memang noise (bukan informative term).
"""

from __future__ import annotations

from typing import Optional

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


# Domain stopwords baseline (kos-specific). Tim Anggota B: extend dari EDA.
DEFAULT_CUSTOM_STOPWORDS: list[str] = [
    # Form kos
    "kos", "kost", "kosan", "kostan", "kos-kosan",
    # Kata generic listing
    "kamar",
    "sewa", "harga",
    "tersedia", "ready",
    "info", "informasi",
    "hubungi", "kontak",
    # Kata penghubung tambahan (Sastrawi sudah handle, tapi extra safety)
    # "untuk", "yang", "ke", "di"  # already di Sastrawi default
]


class StopwordRemover:
    """Wrapper Sastrawi StopWordRemover + custom set."""

    def __init__(
        self,
        custom: Optional[list[str]] = None,
        use_sastrawi_default: bool = True,
    ):
        if use_sastrawi_default:
            factory = StopWordRemoverFactory()
            self.sastrawi_stopwords: set[str] = set(factory.get_stop_words())
        else:
            self.sastrawi_stopwords = set()

        self.custom_stopwords: set[str] = set(custom or DEFAULT_CUSTOM_STOPWORDS)
        self.all_stopwords: set[str] = (
            self.sastrawi_stopwords | self.custom_stopwords
        )

    def remove(self, tokens: list[str]) -> list[str]:
        """Filter tokens — drop yang termasuk stopword."""
        return [tok for tok in tokens if tok.lower() not in self.all_stopwords]

    def is_stopword(self, word: str) -> bool:
        return word.lower() in self.all_stopwords

    def count(self) -> dict[str, int]:
        return {
            "sastrawi": len(self.sastrawi_stopwords),
            "custom": len(self.custom_stopwords),
            "total": len(self.all_stopwords),
        }

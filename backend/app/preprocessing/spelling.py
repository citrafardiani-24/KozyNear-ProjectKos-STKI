"""Domain-specific spelling correction untuk kos listings.

Approach: simple typo dictionary lookup (no statistical model untuk simplicity).
Tim Anggota B: tambah typo umum yang ditemukan saat scrape data.

Alternative future enhancement (Week 3 kalau ada waktu):
- Edit distance (Levenshtein) untuk catch typos baru
- Bigram language model untuk context-aware correction
- Library: pyspellchecker, atau custom n-gram model
"""

from __future__ import annotations

import re


# Common typos di kos listings (curated baseline)
TYPO_FIXES: dict[str, str] = {
    # Fasilitas typos
    "fasilitsa": "fasilitas",
    "fasiltas": "fasilitas",
    "fasilitas2": "fasilitas",
    "fasilitas-fasilitas": "fasilitas",
    # Lokasi typos
    "strategis": "strategis",  # canonical (kept untuk dokumentasi)
    "stratejis": "strategis",
    "lokasinyy": "lokasinya",
    "lokasinya2": "lokasinya",
    # Eksklusif variants (sering muncul di judul listing + query)
    "ekslusif": "eksklusif",
    "ekslusive": "eksklusif",
    "exclusive": "eksklusif",
    "exlusive": "eksklusif",
    # Kondisi
    "bagys": "bagus",
    "barangny": "barangnya",
    "rapih": "rapi",
    "luas2": "luas",
    "bersih2": "bersih",
    # Bahasa gaul yang sering muncul
    "pny": "punya",
    "pnya": "punya",
    "ada2": "ada",
}


# Filter out self-mappings (no-op) saat build pattern
_REAL_FIXES = {variant: fix for variant, fix in TYPO_FIXES.items() if variant != fix}


# Pre-compile patterns sekali (mahal kalau per-call)
_TYPO_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b" + re.escape(typo) + r"\b", re.IGNORECASE), fix)
    for typo, fix in _REAL_FIXES.items()
]


def correct_spelling(text: str) -> str:
    """Replace common typos dengan bentuk standar (word-boundary regex)."""
    if not text:
        return text
    for pattern, fix in _TYPO_PATTERNS:
        text = pattern.sub(fix, text)
    return text


if __name__ == "__main__":
    print(f"TYPO_FIXES: {len(TYPO_FIXES)} total, {len(_REAL_FIXES)} real fixes")

"""Scraper parsing helpers: price extraction, fasilitas normalization, text cleaning.

Fungsi-fungsi di sini standalone dan testable tanpa network. Lihat
tests/test_scraper.py untuk contoh penggunaan tiap helper.
"""

from __future__ import annotations

import re
from typing import Optional


# =============================================================================
# Harga extraction
# =============================================================================
# Pattern umum di Mamikos / OLX kos:
#   "Rp 500.000", "Rp500rb", "500k", "1jt", "1.5jt", "Rp 1.250.000"
#
# PENTING: jalankan extract_price() SEBELUM lowercase, karena `Rp` capitalized.

PRICE_PATTERNS = [
    # Rp 1.250.000 / Rp1.250.000 / Rp 850,000
    (re.compile(r"[Rr][Pp]\s*([\d.,]+)", re.IGNORECASE), "rupiah"),
    # 1.5jt, 1jt, 2,3 jt
    (re.compile(r"(\d+(?:[.,]\d+)?)\s*jt\b", re.IGNORECASE), "juta"),
    # 500k, 500rb, 500 ribu
    (re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:k|rb|ribu)\b", re.IGNORECASE), "ribu"),
]


def extract_price(text: str) -> Optional[int]:
    """Extract harga per bulan dalam IDR (integer).

    Returns None kalau gagal parse.
    """
    if not text:
        return None

    for pattern, unit in PRICE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        raw = match.group(1)
        try:
            if unit == "rupiah":
                # "1.250.000" → 1250000; "850,000" → 850000
                cleaned = raw.replace(".", "").replace(",", "")
                return int(cleaned)
            if unit == "juta":
                # "1,5" → 1.5 → 1500000
                normalized = raw.replace(",", ".")
                return int(float(normalized) * 1_000_000)
            if unit == "ribu":
                normalized = raw.replace(",", ".")
                return int(float(normalized) * 1_000)
        except ValueError:
            continue
    return None


# =============================================================================
# Tipe kos detection
# =============================================================================
TIPE_KEYWORDS = {
    # Mamikos hanya punya 3 tipe gender. "Pasutri" = kategori marketing
    # (campur yang boleh berdua), bukan tipe terpisah. Corpus real: 0 pasutri.
    "putra": ["putra", "pria", "cowo", "cowok", "laki"],
    "putri": ["putri", "wanita", "cewe", "cewek", "perempuan"],
    "campur": ["campur", "mix", "mixed", "umum"],
}


def detect_tipe(text: str) -> Optional[str]:
    """Deteksi tipe kos (putra/putri/campur) dari free text."""
    if not text:
        return None
    text_lower = text.lower()
    for tipe, keywords in TIPE_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                return tipe
    return None


# =============================================================================
# Fasilitas normalization
# =============================================================================
# Map varied raw forms → canonical vocabulary.
# Tim: tambah entry kalau ketemu varian baru saat scraping.

FASILITAS_ALIASES = {
    "ac": ["ac", "a/c", "air conditioning", "air conditioner"],
    "kipas angin": ["kipas angin", "kipas", "fan"],
    "wifi": ["wifi", "wi-fi", "internet"],
    "kamar mandi dalam": [
        "kamar mandi dalam", "km dlm", "km dalam",
        "wc dlm", "wc dalam", "kmd",
    ],
    "kamar mandi luar": ["kamar mandi luar", "km luar", "wc luar"],
    "kasur": ["kasur", "tempat tidur", "spring bed", "ranjang"],
    "lemari": ["lemari pakaian", "lemari"],
    "meja": ["meja belajar", "meja"],
    "dapur": ["dapur", "kitchen"],
    "parkir motor": ["parkir motor", "parkiran motor"],
    "parkir mobil": ["parkir mobil", "parkiran mobil"],
    "air panas": ["air panas", "water heater", "shower panas"],
    "tv": ["tv", "televisi"],
    "kulkas": ["kulkas", "lemari es", "fridge"],
    "mesin cuci": ["mesin cuci", "laundry"],
}


def normalize_fasilitas(raw_facilities: list[str]) -> list[str]:
    """Normalize raw fasilitas list ke vocabulary kanonis.

    Unknown items dikeep apa adanya (lowercase) untuk audit.
    """
    result: list[str] = []
    for item in raw_facilities:
        if not item:
            continue
        item_lower = item.lower().strip()
        if not item_lower:
            continue
        # Buang artefak parsing: digit murni ("0", "2") atau 1 karakter —
        # bukan nama fasilitas (ketemu 7 kasus di scrape real v2).
        if len(item_lower) < 2 or item_lower.isdigit():
            continue
        matched = False
        for canon, aliases in FASILITAS_ALIASES.items():
            if any(alias in item_lower for alias in aliases):
                if canon not in result:
                    result.append(canon)
                matched = True
                break
        if not matched and item_lower not in result:
            result.append(item_lower)
    return result


# =============================================================================
# Text cleaning
# =============================================================================
def clean_text(text: str) -> str:
    """Normalize whitespace, preserve paragraf breaks."""
    if not text:
        return ""
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def word_count(text: str) -> int:
    """Count whitespace-delimited words."""
    return len((text or "").split())


def truncate(text: str, max_chars: int = 200) -> str:
    """Truncate dengan suffix '...' kalau melebihi max_chars."""
    if not text or len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."

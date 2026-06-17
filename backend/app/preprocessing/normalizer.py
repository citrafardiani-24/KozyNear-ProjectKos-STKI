"""Atomic normalization: HTML strip, whitespace, lowercase, price extraction.

ORDER-SENSITIVE: `extract_prices_inline()` MUST jalan SEBELUM `lowercase()`
karena regex match `[Rr][Pp]` (case-insensitive), tapi `Rp` capitalized lebih
robust untuk catch edge case dimana harga tidak dipisah space.
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup


def strip_html(text: str) -> str:
    """Strip HTML tags + decode entities, leave plain text dengan space separator."""
    if not text:
        return ""
    soup = BeautifulSoup(text, "lxml")
    return soup.get_text(separator=" ", strip=True)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace (space/tab/newline) → single space, strip."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def lowercase(text: str) -> str:
    """Lowercase. Wajib jalan SETELAH `extract_prices_inline()`."""
    return (text or "").lower()


# =============================================================================
# Price extraction (inline, return all matches)
# =============================================================================
_PRICE_PATTERNS = [
    # 1.5jt, 1jt, 2,3 jt, 1,5 juta (kata "juta" penuh WAJIB ikut: query user
    # nyata menulis "max 1,5 juta"; dulu cuma `jt\b` sehingga lolos)
    (re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:jt|juta)\b", re.IGNORECASE), "juta"),
    # 500k, 500rb, 500 ribu
    (re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:k|rb|ribu)\b", re.IGNORECASE), "ribu"),
    # Rp 1.250.000 / Rp1.250.000 / Rp 850,000 — dievaluasi TERAKHIR supaya
    # "Rp 1,5 juta" sudah tertangkap pattern juta lebih dulu
    (re.compile(r"[Rr][Pp]\s*([\d.,]+)", re.IGNORECASE), "rupiah"),
]

# Nilai rupiah mentah di bawah ini dianggap noise fragmen (mis. "Rp 1,5" dari
# "Rp 1,5 juta" terbaca 15). Tidak ada harga kos < 10 ribu.
_MIN_PLAUSIBLE_RUPIAH = 10_000


def extract_prices_inline(text: str) -> list[int]:
    """Extract semua harga muncul di text. Return list IDR integer.

    Untuk kos listing biasanya 1-2 harga (sewa bulanan + uang muka), tapi
    return list untuk fleksibilitas.
    """
    if not text:
        return []

    prices: list[int] = []
    for pattern, unit in _PRICE_PATTERNS:
        for match in pattern.finditer(text):
            raw = match.group(1)
            try:
                if unit == "rupiah":
                    value = int(raw.replace(".", "").replace(",", ""))
                    if value < _MIN_PLAUSIBLE_RUPIAH:
                        continue  # fragmen noise, bukan harga
                elif unit == "juta":
                    value = int(float(raw.replace(",", ".")) * 1_000_000)
                elif unit == "ribu":
                    value = int(float(raw.replace(",", ".")) * 1_000)
                else:
                    continue
                if value not in prices:
                    prices.append(value)
            except ValueError:
                continue
    return prices

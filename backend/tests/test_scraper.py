"""Unit tests untuk scraper utilities.

Run: pytest backend/tests/test_scraper.py -v

Tests di sini standalone (no network) — bisa di-run sebelum tim Lead/Scraper
implementasi penuh.
"""

from __future__ import annotations

import pytest

from app.scraper.utils import (
    clean_text,
    detect_tipe,
    extract_price,
    normalize_fasilitas,
    truncate,
    word_count,
)


# =============================================================================
# extract_price
# =============================================================================
class TestExtractPrice:
    def test_rupiah_with_dots(self):
        assert extract_price("Sewa Rp 850.000 per bulan") == 850000

    def test_rupiah_no_space(self):
        assert extract_price("Rp1.250.000") == 1250000

    def test_rupiah_with_commas(self):
        assert extract_price("Rp 850,000") == 850000

    def test_juta_decimal_dot(self):
        assert extract_price("Harga 1.5jt") == 1500000

    def test_juta_decimal_comma(self):
        assert extract_price("Harga 1,5 jt") == 1500000

    def test_juta_integer(self):
        assert extract_price("Mulai 2jt aja") == 2000000

    def test_ribu_k(self):
        assert extract_price("Murah 350k aja") == 350000

    def test_ribu_rb(self):
        assert extract_price("500rb/bulan") == 500000

    def test_ribu_full_word(self):
        assert extract_price("Sewa 750 ribu per bulan") == 750000

    def test_priority_rupiah_first(self):
        # Kalau ada "Rp" eksplisit, prefer itu
        assert extract_price("Rp 850.000 atau 850k") == 850000

    def test_no_match(self):
        assert extract_price("Gratis berbagai fasilitas") is None

    def test_empty_string(self):
        assert extract_price("") is None

    def test_none_handling(self):
        # Defensive: function harus handle None tanpa crash
        assert extract_price(None) is None  # type: ignore[arg-type]


# =============================================================================
# detect_tipe
# =============================================================================
class TestDetectTipe:
    def test_putra_explicit(self):
        assert detect_tipe("Kos Putra Exclusive") == "putra"

    def test_putri_explicit(self):
        assert detect_tipe("Kos putri dekat unila") == "putri"

    def test_campur(self):
        assert detect_tipe("Kos campur murah") == "campur"

    def test_slang_cowo(self):
        assert detect_tipe("kos cowo dekat kampus") == "putra"

    def test_slang_cewe(self):
        assert detect_tipe("kos cewe murah") == "putri"

    def test_case_insensitive(self):
        assert detect_tipe("KOS PUTRA EKSKLUSIF") == "putra"

    def test_no_match(self):
        assert detect_tipe("kos murah strategic") is None

    def test_empty(self):
        assert detect_tipe("") is None


# =============================================================================
# normalize_fasilitas
# =============================================================================
class TestNormalizeFasilitas:
    def test_ac_variants(self):
        result = normalize_fasilitas(["AC", "a/c", "Air Conditioner"])
        assert result == ["ac"]  # dedup ke 1 canon

    def test_wifi_variants(self):
        result = normalize_fasilitas(["WiFi", "wi-fi", "Internet"])
        assert "wifi" in result

    def test_kamar_mandi_dalam_variants(self):
        result = normalize_fasilitas(["KM Dalam", "wc dlm", "KMD"])
        assert "kamar mandi dalam" in result

    def test_unknown_facility_kept(self):
        result = normalize_fasilitas(["WiFi", "sajadah", "musholla"])
        assert "wifi" in result
        assert "sajadah" in result
        assert "musholla" in result

    def test_empty_list(self):
        assert normalize_fasilitas([]) == []

    def test_empty_strings_filtered(self):
        result = normalize_fasilitas(["", "WiFi", "  "])
        assert result == ["wifi"]


# =============================================================================
# Text helpers
# =============================================================================
class TestWordCount:
    def test_basic(self):
        assert word_count("kos putra dekat unila") == 4

    def test_empty(self):
        assert word_count("") == 0

    def test_multiple_spaces(self):
        assert word_count("kos  putra   dekat") == 3

    def test_with_newlines(self):
        assert word_count("kos putra\ndekat unila") == 4


class TestCleanText:
    def test_collapse_whitespace_within_line(self):
        assert clean_text("kos    putra dekat") == "kos putra dekat"

    def test_preserve_paragraph_break(self):
        assert clean_text("kos putra\n\ndekat unila") == "kos putra\ndekat unila"

    def test_strip_empty_lines(self):
        assert clean_text("kos\n\n\nputra") == "kos\nputra"

    def test_empty(self):
        assert clean_text("") == ""

    def test_none(self):
        assert clean_text(None) == ""  # type: ignore[arg-type]


class TestTruncate:
    def test_no_truncate_under_limit(self):
        assert truncate("kos putra", 100) == "kos putra"

    def test_truncate_over_limit(self):
        text = "a" * 200
        result = truncate(text, 100)
        assert len(result) == 100
        assert result.endswith("...")

    def test_empty(self):
        assert truncate("", 100) == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

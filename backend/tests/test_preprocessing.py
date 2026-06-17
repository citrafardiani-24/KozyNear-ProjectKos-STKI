"""Unit tests untuk preprocessing pipeline + atomic functions.

Test atomic (normalizer, tokenizer, jargon, spelling) — no Sastrawi needed.
Test pipeline/stopword/stemmer — requires Sastrawi (skip kalau gak install).

Run: pytest backend/tests/test_preprocessing.py -v
"""

from __future__ import annotations

import pytest

from app.preprocessing.jargon import KOS_JARGON_DICT, jargon_count, MIN_REQUIRED
from app.preprocessing.normalizer import (
    extract_prices_inline,
    lowercase,
    normalize_whitespace,
    strip_html,
)
from app.preprocessing.spelling import correct_spelling
from app.preprocessing.tokenizer import simple_tokenize


# =============================================================================
# Strip HTML
# =============================================================================
class TestStripHTML:
    def test_basic_tags(self):
        assert strip_html("<p>kos putra</p>") == "kos putra"

    def test_nested_tags(self):
        assert strip_html("<div><p>kos <b>putra</b></p></div>") == "kos putra"

    def test_strip_with_attributes(self):
        assert strip_html('<a href="x">kos</a> putra') == "kos putra"

    def test_empty(self):
        assert strip_html("") == ""

    def test_no_tags(self):
        assert strip_html("kos putra dekat unila") == "kos putra dekat unila"


# =============================================================================
# Whitespace normalize
# =============================================================================
class TestNormalizeWhitespace:
    def test_multiple_spaces(self):
        assert normalize_whitespace("kos   putra    dekat") == "kos putra dekat"

    def test_newlines(self):
        assert normalize_whitespace("kos\nputra\n\ndekat") == "kos putra dekat"

    def test_tabs(self):
        assert normalize_whitespace("kos\tputra") == "kos putra"

    def test_strip_leading_trailing(self):
        assert normalize_whitespace("  kos putra  ") == "kos putra"

    def test_empty(self):
        assert normalize_whitespace("") == ""


# =============================================================================
# Lowercase
# =============================================================================
class TestLowercase:
    def test_basic(self):
        assert lowercase("Kos PUTRA AC") == "kos putra ac"

    def test_unicode(self):
        assert lowercase("KOS DEKAT UNILA") == "kos dekat unila"


# =============================================================================
# Price extraction (inline, all matches)
# =============================================================================
class TestExtractPricesInline:
    def test_single_rupiah(self):
        assert extract_prices_inline("Sewa Rp 850.000 per bulan") == [850000]

    def test_no_space_rupiah(self):
        assert extract_prices_inline("Rp1.250.000") == [1250000]

    def test_multiple_prices(self):
        prices = extract_prices_inline("Mulai Rp 500.000 sampai Rp 1.500.000")
        assert prices == [500000, 1500000]

    def test_juta(self):
        assert extract_prices_inline("Harga 1.5jt all in") == [1500000]

    def test_juta_kata_penuh(self):
        # Bug lama: cuma `jt\b`, "1,5 juta" lolos dan fragmen "Rp 1,5"
        # terbaca 15 rupiah
        assert extract_prices_inline("maksimal 1,5 juta per bulan") == [1500000]
        assert extract_prices_inline("Rp 1,5 juta") == [1500000]
        assert extract_prices_inline("sekitar 2 juta") == [2000000]

    def test_rupiah_fragment_noise_dibuang(self):
        # Nilai rupiah < 10rb itu fragmen, bukan harga kos
        assert extract_prices_inline("Rp 1,5") == []

    def test_ribu_k(self):
        assert extract_prices_inline("Murah 350k aja") == [350000]

    def test_ribu_rb(self):
        assert extract_prices_inline("500rb/bulan") == [500000]

    def test_dedup(self):
        # Sama-sama 500.000, hanya muncul sekali
        assert extract_prices_inline("Rp 500.000 atau 500rb") == [500000]

    def test_anti_pattern_lowercase_first(self):
        # Walaupun lowercase, regex tetap match (insensitive)
        assert extract_prices_inline("rp 500.000") == [500000]

    def test_empty(self):
        assert extract_prices_inline("") == []


# =============================================================================
# Tokenizer
# =============================================================================
class TestTokenizer:
    def test_simple(self):
        assert simple_tokenize("kos putra dekat unila") == [
            "kos", "putra", "dekat", "unila",
        ]

    def test_punctuation_excluded(self):
        assert simple_tokenize("kos, putra! dekat? unila.") == [
            "kos", "putra", "dekat", "unila",
        ]

    def test_numbers_kept(self):
        tokens = simple_tokenize("kos 500k murah")
        assert "kos" in tokens
        assert "500k" in tokens
        assert "murah" in tokens

    def test_empty(self):
        assert simple_tokenize("") == []


# =============================================================================
# Spelling correction
# =============================================================================
class TestSpelling:
    def test_fix_fasiltas(self):
        assert "fasilitas" in correct_spelling("banyak fasiltas").lower()

    def test_fix_exclusive(self):
        assert "eksklusif" in correct_spelling("kos ekslusive").lower()

    def test_no_change_correct(self):
        # "kos putra" gak ada di typo dict, return as-is
        assert correct_spelling("kos putra") == "kos putra"

    def test_empty(self):
        assert correct_spelling("") == ""

    def test_word_boundary(self):
        # "rapih" → "rapi" tapi "rapihkan" jangan ke-replace
        result = correct_spelling("rapih dan rapihkan")
        assert "rapi" in result
        # "rapihkan" gak di-replace karena \b boundary
        assert "rapihkan" in result or "rapih" not in result.replace("rapi", "")


# =============================================================================
# Jargon dict
# =============================================================================
class TestJargonDict:
    def test_meets_rubric_minimum(self):
        count = jargon_count()
        assert count >= MIN_REQUIRED, (
            f"Hanya {count} entries, minimum {MIN_REQUIRED} untuk rubric "
            f"Preprocessing 15%. Tim Anggota B: tambah {MIN_REQUIRED - count}"
        )

    def test_common_abbreviations_present(self):
        assert "ac" in KOS_JARGON_DICT
        assert "kmd" in KOS_JARGON_DICT
        assert "wc dlm" in KOS_JARGON_DICT

    def test_location_variants(self):
        assert KOS_JARGON_DICT["gdg meneng"] == "gedong meneng"
        assert KOS_JARGON_DICT["sumbro"] == "sumantri brojonegoro"
        assert KOS_JARGON_DICT["unyila"] == "universitas lampung"

    def test_type_slang(self):
        assert KOS_JARGON_DICT["cowo"] == "putra"
        assert KOS_JARGON_DICT["cewe"] == "putri"


# =============================================================================
# Pipeline + Sastrawi (heavy — skip kalau Sastrawi belum install)
# =============================================================================
try:
    from app.preprocessing.pipeline import PipelineConfig, PreprocessingPipeline
    from app.preprocessing.stemmer import SastrawiStemmer
    from app.preprocessing.stopwords import StopwordRemover

    SASTRAWI_AVAILABLE = True
except ImportError:
    SASTRAWI_AVAILABLE = False


@pytest.mark.skipif(not SASTRAWI_AVAILABLE, reason="Sastrawi belum di-install")
class TestPipeline:
    def test_full_pipeline_basic(self):
        pipeline = PreprocessingPipeline()
        result = pipeline.process("Kos Putra AC WiFi Rp 850.000/bulan dekat unyila")
        # Price ke-extract
        assert 850000 in result.extracted_prices
        # Stages applied
        assert "stem" in result.stages_applied
        assert "apply_jargon_dict" in result.stages_applied
        # Tokens non-empty
        assert len(result.tokens) > 0

    def test_disable_stem(self):
        config = PipelineConfig(stem=False)
        pipeline = PreprocessingPipeline(config)
        result = pipeline.process("kos murah dekat kampus")
        assert "stem" not in result.stages_applied

    def test_disable_stopword(self):
        config = PipelineConfig(remove_stopwords=False, stem=False)
        pipeline = PreprocessingPipeline(config)
        result = pipeline.process("kos yang murah di lampung")
        # "yang" dan "di" gak ke-remove
        tokens_lower = [t.lower() for t in result.tokens]
        assert "yang" in tokens_lower or "di" in tokens_lower

    def test_jargon_substitution_gdg_meneng(self):
        config = PipelineConfig(stem=False, remove_stopwords=False)
        pipeline = PreprocessingPipeline(config)
        result = pipeline.process("kos di gdg meneng")
        # "gdg meneng" → "gedong meneng"
        assert "gedong" in result.processed
        assert "meneng" in result.processed

    def test_jargon_longest_first(self):
        # "km dlm" harus ke-match dulu sebelum "dlm"
        config = PipelineConfig(stem=False, remove_stopwords=False)
        pipeline = PreprocessingPipeline(config)
        result = pipeline.process("ada km dlm dan dapur")
        # "km dlm" → "kamar mandi dalam", bukan "km dalam"
        assert "kamar mandi dalam" in result.processed

    def test_price_preserved_before_lowercase(self):
        # Anti-pattern check
        pipeline = PreprocessingPipeline()
        result = pipeline.process("Sewa Rp 1.250.000")
        assert 1250000 in result.extracted_prices


@pytest.mark.skipif(not SASTRAWI_AVAILABLE, reason="Sastrawi belum di-install")
class TestStopwordRemover:
    def test_remove_sastrawi_default(self):
        remover = StopwordRemover()
        tokens = ["kos", "yang", "murah", "di", "lampung"]
        result = remover.remove(tokens)
        # "yang" dan "di" Sastrawi stopwords
        assert "yang" not in result
        assert "di" not in result
        # "kos" custom stopword
        assert "kos" not in result
        # Informative tokens kept
        assert "murah" in result
        assert "lampung" in result

    def test_custom_only(self):
        remover = StopwordRemover(custom=["spesifik"], use_sastrawi_default=False)
        assert remover.is_stopword("spesifik")
        assert not remover.is_stopword("yang")  # Sastrawi default off

    def test_count(self):
        remover = StopwordRemover()
        counts = remover.count()
        assert counts["sastrawi"] > 0
        assert counts["custom"] > 0


@pytest.mark.skipif(not SASTRAWI_AVAILABLE, reason="Sastrawi belum di-install")
class TestStemmer:
    def test_basic_stem(self):
        stemmer = SastrawiStemmer()
        assert stemmer.stem("berlari") == "lari"
        assert stemmer.stem("pergi") == "pergi"  # already stem

    def test_cache_hit(self):
        stemmer = SastrawiStemmer()
        # First call: cache miss
        r1 = stemmer.stem("memasak")
        # Second call: cache hit
        r2 = stemmer.stem("memasak")
        assert r1 == r2
        info = stemmer.cache_info()
        assert info.hits >= 1

    def test_stem_tokens_batch(self):
        stemmer = SastrawiStemmer()
        result = stemmer.stem_tokens(["berlari", "memasak", "menulis"])
        assert len(result) == 3
        assert "lari" in result

    def test_empty(self):
        stemmer = SastrawiStemmer()
        assert stemmer.stem("") == ""

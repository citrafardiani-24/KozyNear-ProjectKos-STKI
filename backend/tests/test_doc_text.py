"""Test komposisi fielded doc text."""
from app.preprocessing.doc_text import compose_lexical_text, compose_natural_text

LISTING = {
    "judul": "Kost Putri Jocelyn Rajabasa",
    "kecamatan": "Rajabasa",
    "fasilitas": ["wifi", "k. mandi dalam"],
    "deskripsi": "dekat kampus",
}


def test_lexical_repeats_judul_twice():
    text = compose_lexical_text(LISTING)
    assert text.count("Kost Putri Jocelyn Rajabasa") == 2
    assert "Rajabasa" in text and "wifi" in text and "dekat kampus" in text


def test_natural_no_repetition():
    text = compose_natural_text(LISTING)
    assert text.count("Kost Putri Jocelyn Rajabasa") == 1
    assert text.endswith("dekat kampus")


def test_empty_fields_skipped():
    text = compose_lexical_text({"judul": "Kos A", "deskripsi": ""})
    assert "Kos A" in text
    # tidak ada separator menggantung untuk field kosong
    assert not text.endswith(".") and ". ." not in text.replace(" . ", "|")


def test_all_empty_safe():
    assert compose_lexical_text({}) == ""
    assert compose_natural_text({}) == ""

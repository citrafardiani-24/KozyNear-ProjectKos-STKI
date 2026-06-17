"""Komposisi teks dokumen untuk indexing (fielded, ala BM25F ringan).

Masalah yang diobati: sebelumnya HANYA deskripsi yang diindex, padahal 21%
deskripsi corpus < 10 kata dan field terkuat (judul "Kost Putri Jocelyn
Rajabasa", kecamatan, fasilitas) justru tidak terlihat model mana pun.

Pendekatan: gabungkan field jadi satu teks dengan pembobotan via REPETISI
(field concatenation, varian praktis dari BM25F): judul diulang 2x (membawa
nama + tipe + area), kecamatan + fasilitas + deskripsi 1x. Repetisi menaikkan
term frequency field penting tanpa mengubah engine BM25/TF-IDF.

Dua varian:
- compose_lexical_text: input pipeline preprocessing -> BM25/TF-IDF.
- compose_natural_text: raw_text untuk neural encoder (tanpa repetisi,
  dipisah titik; encoder kalimat sensitif terhadap redundansi).

WAJIB dipakai semua jalur build (scripts/preprocess_corpus.py +
scripts/seed_db.py --preprocess) supaya index, eval, dan serving melihat
representasi dokumen yang sama.
"""

from __future__ import annotations

from typing import Any

JUDUL_WEIGHT = 2


def _fields(listing: dict[str, Any]) -> tuple[str, str, str, str]:
    judul = (listing.get("judul") or "").strip()
    kecamatan = (listing.get("kecamatan") or "").strip()
    fasilitas = " ".join(str(f) for f in (listing.get("fasilitas") or []))
    deskripsi = (listing.get("deskripsi") or "").strip()
    return judul, kecamatan, fasilitas, deskripsi


def compose_lexical_text(listing: dict[str, Any]) -> str:
    """Teks untuk pipeline preprocessing -> index lexical (BM25/TF-IDF)."""
    judul, kecamatan, fasilitas, deskripsi = _fields(listing)
    parts = [judul] * JUDUL_WEIGHT + [kecamatan, fasilitas, deskripsi]
    return " . ".join(p for p in parts if p)


def compose_natural_text(listing: dict[str, Any]) -> str:
    """Teks natural untuk neural encoder (raw_text di corpus.json)."""
    judul, kecamatan, fasilitas, deskripsi = _fields(listing)
    return ". ".join(p for p in (judul, kecamatan, fasilitas, deskripsi) if p)

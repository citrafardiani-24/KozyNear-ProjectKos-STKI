"""Rule-based query understanding untuk pencarian kos.

Ekstrak gender / harga / fasilitas / anchor-lokasi dari query natural language,
reuse kamus jargon + price extractor yang sudah ada. Output keyword fasilitas
sengaja pendek (mis. "ac", "wifi", "mandi dalam") supaya match nilai
`Listing.fasilitas` yang lowercase-pendek.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.preprocessing.jargon import TYPE_SLANG
from app.preprocessing.normalizer import extract_prices_inline
from app.search.gazetteer import Anchor, Gazetteer

MURAH_DEFAULT_MAX = 1_000_000  # heuristik "murah" tanpa angka (tunable)

# gender bahasa Inggris (query code-switch ID-EN). SENGAJA hanya di parser,
# TIDAK ditambah ke KOS_JARGON_DICT: ablation menunjukkan substitusi jargon
# di sisi DOKUMEN merugikan (deskripsi kos berbahasa Indonesia, tak butuh ini).
_GENDER_EN = {
    "girls": "putri", "girl": "putri", "women": "putri", "woman": "putri",
    "female": "putri", "ladies": "putri",
    "boys": "putra", "boy": "putra", "men": "putra", "male": "putra",
    "gentlemen": "putra",
    "mixed": "campur", "coed": "campur", "unisex": "campur",
}

# gender: kanonik + slang ID (cowo->putra, cewe->putri) + EN code-switch
_GENDER_WORDS = {
    "putra": "putra", "putri": "putri", "campur": "campur",
    **TYPE_SLANG, **_GENDER_EN,
}

# fasilitas: variasi query -> keyword pendek yang substring-match nilai listing
_FACILITY_ALIASES = {
    "air conditioner": "ac", "ac": "ac",
    "wifi": "wifi", "wi-fi": "wifi", "wf": "wifi",
    "televisi": "tv", "tv": "tv",
    "kamar mandi dalam": "mandi dalam", "km dalam": "mandi dalam",
    "kmd": "mandi dalam", "wc dalam": "mandi dalam",
    "parkir mobil": "parkir mobil", "parkir motor": "parkir motor",
    "dapur": "dapur", "kasur": "kasur",
}
_FACILITY_KEYS = sorted(_FACILITY_ALIASES, key=len, reverse=True)  # longest-first


@dataclass
class ParsedQuery:
    gender: str | None = None
    harga_min: int | None = None
    harga_max: int | None = None
    kecamatan: str | None = None
    fasilitas: list[str] = field(default_factory=list)
    anchor: Anchor | None = None
    residual_text: str = ""
    understood: dict = field(default_factory=dict)
    # Constraint yang datang dari filter UI eksplisit (bukan hasil parse).
    # Hard filter TIDAK PERNAH melonggarkan constraint eksplisit; relaxation
    # hanya berlaku untuk constraint hasil inferensi dari teks query.
    explicit: set[str] = field(default_factory=set)

    def build_understood(self) -> dict:
        self.understood = {
            "gender": self.gender,
            "harga_min": self.harga_min,
            "harga_max": self.harga_max,
            "fasilitas": self.fasilitas,
            "anchor": self.anchor.name if self.anchor else None,
            "kecamatan": self.kecamatan,
        }
        return self.understood


def parse(q: str, gazetteer: Gazetteer) -> ParsedQuery:
    low = f" {q.lower()} "
    p = ParsedQuery()

    # gender (bentrok -> drop)
    genders = {canon for word, canon in _GENDER_WORDS.items()
               if re.search(rf"\b{re.escape(word)}\b", low)}
    p.gender = next(iter(genders)) if len(genders) == 1 else None

    # fasilitas (longest alias first, dedup canonical)
    for alias in _FACILITY_KEYS:
        if re.search(rf"\b{re.escape(alias)}\b", low):
            kw = _FACILITY_ALIASES[alias]
            if kw not in p.fasilitas:
                p.fasilitas.append(kw)

    # harga: angka eksplisit, else heuristik "murah"
    prices = extract_prices_inline(low)
    if prices:
        p.harga_max = max(prices)
    elif re.search(r"\bmurah\b", low):
        p.harga_max = MURAH_DEFAULT_MAX

    # anchor lokasi (gazetteer alias longest-match)
    p.anchor = gazetteer.lookup(low)

    # residual_text untuk BM25: buang token yang sudah dikenali
    residual = low
    for word in list(_GENDER_WORDS) + _FACILITY_KEYS:
        residual = re.sub(rf"\b{re.escape(word)}\b", " ", residual)
    residual = re.sub(r"\b(murah|maksimal|max|rp|deket|dekat)\b", " ", residual)
    residual = re.sub(r"\d+[.,]?\d*\s*(jt|juta|rb|ribu|k)?\b", " ", residual)
    p.residual_text = re.sub(r"\s+", " ", residual).strip()

    p.build_understood()
    return p

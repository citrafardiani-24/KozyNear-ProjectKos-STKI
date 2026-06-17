"""Fusion ranking: gabung skor teks (BM25) + geo + atribut, + hard filter.

Tiap komponen di-min-max normalize per query supaya skalanya sebanding.
Hard filter (gender + harga eksplisit) dengan fallback longgarkan kalau kosong.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.search.gazetteer import haversine_km
from app.search.query_parser import ParsedQuery


@dataclass
class Candidate:
    doc_id: str
    text_score: float
    tipe: str | None
    harga: int | None
    fasilitas: list[str] | None
    lat: float | None
    lng: float | None
    kecamatan: str | None = None


def _minmax(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.0] * len(values)  # semua sama -> netral
    return [(v - lo) / (hi - lo) for v in values]


def _geo_raw(c: Candidate, p: ParsedQuery) -> float:
    if p.anchor is None or c.lat is None or c.lng is None:
        return 0.0
    return 1.0 / (1.0 + haversine_km(float(c.lat), float(c.lng),
                                     p.anchor.lat, p.anchor.lng))


def _attr_raw(c: Candidate, p: ParsedQuery) -> float:
    if not p.fasilitas:
        return 0.0
    have = [f.lower() for f in (c.fasilitas or [])]
    hits = sum(1 for kw in p.fasilitas if any(kw in f for f in have))
    return hits / len(p.fasilitas)


def fuse(cands: list[Candidate], p: ParsedQuery,
         weights: tuple[float, float, float]) -> list[tuple[str, float]]:
    if not cands:
        return []
    w_text, w_geo, w_attr = weights
    text = _minmax([c.text_score for c in cands])
    geo = _minmax([_geo_raw(c, p) for c in cands])
    attr = _minmax([_attr_raw(c, p) for c in cands])
    scored = [(c.doc_id, w_text * text[i] + w_geo * geo[i] + w_attr * attr[i])
              for i, c in enumerate(cands)]
    return sorted(scored, key=lambda x: x[1], reverse=True)


def apply_hard_filter(cands: list[Candidate], p: ParsedQuery
                      ) -> tuple[list[Candidate], list[str]]:
    """Filter gender + harga + kecamatan.

    Dua kelas constraint:
    - Eksplisit (dari filter UI, ada di `p.explicit`): SELALU enforced,
      tidak pernah dilonggarkan — user menyatakannya langsung.
    - Parsed (inferensi dari teks query): kalau hasil kosong, dilonggarkan
      bertahap (harga dulu, lalu gender) dan dilaporkan via `relaxed`.
    """
    def keep(c: Candidate, use_parsed_gender: bool, use_parsed_harga: bool) -> bool:
        # --- Eksplisit: selalu enforced ---
        if "kecamatan" in p.explicit and p.kecamatan:
            if not c.kecamatan or p.kecamatan.lower() not in c.kecamatan.lower():
                return False
        if "harga_min" in p.explicit and p.harga_min and c.harga and c.harga < p.harga_min:
            return False
        if "harga_max" in p.explicit and p.harga_max and c.harga and c.harga > p.harga_max:
            return False
        if "gender" in p.explicit and p.gender and c.tipe and c.tipe != p.gender:
            return False
        # --- Parsed: relaxable ---
        if (use_parsed_gender and "gender" not in p.explicit
                and p.gender and c.tipe and c.tipe != p.gender):
            return False
        if (use_parsed_harga and "harga_max" not in p.explicit
                and p.harga_max and c.harga and c.harga > p.harga_max):
            return False
        return True

    has_parsed_harga = bool(p.harga_max) and "harga_max" not in p.explicit
    has_parsed_gender = bool(p.gender) and "gender" not in p.explicit

    for use_parsed_gender, use_parsed_harga, dropped in [
        (True, True, []),
        (True, False, ["harga"]),
        (False, False, ["harga", "gender"]),
    ]:
        kept = [c for c in cands if keep(c, use_parsed_gender, use_parsed_harga)]
        if kept:
            relaxed = [
                label for label in dropped
                if (label == "harga" and has_parsed_harga)
                or (label == "gender" and has_parsed_gender)
            ]
            return kept, relaxed
    return [], [label for label, present in
                [("harga", has_parsed_harga), ("gender", has_parsed_gender)] if present]

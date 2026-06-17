"""Gazetteer anchor (kampus + landmark) untuk geo ranking.

Statis, dibangun sekali (lihat scripts/build_gazetteer.py), tanpa API saat runtime.
Koordinat kampus diangkat dari scripts/enrich_geo.py.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

# backend/app/search/gazetteer.py -> parents[3] = repo root -> data/gazetteer.json
_DATA = Path(__file__).resolve().parents[3] / "data" / "gazetteer.json"


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Jarak great-circle (km) antar dua koordinat."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


@dataclass(frozen=True)
class Anchor:
    name: str
    lat: float
    lng: float


class Gazetteer:
    """Lookup anchor dari teks query via nama kanonik + alias (longest-match)."""

    # Alias single-token >= panjang ini jadi kandidat fuzzy. Alias pendek
    # (uin/ubl/mbk) sengaja TIDAK di-fuzzy: rawan false-match ke kata pendek.
    _FUZZY_MIN_ALIAS_LEN = 5
    # Token query >= panjang ini boleh di-fuzzy (typo "itra" len 4 harus lolos).
    _FUZZY_MIN_TOKEN_LEN = 4
    _FUZZY_CUTOFF = 0.84

    def __init__(self, entries: list[dict]):
        pairs: list[tuple[str, Anchor]] = []
        for e in entries:
            anchor = Anchor(e["name"], float(e["lat"]), float(e["lng"]))
            for key in [e["name"], *e.get("aliases", [])]:
                pairs.append((key.lower(), anchor))
        # match alias terpanjang dulu biar yang spesifik menang
        self._pairs = sorted(pairs, key=lambda p: len(p[0]), reverse=True)
        # Alias single-token cukup panjang -> kandidat fuzzy fallback
        self._fuzzy_keys = [
            (k, a) for k, a in self._pairs
            if " " not in k and len(k) >= self._FUZZY_MIN_ALIAS_LEN
        ]

    @classmethod
    def load(cls, path: Path = _DATA) -> "Gazetteer":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def lookup(self, text: str, fuzzy: bool = True) -> Anchor | None:
        """Exact substring match dulu (cepat, nol risiko). Kalau gagal dan
        fuzzy=True, coba fuzzy-match token panjang ke alias single-token
        (menangani typo nama kampus, mis. 'unilla' -> 'unila')."""
        low = f" {text.lower()} "
        for key, anchor in self._pairs:
            if f" {key} " in low:
                return anchor
        if fuzzy:
            return self._fuzzy_lookup(low)
        return None

    def _fuzzy_lookup(self, low: str) -> Anchor | None:
        best: Anchor | None = None
        best_score = self._FUZZY_CUTOFF
        for tok in re.findall(r"[a-z]+", low):
            if len(tok) < self._FUZZY_MIN_TOKEN_LEN:
                continue
            for key, anchor in self._fuzzy_keys:
                score = SequenceMatcher(None, tok, key).ratio()
                if score > best_score:
                    best_score, best = score, anchor
        return best

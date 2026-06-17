"""Build data/gazetteer.json dari tabel anchor terverifikasi.

Koordinat diverifikasi manual 2026-06-10 via Nominatim (OpenStreetMap) dan
infobox Wikipedia ID. Sebelumnya 4 kampus (Teknokrat/Darmajaya/UBL/Malahayati)
memakai placeholder lat -5.40xx (pusat kota Tanjung Karang) yang meleset
3-9 km; dengan w_geo=0.4 di fusion ranker itu merusak ranking "dekat X".

Jalankan ulang kalau mau menambah anchor:
    python -m scripts.build_gazetteer            # tulis dari VERIFIED_ANCHORS
    python -m scripts.build_gazetteer --check    # cek konsistensi file vs tabel

Untuk anchor baru: cari koordinat di Nominatim
(https://nominatim.openstreetmap.org/search?q=...&format=jsonv2) atau infobox
Wikipedia, tambah ke VERIFIED_ANCHORS dengan field source, lalu jalankan script.
JANGAN menebak koordinat; lebih baik anchor tidak ada daripada salah tempat.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = REPO_ROOT / "data" / "gazetteer.json"

# (name, lat, lng, aliases, source)
VERIFIED_ANCHORS: list[dict] = [
    {"name": "universitas lampung", "lat": -5.3645, "lng": 105.2434,
     "aliases": ["unila", "unyila", "unl", "fmipa", "ft unila", "fkip", "feb unila"],
     "source": "wikipedia-id 2026-06-10"},
    {"name": "itera", "lat": -5.3668, "lng": 105.3149,
     "aliases": ["institut teknologi sumatera"],
     "source": "nominatim-osm 2026-06-10"},
    {"name": "uin raden intan", "lat": -5.3808, "lng": 105.3038,
     "aliases": ["uin", "raden intan"],
     "source": "nominatim-osm 2026-06-10"},
    {"name": "politeknik negeri lampung", "lat": -5.3584, "lng": 105.2329,
     "aliases": ["polinela"],
     "source": "wikipedia-id 2026-06-10"},
    {"name": "universitas teknokrat indonesia", "lat": -5.3824, "lng": 105.2578,
     "aliases": ["teknokrat"],
     "source": "wikipedia-id 2026-06-10"},
    {"name": "ibi darmajaya", "lat": -5.3772, "lng": 105.2498,
     "aliases": ["darmajaya"],
     "source": "nominatim-osm 2026-06-10"},
    {"name": "universitas bandar lampung", "lat": -5.3796, "lng": 105.2518,
     "aliases": ["ubl"],
     "source": "nominatim-osm 2026-06-10"},
    {"name": "universitas malahayati", "lat": -5.3815, "lng": 105.2187,
     "aliases": ["malahayati", "unmal"],
     "source": "wikipedia-id 2026-06-10"},
    {"name": "mall boemi kedaton", "lat": -5.3821, "lng": 105.2596,
     "aliases": ["mbk", "mall bumi kedaton", "mal boemi kedaton"],
     "source": "nominatim-osm 2026-06-10"},
    {"name": "transmart lampung", "lat": -5.3830, "lng": 105.2819,
     "aliases": ["transmart", "transmart way halim"],
     "source": "nominatim-osm 2026-06-10"},
    {"name": "stasiun tanjungkarang", "lat": -5.4050, "lng": 105.2572,
     "aliases": ["stasiun tanjung karang", "stasiun"],
     "source": "wikipedia-id 2026-06-10"},
]

# Bounding box kasar Bandar Lampung raya (sanity guard anti-typo)
LAT_RANGE = (-5.55, -5.20)
LNG_RANGE = (105.10, 105.40)


def validate(entries: list[dict]) -> list[str]:
    errors: list[str] = []
    seen_keys: set[str] = set()
    for e in entries:
        for field in ("name", "lat", "lng", "aliases", "source"):
            if field not in e:
                errors.append(f"{e.get('name', '?')}: field '{field}' hilang")
        if not (LAT_RANGE[0] <= e["lat"] <= LAT_RANGE[1]):
            errors.append(f"{e['name']}: lat {e['lat']} di luar Bandar Lampung")
        if not (LNG_RANGE[0] <= e["lng"] <= LNG_RANGE[1]):
            errors.append(f"{e['name']}: lng {e['lng']} di luar Bandar Lampung")
        for key in [e["name"], *e["aliases"]]:
            if key in seen_keys:
                errors.append(f"alias duplikat: '{key}'")
            seen_keys.add(key)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Build/check gazetteer.json")
    parser.add_argument("--check", action="store_true",
                        help="cek file existing == tabel (exit 1 kalau beda)")
    args = parser.parse_args()

    errors = validate(VERIFIED_ANCHORS)
    if errors:
        for err in errors:
            print(f"[error] {err}")
        return 1

    if args.check:
        current = json.loads(OUTPUT.read_text(encoding="utf-8"))
        if current != VERIFIED_ANCHORS:
            print(f"[check] {OUTPUT} BEDA dari VERIFIED_ANCHORS — jalankan ulang tanpa --check")
            return 1
        print(f"[check] {OUTPUT} konsisten ({len(current)} anchors)")
        return 0

    OUTPUT.write_text(
        json.dumps(VERIFIED_ANCHORS, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[build] {OUTPUT} ditulis: {len(VERIFIED_ANCHORS)} anchors")
    return 0


if __name__ == "__main__":
    sys.exit(main())

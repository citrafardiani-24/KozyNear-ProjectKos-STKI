"""Enrich listings dengan geocoding (Nominatim OSM, free no API key).

Untuk tiap listing dengan alamat tapi tanpa koordinat:
1. Geocode via Nominatim (OpenStreetMap, free, ~1 request/sec rate limit)
2. Compute jarak ke 9 universitas Bandar Lampung via Haversine
3. Update field koordinat, jarak_kampus_km

Alternative: Google Maps Geocoding API (200 free/day, butuh API key).
Set GOOGLE_MAPS_API_KEY env var untuk pakai itu (lebih akurat di Indonesia).

Usage:
    cd backend
    python -m scripts.enrich_geo \\
        --input ../data/raw/mamikos_real_v2.jsonl \\
        --output ../data/raw/kozynear_enriched.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import requests


# 9 universitas Bandar Lampung (sama dengan generator)
UNIVERSITIES = [
    ("UNILA",                          -5.3692, 105.2433),
    ("Politeknik Negeri Lampung",      -5.3650, 105.2400),
    ("IBI Darmajaya",                  -5.4017, 105.2895),
    ("Universitas Bandar Lampung",     -5.4017, 105.2900),
    ("UIN Raden Intan Lampung",        -5.3877, 105.3050),
    ("Universitas Teknokrat Indonesia",-5.4017, 105.2783),
    ("Universitas Malahayati",         -5.4060, 105.2929),
    ("ITERA",                          -5.3577, 105.3145),
    ("Universitas Saburai",            -5.4100, 105.3200),
]


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def nominatim_geocode(query: str) -> Optional[tuple[float, float]]:
    """Geocode via Nominatim. Polite: 1 req/sec rate limit per ToS."""
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": query,
        "format": "json",
        "limit": 1,
        "countrycodes": "id",
    }
    headers = {
        "User-Agent": "KozyNear/0.2 (STKI Final Project UNILA; contact: dymaz.satya2005@gmail.com)",
        "Accept": "application/json",
        "Accept-Language": "id,en",
    }
    try:
        r = requests.get(url, params=params, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception as e:
        print(f"[geocode fail] {query[:60]}: {e}")
    return None


def google_geocode(query: str, api_key: str) -> Optional[tuple[float, float]]:
    """Geocode via Google Maps Geocoding API (kalau ada API key)."""
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": query, "key": api_key, "region": "id"}
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        if data.get("status") == "OK" and data.get("results"):
            loc = data["results"][0]["geometry"]["location"]
            return loc["lat"], loc["lng"]
    except Exception as e:
        print(f"[google geocode fail] {query[:60]}: {e}")
    return None


def find_nearest_university(lat: float, lng: float) -> tuple[str, float]:
    """Return (name, distance_km) of nearest university."""
    min_dist = float("inf")
    nearest = ""
    for name, ulat, ulng in UNIVERSITIES:
        d = haversine_km(lat, lng, ulat, ulng)
        if d < min_dist:
            min_dist = d
            nearest = name
    return nearest, round(min_dist, 2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--use-google", action="store_true",
                        help="Pakai Google Maps API (butuh GOOGLE_MAPS_API_KEY env)")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip listings yang sudah punya koordinat")
    args = parser.parse_args()

    google_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if args.use_google and not google_key:
        print("ERROR: GOOGLE_MAPS_API_KEY env var required for --use-google")
        return 1

    # Load
    listings: list[dict] = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                listings.append(json.loads(line))
    print(f"[load] {len(listings)} listings")

    # Stats
    has_coords = sum(1 for l in listings if l.get("koordinat"))
    needs_geocode = len(listings) - has_coords
    print(f"[stats] {has_coords} already geocoded, {needs_geocode} need geocoding")

    # Enrich
    enriched: list[dict] = []
    geocode_count = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as out_f:
        for i, listing in enumerate(listings, start=1):
            # Skip kalau sudah ada koordinat
            if listing.get("koordinat") and args.skip_existing:
                # Just compute nearest_university kalau belum ada
                if not listing.get("nearest_university"):
                    lat, lng = listing["koordinat"]
                    name, dist = find_nearest_university(lat, lng)
                    listing["nearest_university"] = name
                    listing["jarak_kampus_km"] = dist
                out_f.write(json.dumps(listing, ensure_ascii=False) + "\n")
                enriched.append(listing)
                continue

            # Geocode
            alamat = listing.get("alamat", "")
            if not alamat:
                out_f.write(json.dumps(listing, ensure_ascii=False) + "\n")
                continue

            query = f"{alamat}, Bandar Lampung"
            if google_key and args.use_google:
                coords = google_geocode(query, google_key)
                time.sleep(0.1)  # Google rate limit much higher
            else:
                coords = nominatim_geocode(query)
                time.sleep(1.1)  # Nominatim 1 req/sec

            if coords:
                listing["koordinat"] = coords
                name, dist = find_nearest_university(coords[0], coords[1])
                listing["nearest_university"] = name
                listing["jarak_kampus_km"] = dist
                geocode_count += 1
                if i % 50 == 0:
                    print(f"[{i}/{len(listings)}] geocoded={geocode_count}")

            out_f.write(json.dumps(listing, ensure_ascii=False) + "\n")
            enriched.append(listing)

    print(f"\n[done] {len(enriched)} enriched listings -> {args.output}")
    print(f"[geocoded] {geocode_count} new geocodes")

    # Stats: distribusi nearest university
    from collections import Counter

    univ_counts = Counter(l.get("nearest_university") for l in enriched if l.get("nearest_university"))
    print(f"\n[nearest university distribution]")
    for univ, cnt in univ_counts.most_common():
        print(f"  {univ}: {cnt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

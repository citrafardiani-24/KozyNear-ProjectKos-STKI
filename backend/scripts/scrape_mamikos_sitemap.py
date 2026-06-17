"""Scrape Mamikos sitemap kategori Lampung -- static HTTP only (no JS).

Mamikos individual listing pages JS-rendered, tapi:
- Sitemap (XML) accessible + lists 329 kategori URL Lampung
- Tiap kategori page punya static HTML dengan:
  - Page title: 'Kost AC Lampung Murah - Tersedia 82 Kost'
  - Breadcrumb JSON-LD (kecamatan hierarchy)
  - Meta description (deskripsi kategori)

Tujuan: extract REAL Mamikos metadata sebagai signal augmentation untuk
synthetic data. Output JSON dengan struktur per-kategori.

Note: ini BUKAN scraping individual rooms (need JS render via Playwright).
Untuk full real data, butuh Playwright + Chromium (user authorize separately).

Usage:
    cd backend
    python -m scripts.scrape_mamikos_sitemap \\
        --output ../data/raw/mamikos_categories.json \\
        --max 50

Polite scraping: 3-7 detik delay antar request.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path
from typing import Optional

import requests

SITEMAP_URL = "https://mamikos.com/sitemaps/sitemap/sitemap-main.xml"
LAMPUNG_PATTERN = re.compile(r"lampung", re.IGNORECASE)

# Realistic browser UA pool
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0 Safari/537.36",
]


def headers(rng: random.Random) -> dict:
    return {
        "User-Agent": rng.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "id-ID,id;q=0.9,en;q=0.8",
        "Cache-Control": "max-age=0",
    }


def fetch_sitemap() -> list[str]:
    """Fetch + parse sitemap, return Lampung URL list."""
    print(f"[fetch sitemap] {SITEMAP_URL}")
    r = requests.get(SITEMAP_URL, headers=headers(random.Random()), timeout=30)
    r.raise_for_status()

    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    root = ET.fromstring(r.text)
    urls = root.findall(".//sm:url/sm:loc", ns)
    lampung = [u.text for u in urls if u.text and LAMPUNG_PATTERN.search(u.text)]
    print(f"[sitemap] {len(lampung)} Lampung URLs found")
    return lampung


def clean_text(text: str) -> str:
    """Normalize whitespace + non-breaking spaces (Mamikos pakai \\xa0)."""
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def extract_category_metadata(html: str, url: str) -> dict:
    """Extract metadata dari static HTML kategori page."""
    data: dict = {"url": url}

    # Title pattern: 'Kost AC Lampung Murah - Tersedia 82 Kost'
    title_match = re.search(r"<title>([^<]+)</title>", html)
    if title_match:
        data["title"] = clean_text(title_match.group(1))
        # Extract count: 'Tersedia X Kost'
        count_match = re.search(r"Tersedia\s+(\d+)\s+Kost", data["title"])
        if count_match:
            data["listing_count"] = int(count_match.group(1))

    # Meta description
    desc_match = re.search(
        r'<meta name="description" content="([^"]+)"', html, re.IGNORECASE,
    )
    if desc_match:
        data["meta_description"] = desc_match.group(1).strip()

    # H1
    h1_match = re.search(r"<h1[^>]*>([^<]+)</h1>", html, re.IGNORECASE)
    if h1_match:
        data["h1"] = h1_match.group(1).strip()

    # JSON-LD blocks
    ld_matches = re.findall(
        r"<script[^>]+application/ld\+json[^>]*>([^<]+)</script>",
        html,
        re.DOTALL,
    )
    data["json_ld_count"] = len(ld_matches)
    for ld_text in ld_matches:
        try:
            ld = json.loads(ld_text.strip())
            if isinstance(ld, dict):
                if ld.get("@type") == "BreadcrumbList":
                    data["breadcrumb"] = [
                        item.get("item", {}).get("name") or item.get("name")
                        for item in ld.get("itemListElement", [])
                    ]
        except json.JSONDecodeError:
            continue

    # Parse URL slug untuk category type
    # Pattern: /kost/kost-{feature}-{location}-murah
    slug_match = re.search(r"/kost/kost-(.+?)-murah", url)
    if slug_match:
        slug = slug_match.group(1)
        # Identify feature vs location
        # Lampung-specific: bandar-lampung, lampung
        if "bandar-lampung" in slug:
            location = "Bandar Lampung"
            feature = slug.replace("-bandar-lampung", "").replace("bandar-lampung-", "").strip("-")
        else:
            location = "Lampung"
            feature = slug.replace("-lampung", "").replace("lampung-", "").strip("-")
        data["slug_feature"] = feature
        data["slug_location"] = location

    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max", type=int, default=0, help="Max URLs to scrape (0=all)")
    parser.add_argument("--min-delay", type=float, default=3.0)
    parser.add_argument("--max-delay", type=float, default=7.0)
    args = parser.parse_args()

    rng = random.Random(42)

    # 1. Fetch sitemap
    lampung_urls = fetch_sitemap()
    if args.max > 0:
        lampung_urls = lampung_urls[: args.max]
        print(f"[limit] capped to {args.max} URLs")

    # 2. Scrape each URL
    results: list[dict] = []
    for i, url in enumerate(lampung_urls, start=1):
        try:
            # Politeness delay
            if i > 1:
                delay = rng.uniform(args.min_delay, args.max_delay)
                time.sleep(delay)

            r = requests.get(url, headers=headers(rng), timeout=20)
            if r.status_code == 200:
                meta = extract_category_metadata(r.text, url)
                meta["scraped_at"] = date.today().isoformat()
                meta["http_status"] = 200
                results.append(meta)
                title = meta.get("title", "")[:60]
                count = meta.get("listing_count", "?")
                print(f"[{i}/{len(lampung_urls)}] {title} ({count} kos)")
            else:
                print(f"[{i}/{len(lampung_urls)}] SKIP {url} (HTTP {r.status_code})")
        except Exception as e:
            print(f"[{i}/{len(lampung_urls)}] FAIL {url} -- {type(e).__name__}: {e}")

    # 3. Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    total_listings = sum(r.get("listing_count", 0) for r in results)
    print(f"\n[done] {len(results)} categories scraped, total {total_listings} listings "
          f"di Mamikos -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Discovery: scrape Mamikos kategori pages untuk dapat /room/ slug URLs.

Mamikos kategori pages JS-rendered, jadi butuh Playwright. Tapi kita hanya
extract slug URLs (no metadata) — much faster dari full card extraction.

Output: text file dengan satu URL per baris (siap di-feed ke
extract_mamikos_detail.py).

Usage:
  cd backend
  python -m scripts.discover_mamikos_slugs \\
      --output ../data/raw/_discovered_slugs.txt \\
      --max-per-category 100
"""
from __future__ import annotations

import argparse
import io
import random
import re
import sys
import time
from pathlib import Path
from typing import Set

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# Mamikos kategori URL slugs untuk Bandar Lampung
DEFAULT_CATEGORIES = [
    "kost-bandar-lampung-murah",
    "kost-mahasiswa-bandar-lampung-murah",
    "kost-bulanan-bandar-lampung-murah",
    "kost-putra-bandar-lampung-murah",
    "kost-putri-bandar-lampung-murah",
    "kost-campur-bandar-lampung-murah",
    "kost-eksklusif-bandar-lampung-murah",
    "kost-ac-bandar-lampung-murah",
    "kost-wifi-bandar-lampung-murah",
    "kost-kamar-mandi-dalam-bandar-lampung-murah",
    "kost-dekat-unila-murah",
    "kost-dekat-itera-murah",
    "kost-dekat-darmajaya-murah",
]

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
]


def discover_category(page, cat_url: str, max_listings: int, rng: random.Random,
                       delay_min: float, delay_max: float) -> Set[str]:
    """Render kategori page + extract all /room/ href URLs."""
    print(f"[kategori] {cat_url}")
    try:
        page.goto(cat_url, wait_until="networkidle", timeout=45000)
    except Exception as e:
        print(f"  [load fail] {type(e).__name__}: {e}")
        return set()
    time.sleep(2)

    found: Set[str] = set()
    pages_done = 0
    max_pages = 8

    while len(found) < max_listings and pages_done < max_pages:
        for _ in range(4):
            page.mouse.wheel(0, 2400)
            time.sleep(rng.uniform(0.8, 1.5))

        hrefs = page.evaluate("""
            () => {
                const links = document.querySelectorAll('a[href*="/room/"]');
                const out = new Set();
                for (const a of links) {
                    if (a.href && a.href.includes('/room/')) {
                        const clean = a.href.split('?')[0].split('#')[0];
                        out.add(clean);
                    }
                }
                return Array.from(out);
            }
        """)
        before = len(found)
        for h in hrefs:
            found.add(h)
        new_count = len(found) - before
        print(f"  page {pages_done + 1}: +{new_count} URLs (total {len(found)})")

        if new_count == 0:
            break

        try:
            next_btn = page.locator(
                'button:has-text("Selanjutnya"), a:has-text("Selanjutnya"), .pagination-next'
            ).first
            if next_btn.count() > 0 and next_btn.is_visible(timeout=2000):
                next_btn.click(timeout=5000)
                time.sleep(rng.uniform(2.0, 4.0))
                pages_done += 1
            else:
                break
        except Exception:
            break

        time.sleep(rng.uniform(delay_min, delay_max))

    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--categories", type=str, default="",
                        help="Comma-separated kategori slugs (default: 14 BDL categories)")
    parser.add_argument("--max-per-category", type=int, default=100)
    parser.add_argument("--min-delay", type=float, default=3.0)
    parser.add_argument("--max-delay", type=float, default=6.0)
    args = parser.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("ERROR: pip install playwright + python -m playwright install chromium")
        return 1

    rng = random.Random(42)
    cats = args.categories.split(",") if args.categories else DEFAULT_CATEGORIES
    print(f"[start] {len(cats)} kategori, max {args.max_per_category} URLs each")

    all_urls: Set[str] = set()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=rng.choice(USER_AGENTS),
            locale="id-ID",
            viewport={"width": 1280, "height": 900},
        )
        page = context.new_page()

        for slug in cats:
            cat_url = f"https://mamikos.com/kost/{slug.strip()}"
            try:
                urls = discover_category(
                    page, cat_url, args.max_per_category, rng,
                    args.min_delay, args.max_delay,
                )
                all_urls.update(urls)
                print(f"  [kategori done] {slug}: {len(urls)} URLs (total unique {len(all_urls)})")
            except Exception as e:
                print(f"  [kategori FAIL] {slug}: {type(e).__name__}: {e}")
            time.sleep(rng.uniform(args.min_delay, args.max_delay))

        browser.close()

    # Write unique URLs (sorted untuk deterministic)
    with args.output.open("w", encoding="utf-8") as f:
        for u in sorted(all_urls):
            f.write(u + "\n")

    print(f"\n[done] {len(all_urls)} unique slug URLs -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

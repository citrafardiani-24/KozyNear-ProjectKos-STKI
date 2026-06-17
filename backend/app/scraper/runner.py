"""CLI runner untuk scraper.

Usage:
    # Dari backend/ dengan venv activated
    python -m app.scraper.runner --source mamikos --max 1500 \\
        --output ../data/raw/mamikos.jsonl

    # Debug mode (5 listing, longer delay)
    python -m app.scraper.runner --source mamikos --max 5 \\
        --output ../data/raw/test.jsonl --min-delay 4 --max-delay 8

    # Playwright fallback (kalau Mamikos JS-rendered)
    python -m app.scraper.runner --source mamikos --max 50 \\
        --use-playwright --output ../data/raw/mamikos_pw.jsonl

    # OLX fallback
    python -m app.scraper.runner --source olx --max 500 \\
        --output ../data/raw/olx.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from loguru import logger

from .base import BaseScraper, ScraperConfig
from .mamikos import MamikosScraper
from .olx import OLXScraper


SCRAPER_REGISTRY: dict[str, type[BaseScraper]] = {
    "mamikos": MamikosScraper,
    "olx": OLXScraper,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="TKI-KOS scraper runner")
    parser.add_argument(
        "--source", choices=list(SCRAPER_REGISTRY.keys()), required=True,
    )
    parser.add_argument("--max", type=int, default=1500)
    parser.add_argument("--output", type=Path, required=True, help="JSONL output path")
    parser.add_argument("--cache-dir", type=Path, default=Path(".scrape_cache"))
    parser.add_argument(
        "--use-playwright",
        action="store_true",
        help="Mamikos only: enable Playwright headless fallback",
    )
    parser.add_argument("--min-delay", type=float, default=2.0)
    parser.add_argument("--max-delay", type=float, default=5.0)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    config = ScraperConfig(
        min_delay_sec=args.min_delay,
        max_delay_sec=args.max_delay,
        cache_dir=args.cache_dir,
    )

    scraper_cls = SCRAPER_REGISTRY[args.source]
    if args.source == "mamikos":
        scraper: BaseScraper = MamikosScraper(config, use_playwright=args.use_playwright)
    else:
        scraper = scraper_cls(config)

    logger.info(
        f"[start] source={args.source} max={args.max} "
        f"output={args.output} delay={args.min_delay}-{args.max_delay}s"
    )

    count = 0
    with args.output.open("w", encoding="utf-8") as f:
        for listing in scraper.crawl(max_listings=args.max):
            f.write(json.dumps(asdict(listing), ensure_ascii=False) + "\n")
            count += 1
            if count % 50 == 0:
                logger.info(f"[progress] {count}/{args.max}")

    logger.info(f"[done] {count} listings -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

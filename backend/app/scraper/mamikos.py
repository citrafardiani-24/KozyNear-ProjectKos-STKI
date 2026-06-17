"""Mamikos scraper.

Strategy 2-tier:
1. Default: requests + BeautifulSoup (lightweight). Cocok kalau Mamikos
   serve content via SSR atau ada __NEXT_DATA__ JSON di static HTML.
2. Fallback: Playwright headless Chrome (use_playwright=True). Butuh
   `pip install playwright && playwright install chromium`.

URL discovery:
- Sitemap: mamikos.com/sitemaps/sitemap/sitemap-main.xml (filter Lampung)
- Filter category pattern: /kost/kost-<feature>-lampung-murah

Target: ≥1500 listings dengan deskripsi ≥100 kata.

CATATAN UNTUK TIM SCRAPER (Anggota A):
- Selector di parse_listing_urls() dan parse_detail() adalah PLACEHOLDER
  yang perlu di-verify dengan layout Mamikos terkini (lihat README.md scraper).
- Kalau Mamikos pakai Next.js, prioritaskan parse __NEXT_DATA__ JSON di
  parse_listing_urls() — JAUH lebih reliable dari CSS selectors.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import date
from typing import Iterator, Optional

from bs4 import BeautifulSoup
from loguru import logger

from .base import BaseScraper, Listing, ScraperConfig
from .utils import (
    clean_text,
    detect_tipe,
    extract_price,
    normalize_fasilitas,
    word_count,
)


SITEMAP_URL = "https://mamikos.com/sitemaps/sitemap/sitemap-main.xml"
LAMPUNG_URL_PATTERN = re.compile(r"lampung", re.IGNORECASE)
MIN_DESKRIPSI_WORDS = 100  # course requirement


class MamikosScraper(BaseScraper):
    name = "mamikos"

    def __init__(
        self,
        config: Optional[ScraperConfig] = None,
        use_playwright: bool = False,
        min_deskripsi_words: int = MIN_DESKRIPSI_WORDS,
    ):
        super().__init__(config)
        self.use_playwright = use_playwright
        self.min_deskripsi_words = min_deskripsi_words

    # -------------------------------------------------------------------------
    # Seed URLs (kategori filter Lampung dari sitemap)
    # -------------------------------------------------------------------------
    def seed_urls(self) -> Iterator[str]:
        """Yield kategori filter URLs untuk Lampung."""
        try:
            sitemap_xml = self.fetch(SITEMAP_URL)
        except Exception as e:
            logger.error(f"[sitemap fail] {e} -- fallback hardcoded")
            yield from self._fallback_seed_urls()
            return

        try:
            root = ET.fromstring(sitemap_xml)
        except ET.ParseError as e:
            logger.error(f"[sitemap parse fail] {e} -- fallback hardcoded")
            yield from self._fallback_seed_urls()
            return

        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        urls = root.findall(".//sm:url/sm:loc", ns)
        lampung_urls = [
            url.text
            for url in urls
            if url.text and LAMPUNG_URL_PATTERN.search(url.text)
        ]

        logger.info(f"[seed] {len(lampung_urls)} Lampung URLs dari sitemap")
        for url in lampung_urls:
            yield url

    def _fallback_seed_urls(self) -> Iterator[str]:
        """Hardcoded category URLs (confirmed exist Mei 2026)."""
        yield "https://mamikos.com/kost/kost-ac-lampung-murah"
        yield "https://mamikos.com/kost/kost-kamar-mandi-dalam-lampung-murah"
        yield "https://mamikos.com/kost/kost-parkir-mobil-lampung-murah"
        yield "https://mamikos.com/kost/kost-500k-lampung-murah"
        yield "https://mamikos.com/kost/kost-300k-lampung-murah"

    # -------------------------------------------------------------------------
    # Parse listing URLs dari halaman kategori
    # -------------------------------------------------------------------------
    def parse_listing_urls(self, page_html: str, base_url: str) -> list[str]:
        """Extract URL detail tiap kos dari satu halaman kategori.

        TODO tim: prioritize __NEXT_DATA__ JSON parse kalau Mamikos pakai Next.js.
        """
        soup = BeautifulSoup(page_html, "lxml")
        urls: list[str] = []

        # Pattern 1: direct <a href> link ke /room/...
        for anchor in soup.select('a[href*="/room/"]'):
            href = anchor.get("href", "")
            if href.startswith("/"):
                href = f"https://mamikos.com{href}"
            if href not in urls:
                urls.append(href)

        # Pattern 2: cari embedded JSON Next.js (kalau ada)
        next_data = soup.find("script", id="__NEXT_DATA__")
        if next_data and next_data.string:
            logger.debug("[parse] __NEXT_DATA__ found -- tim TODO: parse JSON utk URLs")
            # TODO tim:
            # import json
            # data = json.loads(next_data.string)
            # rooms = data.get("props", {}).get("pageProps", {}).get("rooms", [])
            # for room in rooms:
            #     slug = room.get("slug")
            #     if slug:
            #         urls.append(f"https://mamikos.com/room/{slug}")

        logger.info(f"[parse] {base_url} -> {len(urls)} listing URLs")
        return urls

    # -------------------------------------------------------------------------
    # Parse detail kos
    # -------------------------------------------------------------------------
    def parse_detail(self, detail_html: str, url: str) -> Optional[Listing]:
        """Parse halaman detail kos jadi Listing object.

        TODO tim: verify selectors. Selectors di bawah PLACEHOLDER.
        """
        soup = BeautifulSoup(detail_html, "lxml")

        # Detect SPA shell (page kosong, JS-rendered)
        if self._page_likely_empty(soup):
            if self.use_playwright:
                logger.warning(f"[empty page] {url} -- fallback Playwright")
                detail_html = self._fetch_with_playwright(url)
                soup = BeautifulSoup(detail_html, "lxml")
            else:
                logger.warning(f"[empty page] {url} -- skip (enable use_playwright)")
                return None

        # Extract fields (PLACEHOLDER selectors)
        judul_el = soup.select_one("h1, h2.title, [class*=title]")
        judul = clean_text(judul_el.get_text() if judul_el else "")

        deskripsi_el = soup.select_one(
            "[class*=description], [class*=deskripsi], "
            "section.description, div#description"
        )
        deskripsi = clean_text(deskripsi_el.get_text() if deskripsi_el else "")

        # Course requirement: ≥100 kata
        if word_count(deskripsi) < self.min_deskripsi_words:
            logger.debug(f"[skip <{self.min_deskripsi_words} kata] {url}")
            return None

        # Extract harga dari seluruh page (regex tahan banting)
        harga = extract_price(soup.get_text())

        # Fasilitas
        fasilitas_els = soup.select(
            "[class*=facility], [class*=fasilitas], ul.facility li, ul.fasilitas li"
        )
        raw_fasilitas = [
            el.get_text(strip=True)
            for el in fasilitas_els
            if el.get_text(strip=True)
        ]
        fasilitas = normalize_fasilitas(raw_fasilitas)

        # Tipe
        tipe = detect_tipe(judul + " " + deskripsi)

        # Alamat
        alamat_el = soup.select_one(
            "[class*=address], [class*=alamat], [class*=location]"
        )
        alamat = clean_text(alamat_el.get_text() if alamat_el else "")

        return Listing(
            id=self._extract_id_from_url(url),
            judul=judul,
            deskripsi=deskripsi,
            harga_per_bulan=harga,
            tipe=tipe,
            fasilitas=fasilitas,
            alamat=alamat,
            kecamatan=None,  # TODO tim: extract dari alamat / breadcrumb
            koordinat=None,  # TODO tim: extract dari map widget data attr
            jarak_kampus_km=None,  # post-process dengan area UNILA list
            url_source=url,
            scrape_date=date.today().isoformat(),
            source="mamikos",
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _page_likely_empty(self, soup: BeautifulSoup) -> bool:
        """Heuristic: page hanya SPA shell tanpa content."""
        body = soup.find("body")
        if not body:
            return True
        text_length = len(body.get_text(strip=True))
        return text_length < 500

    def _fetch_with_playwright(self, url: str) -> str:
        """Fallback ke Playwright untuk JS-rendered page.

        Requires: pip install playwright && playwright install chromium
        """
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as e:
            raise RuntimeError(
                "Playwright belum di-install. "
                "Run: pip install playwright && playwright install chromium"
            ) from e

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                page = browser.new_page(user_agent=self._current_ua)
                page.goto(
                    url,
                    wait_until="networkidle",
                    timeout=int(self.config.timeout_sec * 1000),
                )
                content = page.content()
            finally:
                browser.close()
        return content

    def _extract_id_from_url(self, url: str) -> str:
        """ID unik dari URL Mamikos (pattern: /room/<slug>)."""
        match = re.search(r"/room/([\w-]+)", url)
        if match:
            return match.group(1)
        return url

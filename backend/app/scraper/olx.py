"""OLX kos-kosan fallback scraper (skeleton).

Activate kalau Mamikos block / gagal. URL pattern OLX kost Lampung:
https://www.olx.co.id/lampung/kost-kost-an_c4709

TODO tim (Anggota A — Scraper):
- Verify selector listing card (biasanya <a data-aut-id="itemBox"> di OLX)
- Implementasi pagination (URL ?page=2, atau scroll)
- Parser detail page (harga, deskripsi, alamat)
- Test dengan fixture HTML sample
"""

from __future__ import annotations

from datetime import date
from typing import Iterator, Optional

from bs4 import BeautifulSoup
from loguru import logger

from .base import BaseScraper, Listing
from .utils import clean_text, detect_tipe, extract_price, word_count


MIN_DESKRIPSI_WORDS = 100


class OLXScraper(BaseScraper):
    name = "olx"

    BASE_URL = "https://www.olx.co.id"
    LAMPUNG_KOST_URL = f"{BASE_URL}/lampung/kost-kost-an_c4709"

    def seed_urls(self) -> Iterator[str]:
        yield self.LAMPUNG_KOST_URL
        # TODO tim: tambah pagination
        # for page in range(2, 50):
        #     yield f"{self.LAMPUNG_KOST_URL}?page={page}"

    def parse_listing_urls(self, page_html: str, base_url: str) -> list[str]:
        """TODO tim: implementasi parser. Sample selector di bawah PLACEHOLDER."""
        soup = BeautifulSoup(page_html, "lxml")
        urls: list[str] = []
        # OLX biasanya pakai data-aut-id="itemBox" untuk card
        for anchor in soup.select('a[href*="/item/"]'):
            href = anchor.get("href", "")
            if href.startswith("/"):
                href = f"{self.BASE_URL}{href}"
            if href not in urls:
                urls.append(href)
        logger.info(f"[parse] {base_url} -> {len(urls)} OLX listing URLs")
        return urls

    def parse_detail(self, detail_html: str, url: str) -> Optional[Listing]:
        """TODO tim: implementasi parser detail."""
        soup = BeautifulSoup(detail_html, "lxml")
        judul_el = soup.select_one("h1")
        judul = clean_text(judul_el.get_text() if judul_el else "")

        # OLX biasanya punya div[data-aut-id="itemDescriptionContent"]
        desc_el = soup.select_one('[data-aut-id="itemDescriptionContent"], div.description')
        deskripsi = clean_text(desc_el.get_text() if desc_el else "")

        if word_count(deskripsi) < MIN_DESKRIPSI_WORDS:
            return None

        harga = extract_price(soup.get_text())
        tipe = detect_tipe(judul + " " + deskripsi)

        return Listing(
            id=url.rstrip("/").split("/")[-1],
            judul=judul,
            deskripsi=deskripsi,
            harga_per_bulan=harga,
            tipe=tipe,
            url_source=url,
            scrape_date=date.today().isoformat(),
            source="olx",
        )

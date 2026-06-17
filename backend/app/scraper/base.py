"""Base abstract scraper dengan rate limiting, UA rotation, response cache.

Pattern: subclass implements seed_urls() + parse_listing_urls() + parse_detail();
base handles HTTP politeness + retry + caching.

Analogi Laravel: ini seperti abstract Job base class. Subclass-nya
(MamikosScraper, OLXScraper) handle specific dispatch logic, base handle
infrastructure (queue, retry, logging).
"""

from __future__ import annotations

import hashlib
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import httpx
from fake_useragent import UserAgent
from loguru import logger


@dataclass
class ScraperConfig:
    """Konfigurasi politeness untuk scraper."""
    min_delay_sec: float = 2.0
    max_delay_sec: float = 5.0
    timeout_sec: float = 30.0
    max_retries: int = 3
    cache_dir: Optional[Path] = None  # None = no cache
    user_agents: list[str] = field(default_factory=list)  # empty = fake_useragent
    rotate_ua_every_n_requests: int = 1


@dataclass
class Listing:
    """Standardized listing schema (selaras dengan brief course)."""
    id: str
    judul: str
    deskripsi: str
    harga_per_bulan: Optional[int] = None
    tipe: Optional[str] = None  # putra | putri | campur
    fasilitas: list[str] = field(default_factory=list)
    alamat: Optional[str] = None
    kecamatan: Optional[str] = None
    koordinat: Optional[tuple[float, float]] = None
    jarak_kampus_km: Optional[float] = None
    url_source: Optional[str] = None
    scrape_date: Optional[str] = None
    source: str = "unknown"  # mamikos | olx | cove


class BaseScraper(ABC):
    """Abstract base scraper. Subclass per data source."""

    name: str = "base"  # override per subclass

    def __init__(self, config: Optional[ScraperConfig] = None):
        self.config = config or ScraperConfig()
        self._ua_pool: Optional[UserAgent] = UserAgent() if not self.config.user_agents else None
        self._request_count = 0
        self._current_ua = self._pick_ua()
        if self.config.cache_dir:
            self.config.cache_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # User-Agent rotation
    # -------------------------------------------------------------------------
    def _pick_ua(self) -> str:
        if self.config.user_agents:
            return random.choice(self.config.user_agents)
        assert self._ua_pool is not None
        return self._ua_pool.random

    def _maybe_rotate_ua(self) -> None:
        if self._request_count % self.config.rotate_ua_every_n_requests == 0:
            self._current_ua = self._pick_ua()

    # -------------------------------------------------------------------------
    # Politeness delay
    # -------------------------------------------------------------------------
    def _delay(self) -> None:
        sleep_for = random.uniform(self.config.min_delay_sec, self.config.max_delay_sec)
        time.sleep(sleep_for)

    # -------------------------------------------------------------------------
    # File-based response cache (untuk dev, hindari re-hit target server)
    # -------------------------------------------------------------------------
    def _cache_path(self, url: str) -> Optional[Path]:
        if not self.config.cache_dir:
            return None
        key = hashlib.sha256(url.encode()).hexdigest()[:16]
        return self.config.cache_dir / f"{self.name}_{key}.html"

    def _cache_get(self, url: str) -> Optional[str]:
        path = self._cache_path(url)
        if path and path.exists():
            logger.debug(f"[cache HIT] {url}")
            return path.read_text(encoding="utf-8")
        return None

    def _cache_set(self, url: str, content: str) -> None:
        path = self._cache_path(url)
        if path:
            path.write_text(content, encoding="utf-8")

    # -------------------------------------------------------------------------
    # Core fetch dengan retry + politeness
    # -------------------------------------------------------------------------
    def fetch(self, url: str, force_refresh: bool = False) -> str:
        """Fetch URL: cache → delay → request → cache.

        Raises httpx.HTTPError kalau gagal setelah max_retries.
        """
        if not force_refresh:
            cached = self._cache_get(url)
            if cached is not None:
                return cached

        # Politeness delay sebelum request (kecuali request pertama)
        if self._request_count > 0:
            self._delay()
        self._maybe_rotate_ua()
        self._request_count += 1

        last_error: Optional[Exception] = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                headers = {
                    "User-Agent": self._current_ua,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "id-ID,id;q=0.9,en;q=0.8",
                }
                with httpx.Client(
                    timeout=self.config.timeout_sec,
                    follow_redirects=True,
                ) as client:
                    resp = client.get(url, headers=headers)
                    resp.raise_for_status()
                    content = resp.text
                    self._cache_set(url, content)
                    return content
            except httpx.HTTPError as e:
                logger.warning(
                    f"[fetch retry {attempt}/{self.config.max_retries}] {url} -- {e}"
                )
                last_error = e
                # Exponential backoff sebelum retry
                time.sleep(2**attempt)

        assert last_error is not None
        raise last_error

    # -------------------------------------------------------------------------
    # Abstract methods — subclass MUST implement
    # -------------------------------------------------------------------------
    @abstractmethod
    def seed_urls(self) -> Iterator[str]:
        """Yield URL kategori/index awal untuk di-crawl."""

    @abstractmethod
    def parse_listing_urls(self, page_html: str, base_url: str) -> list[str]:
        """Extract URL detail kos dari satu halaman kategori."""

    @abstractmethod
    def parse_detail(self, detail_html: str, url: str) -> Optional[Listing]:
        """Parse halaman detail kos jadi Listing. Return None kalau skip."""

    # -------------------------------------------------------------------------
    # High-level orchestration
    # -------------------------------------------------------------------------
    def crawl(self, max_listings: Optional[int] = None) -> Iterator[Listing]:
        """Streaming crawl: seed → listing URLs → detail → Listing.

        Caller bisa save streaming ke file/DB (hemat memory).
        """
        seen_urls: set[str] = set()
        emitted = 0

        for seed_url in self.seed_urls():
            try:
                index_html = self.fetch(seed_url)
            except Exception as e:
                logger.error(f"[seed fail] {seed_url} -- {e}")
                continue

            listing_urls = self.parse_listing_urls(index_html, seed_url)
            logger.info(f"[seed] {seed_url} -> {len(listing_urls)} listing URLs")

            for url in listing_urls:
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                try:
                    detail_html = self.fetch(url)
                    listing = self.parse_detail(detail_html, url)
                except Exception as e:
                    logger.warning(f"[detail fail] {url} -- {e}")
                    continue

                if listing is None:
                    continue

                yield listing
                emitted += 1
                if max_listings and emitted >= max_listings:
                    logger.info(f"[crawl] reached max_listings={max_listings}, stop")
                    return

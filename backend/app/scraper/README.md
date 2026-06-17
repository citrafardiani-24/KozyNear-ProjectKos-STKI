# Scraper Module

Polite scraping untuk Mamikos kos-kosan, dengan OLX sebagai fallback.

## Architecture

```
runner.py (CLI)
    └─ uses
       BaseScraper (rate limit, UA rotation, cache, retry)
       ├─ MamikosScraper (requests+BS4 default, Playwright fallback)
       └─ OLXScraper (skeleton — TODO tim)

utils.py — extract_price, detect_tipe, normalize_fasilitas, clean_text
```

## Quick Start

```bash
# Dari backend/, dengan venv activated
cd backend
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Mac/Linux

pip install -r requirements.txt

# 1. Test small batch (5 listings)
python -m app.scraper.runner --source mamikos --max 5 \
    --output ../data/raw/test.jsonl --min-delay 4 --max-delay 8

# 2. Cek output
type ..\data\raw\test.jsonl    # Windows
# cat ../data/raw/test.jsonl   # Mac/Linux

# 3. Kalau lookalike sukses, full scrape
python -m app.scraper.runner --source mamikos --max 1500 \
    --output ../data/raw/mamikos.jsonl
```

## What's Done (oleh mentor — scaffold)

- [x] `base.py` — abstract BaseScraper dengan:
  - Rate limiting (random delay 2-5s, configurable via `ScraperConfig`)
  - User-Agent rotation (via `fake_useragent`)
  - File-based response cache (sha256 hash of URL → HTML file di `.scrape_cache/`)
  - Retry dengan exponential backoff (max 3x)
  - `Listing` dataclass (schema sesuai brief STKI)
- [x] `utils.py` — helpers tested:
  - `extract_price()` handle Rp X.XXX, Xjt, Xrb, Xk
  - `detect_tipe()` putra/putri/campur (Mamikos gender codes 0/1/2)
  - `normalize_fasilitas()` map varian ke vocab kanonis (15+ entries)
  - `clean_text()`, `word_count()`, `truncate()`
- [x] `mamikos.py` — skeleton:
  - Sitemap-based URL discovery (filter "lampung")
  - Fallback 5 hardcoded category URLs (confirmed exist Mei 2026)
  - SPA shell detection → auto-Playwright fallback
  - Schema mapping dengan TODO selector placeholders
- [x] `olx.py` — minimal skeleton (TODO tim)
- [x] `runner.py` — CLI dengan args
- [x] `tests/test_scraper.py` — unit test untuk utils

## What Tim Lead/Scraper (Anggota A) NEEDS to Do

### Priority 1 — Verify Mamikos Selectors

File: `mamikos.py` di method `parse_listing_urls()` dan `parse_detail()`.

Step debug:

1. Run smoke test:
   ```bash
   python -m app.scraper.runner --source mamikos --max 5 \
       --output ../data/raw/test.jsonl --min-delay 4 --max-delay 8
   ```
2. Inspect `test.jsonl` — kalau empty/field None semua, selectors salah.
3. Buka `https://mamikos.com/kost/kost-ac-lampung-murah` di Chrome.
4. **Save Page Source** (Ctrl+U → Save As) ke `backend/tests/fixtures/mamikos_sample.html`.
5. Inspect listing card di DevTools → catat class name actual.
6. Update selectors di `mamikos.py`:
   - `parse_listing_urls`: ganti `'a[href*="/room/"]'` kalau pattern beda
   - `parse_detail`: update selector untuk `judul`, `deskripsi`, `harga`, `fasilitas`, `alamat`

### Priority 2 — Parse `__NEXT_DATA__` JSON (HIGHLY recommended)

Mamikos kemungkinan pakai Next.js — ada `<script id="__NEXT_DATA__" type="application/json">{...}</script>` dengan **semua listing data dalam JSON**. JAUH lebih reliable dari CSS selectors yang brittle.

Stub di `parse_listing_urls()`:

```python
import json

next_data = soup.find("script", id="__NEXT_DATA__")
if next_data and next_data.string:
    data = json.loads(next_data.string)
    rooms = data.get("props", {}).get("pageProps", {}).get("rooms", [])
    for room in rooms:
        slug = room.get("slug")
        if slug:
            urls.append(f"https://mamikos.com/room/{slug}")
```

Cek struktur JSON di Chrome DevTools console:
```javascript
JSON.parse(document.getElementById("__NEXT_DATA__").textContent)
```

### Priority 3 — Pagination

Mamikos category page mungkin punya pagination. Cek manual:
- Apakah ada tombol "Load More" / "Next" di bottom?
- URL `?page=2` valid?

Update `seed_urls()` di `mamikos.py` untuk yield page variants.

### Priority 4 — Geographic Filter (UNILA-area only)

Brief minta radius 5km dari kampus UNILA. Setelah scrape, filter listing yang **alamat-nya mengandung area UNILA**:
- Gedong Meneng
- Rajabasa
- Kedaton
- Sumantri Brojonegoro
- Labuhan Ratu
- Way Halim
- Tanjung Senang

Tambah method `_is_unila_area(listing.alamat) -> bool` di `mamikos.py` dan filter di `crawl()` sebelum yield.

### Priority 5 — OLX Fallback

`olx.py` masih skeleton. Implementasi:
- Verify selector di OLX Lampung Kost page
- Pagination handling
- Detail page parser

Pattern sama dengan Mamikos, just different selectors.

## Polite Scraping Discipline

JANGAN diabaikan:

- **Delay 2-5 detik antar request** (sudah default di `ScraperConfig`)
- **Cache aktif saat dev** (`--cache-dir .scrape_cache`) — hindari re-hit Mamikos saat debug
- **Respect HTTP 429** (retry exponential backoff sudah handle)
- **Stop kalau dapat banyak 403** — IP mungkin di-flag, ganti VPN / tunggu beberapa jam
- **JANGAN commit `.scrape_cache/`** atau hasil JSON ke git (sudah di-block oleh `.gitignore`)

## Anti-Patterns

- [BAD] `time.sleep(0.5)` — terlalu cepat, IP bakal di-blacklist
- [BAD] Hardcoded User-Agent yang sama tiap request — ditangkap anti-bot
- [BAD] Run full scrape tanpa test 5-10 listing dulu — bisa generate 1500 sampah
- [BAD] Commit hasil scrape ke git — `.gitignore` block, jangan force add
- [BAD] Lowercase deskripsi sebelum extract harga — regex bakal miss `Rp` capitalized

## Testing

```bash
# Run unit test scraper utils (no network)
cd backend
pytest tests/test_scraper.py -v
```

Saat ada `tests/fixtures/mamikos_sample.html`, bisa tambah integration test:

```python
def test_mamikos_parse_listing_urls():
    with open("tests/fixtures/mamikos_sample.html") as f:
        html = f.read()
    scraper = MamikosScraper()
    urls = scraper.parse_listing_urls(html, "https://mamikos.com/kost/...")
    assert len(urls) > 0
```

## Output Format

JSONL — satu listing per baris (streaming-friendly):

```json
{"id":"kos-ac-xyz-123","judul":"Kos AC Gedong Meneng","deskripsi":"...","harga_per_bulan":850000,"tipe":"putra","fasilitas":["ac","wifi","kamar mandi dalam"],"alamat":"Jl. Sumantri ...","kecamatan":null,"koordinat":null,"jarak_kampus_km":null,"url_source":"https://...","scrape_date":"2026-05-21","source":"mamikos"}
{"id":"...","judul":"...","..." }
```

Setelah scrape selesai, run preprocessing pipeline (Week 2) untuk bersihin dan generate `data/processed/corpus.json` dengan deskripsi yang sudah di-stem.

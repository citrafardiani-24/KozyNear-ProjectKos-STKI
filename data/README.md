# Data

## Folder Structure

| Folder | Content | Git-tracked? |
|--------|---------|--------------|
| `raw/` | Scrape output Mamikos (JSONL per listing) + discovery URLs | **Selective** — `mamikos_real_v2.jsonl` + `_discovered_slugs.txt` committed |
| `processed/` | Corpus setelah preprocessing pipeline (JSON) | **Selective** — `corpus.json` committed (untuk Docker COPY) |
| `indexes/` | Serialized IR indexes (TF-IDF pkl, BM25 pkl, IndoBERT+FAISS dir) | **Yes** — committed untuk fast Docker startup |

`.gitignore` whitelist (lihat root `.gitignore`):
- `!data/raw/mamikos_real_v2.jsonl` — sumber data REAL (source of truth)
- `!data/raw/_discovered_slugs.txt` — daftar URL discovery (reproducibility)
- `!data/processed/corpus.json`
- `!data/indexes/tfidf.pkl`, `!data/indexes/bm25.pkl`, `!data/indexes/indobert/`

`kozynear_combined.jsonl` (canonical corpus untuk preprocess) **gitignored** —
reconstructable via `rebuild_v2.py` dari `mamikos_real_v2.jsonl`.

## Data Acquisition — Real Mamikos Scrape

Corpus adalah **227 listing kos REAL** dari Mamikos.com (bukan synthetic).
Pipeline 3-tahap:

1. **Discovery** (`backend/scripts/discover_mamikos_slugs.py` + WebSearch):
   kumpulkan URL halaman detail `/room/`. Halaman kategori Mamikos
   JS-rendered dengan API listing ter-enkripsi (AES), jadi discovery via
   ~50 query `site:mamikos.com inurl:/room/` (per-kecamatan, per-universitas,
   per-fasilitas, per-harga, street-level) → **312 URL unik** (`_discovered_slugs.txt`).
2. **Detail extraction** (`backend/scripts/extract_mamikos_detail.py`):
   tiap halaman detail meng-*inject* `var detail = {...}` (JSON 146 field) di
   static HTML — fetch HTTP-only tanpa browser. Parse `_id`, `room_title`,
   `description` (cerita pemilik asli), `latitude`/`longitude`, harga, gender,
   fasilitas, booking_rules, owner_name, verification. Success ~76%.
3. **Canonical build** (`backend/scripts/rebuild_v2.py`):
   normalisasi kecamatan, hitung `jarak_kampus_km` (haversine ke 9 universitas),
   drop deskripsi kosong → `kozynear_combined.jsonl` (227 listing).

**Catatan**: field gender Mamikos hanya 0/1/2 → tipe **campur/putra/putri**
(tidak ada pasutri — "pasutri" kategori marketing, bukan tipe gender).

**Kenapa real, bukan synthetic**: web app ditujukan production (pengguna cari
kos nyata + Google Maps). Synthetic dibuang total karena koordinat fabricated
akan menaruh pin di lokasi salah. Iterasi awal pernah memakai 2000 synthetic;
arsip generator masih ada di `scripts/generate_synthetic_corpus.py` tapi tidak
dipakai. Trade-off: corpus lebih kecil (Mamikos BDL hanya ~300-400 listing unik)
ditebus dengan **keaslian 100%** (0 template-leak, koordinat real, url_source).

## Schema per Listing (real)

```json
{
  "id": "mamikos-15426487",
  "judul": "Kost Agape Kedaton Bandar Lampung",
  "deskripsi": "belakang UNILA (sangat dekat, jalan kaki 5mnt), dekat rel kereta",
  "harga_per_bulan": 600000,
  "tipe": "putri",
  "fasilitas": ["kasur", "lemari / storage", "meja", "kursi", "bantal", "jendela"],
  "alamat": null,
  "kecamatan": "Kedaton",
  "koordinat": [-5.367381, 105.249643],
  "jarak_kampus_km": 0.73,
  "kampus_terdekat": "UNILA",
  "url_source": "https://mamikos.com/room/kost-kota-bandar-lampung-kost-putri-murah-kost-agape-kedaton-bandar-lampung",
  "scrape_date": "2026-05-29",
  "source": "mamikos-real-v2",
  "owner_name": "Paulus Maruli Tamba",
  "available_room": 3,
  "rules": ["tidak boleh bawa anak"],
  "verified": true
}
```

## Statistik Corpus (real, 227 listing)

| Metric | Value |
|--------|-------|
| Total listings | 227 (100% real Mamikos) |
| Kecamatan covered | 18 unique (Bandar Lampung) |
| Tipe | campur 114 (50%), putri 85 (37%), putra 28 (13%) |
| Harga (per bulan) | min Rp 300k, median Rp 800k, max Rp 6jt |
| Deskripsi (kata) | median ~23 (cerita pemilik, terse) |
| Koordinat valid (Maps) | 226/227 (99.6%) |
| url_source / verified | 100% / 100% |

## Reproducibility

Rebuild canonical corpus + indexes dari sumber real:
```bash
cd backend
# 1. (Opsional) re-discover URL — atau pakai _discovered_slugs.txt yang ada
python -m scripts.discover_mamikos_slugs --output ../data/raw/_discovered_slugs.txt

# 2. Extract detail pages (HTTP-only, polite delay 3s)
python -m scripts.extract_mamikos_detail \
    --urls ../data/raw/_discovered_slugs.txt \
    --output ../data/raw/mamikos_real_v2.jsonl --delay 3

# 3. Build canonical real-only corpus
python -m scripts.rebuild_v2

# 4. Preprocess + build indexes
python -m scripts.preprocess_corpus --input ../data/raw/kozynear_combined.jsonl --output ../data/processed/corpus.json
python -m app.indexing.build --corpus ../data/processed/corpus.json --output-dir ../data/indexes

# 5. (Opsional) regenerate ground truth + eval
python -m scripts.generate_ground_truth --queries ../eval/queries.json --corpus ../data/processed/corpus.json --indexes-dir ../data/indexes --output-dir ../eval
python -m app.evaluation.runner --queries ../eval/queries.json --ground-truth ../eval/ground_truth.csv --indexes-dir ../data/indexes --output ../eval/results.csv
```

Audit data quality kapan saja: `python eval/_audit_dataset.py`.

# Data Deduplication Rules

> ⚠️ **DOKUMEN HISTORIS** (iterasi V1→V2). Pipeline data **saat ini**
> (100% real Mamikos, synthetic sudah di-drop) didokumentasikan di
> [`data/README.md`](../data/README.md). Beberapa file yang disebut di bawah
> (`mamikos_real.jsonl`, `_extra`, `_merged`, `kozynear_synthetic.jsonl`)
> **sudah dihapus** — bagian ini disimpan sebagai catatan metodologi dedup
> by-judul yang dipakai saat masih ada multi-batch real + synthetic.

Dokumen ini menjelaskan rule dedup yang dipakai saat membentuk
`data/raw/kozynear_combined.jsonl` (canonical corpus) dari sub-source.

## Sumber data

| File | Records | Source |
|---|---|---|
| `mamikos_real.jsonl` | 69 | Scrape Mamikos batch 1 (sitemap) |
| `mamikos_real_extra.jsonl` | 95 | Scrape Mamikos batch 2 (Playwright extra search) |
| `mamikos_real_merged.jsonl` | 122 | Dedup union dari batch 1 + 2 |
| `kozynear_synthetic.jsonl` | 2000 | Generated synthetic listings |
| `kozynear_combined.jsonl` | 2074 | merged + synthetic, post-cleaning |

## Rule dedup `mamikos_real_merged.jsonl`

**Catatan**: ID Mamikos (`mamikos-{listing_id}`) berbeda untuk listing fisik
yang sama karena re-scrape me-generate ID baru di sesi berbeda. Jadi dedup
by ID gak akan menemukan duplikasi — perlu dedup by konten.

**Aturan**: dedup by lowercased + stripped `judul`, prefer first occurrence.

```python
def dedup_judul(records):
    seen = set()
    out = []
    for r in records:
        key = r["judul"].strip().lower()
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out

real = load_jsonl("mamikos_real.jsonl")        # 69 records, 69 unique juduls
extra = load_jsonl("mamikos_real_extra.jsonl") # 95 records, 95 unique juduls

# Prefer real (batch 1) over extra (batch 2) untuk judul yang sama
merged = dedup_judul(real + extra)             # 122 records (69 + 53 new)
```

**Hasil**: 69 real (semua dipertahankan) + 53 extra (yang judul-nya belum
ada di real) = 122 unique juduls.

**Drop count**: 42 records dari extra di-skip karena judul-nya sudah ada di
real.

## Verifikasi

```python
real_ids = {r['id'] for r in real}
extra_ids = {r['id'] for r in extra}
merged_ids = {r['id'] for r in merged}

assert real_ids - extra_ids == real_ids                 # zero ID overlap
assert merged_ids <= real_ids | extra_ids               # subset of union
assert len({r['judul'].strip().lower() for r in merged}) == len(merged)  # all unique
```

## V2 Real Data Pipeline (2026-05-29, replaces v1 templated)

V1 data (122 listings di `mamikos_real_merged.jsonl`) menggunakan scrape card-only
+ `build_full_deskripsi()` template generator → deskripsi 100% templated (bukan
real owner story). V2 fix ini dengan extract langsung dari halaman detail
Mamikos via embedded `var detail = {...}` JSON.

**Pipeline v2** (3 scripts):
1. `backend/scripts/discover_mamikos_slugs.py` — Playwright crawl category pages
   untuk dapat `/room/` slug URLs (fallback: WebSearch `site:mamikos.com inurl:/room/`)
2. `backend/scripts/extract_mamikos_detail.py` — HTTP-only fetch detail pages,
   parse `var detail` (~28KB JSON dengan 146 fields), map ke canonical schema
3. `backend/scripts/rebuild_v2.py`: normalize, drop empty deskripsi (real-only),
   replace `kozynear_combined.jsonl`

**Schema v2** (extra fields vs v1):
- `koordinat` [lat, lng] — REAL Mamikos data (sebelumnya null)
- `kampus_terdekat` — computed via haversine vs 9 universitas
- `url_source` — canonical Mamikos URL (sebelumnya null)
- `owner_name`, `available_room`, `rules`, `verified`, `view_count`
- `id` — REAL Mamikos internal ID (`mamikos-{_id}`), bukan hash judul

**Hasil 2026-05-29:**
- Discovery: 117 URLs (12 WebSearch queries × ~10 results, deduped)
- Extraction: 86 successful (74%), 31 failed (listing inactive/removed)
- Authenticity: 86/86 dengan REAL deskripsi pemilik (0 template phrase)
- Coverage: 100% koordinat, 100% url_source, 100% Mamikos-verified

**Backups:** `data/raw/kozynear_combined.jsonl.v1.bak` (old templated),
`eval/*.csv.preV2.bak` (annotations sebelum filter v2 IDs).

## Rule cleaning `kozynear_combined.jsonl` (P0 remediation, 2026-05-29)

Setelah merge real+synth, ada 48 record yang di-drop sebagai data quality
remediation (lihat [eval/_audit_report.json](../eval/_audit_report.json)):

| Kriteria | Count | Reason |
|---|---|---|
| `judul.lower().startswith("notification ikut daftar tunggu")` | 34 | Placeholder waitlist, bukan listing real |
| `harga_per_bulan < 200_000` | 8 | Physically impossible price |
| `harga_per_bulan > 6_000_000` | 6 | Implausible price + basic facilities → likely scraping error |

Script: [backend/scripts/clean_corpus.py](../backend/scripts/clean_corpus.py).
Backup original: `data/raw/kozynear_combined.jsonl.bak`.
Audit trail: `data/raw/dropped_dirty_docs.jsonl`.

**Net effect**: 2122 → 2074 records (–48, –2.3%).

## Reproducibility

Untuk re-build canonical corpus dari scratch:

```bash
cd backend
# 1. Scrape (jangan run ulang kalau gak perlu; menghasilkan ID baru)
python -m scripts.scrape_mamikos_sitemap        # → mamikos_real.jsonl
python -m scripts.scrape_mamikos_playwright     # → mamikos_real_extra.jsonl

# 2. Dedup merge by judul
python -c "
import json
def load(p): return [json.loads(l) for l in open(p,encoding='utf-8')]
def dedup(rs):
    seen=set(); out=[]
    for r in rs:
        k=r['judul'].strip().lower()
        if k not in seen: seen.add(k); out.append(r)
    return out
m = dedup(load('../data/raw/mamikos_real.jsonl') + load('../data/raw/mamikos_real_extra.jsonl'))
with open('../data/raw/mamikos_real_merged.jsonl','w',encoding='utf-8') as f:
    for r in m: f.write(json.dumps(r,ensure_ascii=False)+'\n')
"

# 3. Build canonical (real-only) — lihat data/README.md untuk pipeline current
python -m scripts.rebuild_v2   # → kozynear_combined.jsonl

# 5. Clean (drop waitlist, low/high price outliers)
python -m scripts.clean_corpus

# 6. Preprocess + build indexes
python -m scripts.preprocess_corpus --input ../data/raw/kozynear_combined.jsonl --output ../data/processed/corpus.json
python -m app.indexing.build --corpus ../data/processed/corpus.json --output-dir ../data/indexes
```

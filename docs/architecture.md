# Architecture

> **Status dokumen (update 2026-06-10)**: ini design doc AWAL (three-tier,
> 3 service Render, neural via sentence-transformers). Arsitektur AKTUAL
> sudah berevolusi: single Docker container (frontend+API), model default
> **smart** (query parser + gazetteer geo + BM25 fusion), neural = MiniLM
> via fastembed di-gate `ENABLE_NEURAL` (ON di HF Spaces 16GB, OFF di
> Render 512MB), UI 4 tab, eval endpoint file-based. Referensi terkini:
> **LAPORAN.md §9** + `docs/specs/2026-06-01-smart-retrieval-design.md` +
> `docs/deploy_huggingface.md`. Dokumen di bawah dipertahankan sebagai
> jejak desain.

## Overview

TKI-KOS adalah three-tier IR application:

```
┌──────────────┐   HTTPS    ┌──────────────────┐   asyncpg   ┌─────────────────┐
│   React      │ ──────────►│  FastAPI Backend │ ──────────► │  PostgreSQL 16  │
│   (Vite SPA) │            │  (IR + REST API) │             │  (Managed DB)   │
└──────────────┘            └──────────────────┘             └─────────────────┘
   Render Static                  Render Web                       Render
   Site (free)                   Service (free)                   Managed PG
```

## Components

### Frontend — React + Vite + TypeScript

- SPA dengan pages: `SearchPage`, `ResultsPage`, `DetailPage`
- API client wrapper di `src/api/client.ts`
- Env: `VITE_API_URL` pointing ke backend (`https://<backend>.onrender.com`)
- Build: `npm run build` → `dist/` (static files)
- Deploy: Render Static Site (CDN, unlimited bandwidth, free)

### Backend — Python FastAPI + Uvicorn

REST API endpoints:
- `GET /search?q={query}&model={tfidf|bm25|indobert|hybrid}&top_k=10` — main search
- `GET /listings/{id}` — detail per listing
- `GET /health` — liveness probe untuk Render
- `GET /eval/summary` — eval metrics summary (opsional, Week 4)

IR engine:
- 3 baseline model + 1 optional hybrid
- Lifespan: load FAISS index + TF-IDF pickle + BM25 instance dari `data/indexes/` saat startup
- Embedding model di-cache di memori (RAM ~120MB untuk MiniLM, ~440MB untuk IndoBERT base)

Deploy: Render Web Service (free tier, 512MB RAM, spin down setelah 15 menit idle, cold start ~30s)

### Database — PostgreSQL 16

Tables (schema akan finalize di Week 2):

```sql
-- Raw + processed listing data
CREATE TABLE listings (
    id TEXT PRIMARY KEY,
    judul TEXT NOT NULL,
    deskripsi TEXT NOT NULL,
    deskripsi_processed TEXT,           -- after preprocessing pipeline
    harga_per_bulan INTEGER,
    tipe TEXT,                          -- putra/putri/campur (Mamikos gender 0/1/2)
    fasilitas TEXT[],                   -- PG array
    alamat TEXT,
    kecamatan TEXT,
    koordinat POINT,                    -- PG geometry
    jarak_kampus_km NUMERIC(5,2),
    url_source TEXT,
    scrape_date DATE,
    inserted_at TIMESTAMPTZ DEFAULT now()
);

-- Query set untuk eval
CREATE TABLE queries (
    id TEXT PRIMARY KEY,
    query_text TEXT NOT NULL,
    context TEXT,
    expected_tipe TEXT
);

-- Ground truth annotation
CREATE TABLE ground_truth (
    query_id TEXT REFERENCES queries(id),
    listing_id TEXT REFERENCES listings(id),
    annotator TEXT NOT NULL,
    relevance SMALLINT NOT NULL CHECK (relevance IN (0, 1, 2)),
    PRIMARY KEY (query_id, listing_id, annotator)
);

-- Consensus label (post-discussion)
CREATE TABLE ground_truth_consensus (
    query_id TEXT REFERENCES queries(id),
    listing_id TEXT REFERENCES listings(id),
    relevance SMALLINT NOT NULL,
    PRIMARY KEY (query_id, listing_id)
);

-- Per-model per-query metric results
CREATE TABLE eval_results (
    model TEXT NOT NULL,
    query_id TEXT REFERENCES queries(id),
    precision_at_5 NUMERIC(5,4),
    precision_at_10 NUMERIC(5,4),
    average_precision NUMERIC(5,4),
    ndcg_at_10 NUMERIC(5,4),
    reciprocal_rank NUMERIC(5,4),
    computed_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (model, query_id)
);
```

Deploy: Render Managed PostgreSQL (free tier — 1GB storage, expire 90 hari setelah create)

## Data Flow

1. **Scraping** (Week 1)
   `backend/app/scraper/mamikos.py` → JSON di `data/raw/listings.jsonl` → bulk insert ke `listings` table

2. **Preprocessing** (Week 2)
   `backend/app/preprocessing/pipeline.py` baca dari DB `listings.deskripsi` → run pipeline → write balik ke `listings.deskripsi_processed`

3. **Indexing** (Week 2)
   `backend/app/indexing/{tfidf,bm25,indobert}.py` baca `listings.deskripsi_processed` → build index → serialize ke `data/indexes/`

4. **Search runtime** (Week 2-3)
   User query → `app/search/query_parser.py` (parse filter, normalize) → call selected index → rank → return top-K + scores

5. **Evaluation** (Week 3-4)
   `backend/app/evaluation/run_eval.py` baca queries + ground_truth_consensus → run setiap model → compute metric → write ke `eval_results` + `eval/results.csv`

## IR Pipeline Detail

### Preprocessing Order (URUTAN PENTING)

```
raw text
  → HTML strip (BeautifulSoup)
  → whitespace normalize
  → price/jargon extraction (SEBELUM lowercase — preserve `Rp`, `AC`)
  → lowercase
  → custom jargon dict substitution (`gdg meneng` → `gedong meneng`)
  → spelling correction (kos-specific)
  → tokenize (whitespace + punctuation)
  → stopword removal (Sastrawi default + custom)
  → stem (Sastrawi StemmerFactory)
```

**Anti-pattern**: lowercase SEBELUM extract harga → regex bakal miss `Rp` capitalized.

### Model Comparison

| Model | Library | Hyperparameter Default | Strength | Weakness |
|-------|---------|------------------------|----------|----------|
| TF-IDF | scikit-learn | `ngram_range=(1,2), min_df=2, max_features=10000` | Fast, interpretable, lightweight | Lexical mismatch (synonym, paraphrase) |
| BM25 | rank_bm25 | `k1=1.5, b=0.75` | Term saturation handling, biasanya beat TF-IDF tipis | Still lexical, gak handle semantic |
| IndoBERT | sentence-transformers + FAISS | Mean-pool, dim 384/768, FAISS IndexFlatIP | Semantic matching, paraphrase-friendly | Heavy (~120MB-440MB), slower, kalah di exact-match seperti nama jalan |
| Hybrid | BM25 top-50 → IndoBERT rerank top-10 | — | Best of both worlds | Latency tertinggi, kompleks |

### Evaluation Metrics

| Metric | Definisi | Range | Target |
|--------|----------|-------|--------|
| Precision@K (K=5,10) | Proporsi dokumen relevant di top-K | [0,1] | >0.5 untuk best model |
| MAP | Mean Average Precision lintas queries | [0,1] | >0.4 untuk best model |
| NDCG@10 | Normalized Discounted Cumulative Gain | [0,1] | >0.5 untuk best model |
| MRR | Mean Reciprocal Rank (posisi first relevant) | [0,1] | >0.7 untuk best model |
| Cohen's Kappa | Inter-annotator agreement | [-1,1] | ≥0.7 (rubric requirement) |

Statistical test:
- **Paired t-test** kalau metric per-query roughly normal
- **Wilcoxon signed-rank** kalau distribusi skewed (likely untuk metric in [0,1])
- α = 0.05, report p-value + effect size

## Deployment Strategy (Render.com)

### Service Order

1. **PostgreSQL Managed Database** — create dulu, catat `DATABASE_URL` connection string
2. **FastAPI Backend Web Service**
   - Repo: `https://github.com/DYmazeh/TKI-KOS`
   - Root Directory: `backend`
   - Build: `pip install -r requirements.txt`
   - Start: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - Env: `DATABASE_URL` (link ke PG service), `CORS_ORIGINS`, `ENVIRONMENT=production`
3. **React Frontend Static Site**
   - Repo: same
   - Root Directory: `frontend`
   - Build: `npm install && npm run build`
   - Publish Dir: `dist`
   - Env: `VITE_API_URL=https://<backend-name>.onrender.com`

### Cold Start Mitigation

Free tier backend spin down setelah 15 menit idle, cold start ~30 detik. Kalau IndoBERT base di-load saat startup → +30-60 detik = total bisa 60-90s. **Bad untuk demo.**

Strategi:
1. **Pakai model ringan** — default `paraphrase-multilingual-MiniLM-L12-v2` (~118MB) instead of IndoBERT base (~440MB)
2. **Precompute FAISS index** ke disk, load saja saat lifespan startup (gak re-encode tiap startup)
3. **UptimeRobot ping** setiap 14 menit sebelum jadwal presentasi (free tier service tetap warm)

### PostgreSQL Free Tier Expiry

Free tier expire 90 hari setelah create. Setup target: akhir Mei → expire akhir Agustus. Aman untuk grading window 17 Juni + buffer.

## Security

- HTTPS auto via Render
- CORS whitelist: hanya frontend domain
- DB credentials via Render env vars (NEVER commit `.env`)
- Tidak ada user auth (sesuai scope brief — out of scope)
- Tidak ada PII (data publik dari Mamikos)

## Performance Notes

Corpus 227 real listing (footprint ringan, Render free tier 512MB OK):
- TF-IDF query: <5ms
- BM25 query: <10ms
- IndoBERT encode + FAISS search: 50-200ms (query encode dominasi)
- Hybrid: ~250ms

Acceptable untuk demo. Optimization optional kalau ada budget waktu di Week 4.

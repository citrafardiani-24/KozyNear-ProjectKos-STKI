# KozyNear &mdash; Indonesian Kos-Kosan Search Engine

> Final project Mata Kuliah **Temu Kembali Informasi** (COM620321, 3 SKS) &mdash; Universitas Lampung (UNILA)
> Search engine kos-kosan **Bandar Lampung raya** &mdash; 17 kecamatan kota + perbatasan ITERA, untuk mahasiswa 9 universitas (UNILA, ITERA, Darmajaya, UBL, UIN, Teknokrat, Polinela, Malahayati, Saburai)

🌐 **Live**: deploy via [HuggingFace Spaces](https://huggingface.co/spaces) (lihat [docs/deploy_huggingface.md](docs/deploy_huggingface.md))

![Status](https://img.shields.io/badge/status-deployed-green)
![Deadline](https://img.shields.io/badge/deadline-17%20Jun%202026-red)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.11-blue)
![React](https://img.shields.io/badge/react-18-61dafb)

## Overview

KozyNear adalah Information Retrieval system untuk listing kos-kosan di **seluruh Bandar Lampung** &mdash; berguna untuk mahasiswa dari **9 universitas** (UNILA, ITERA, Darmajaya, UBL, UIN Raden Intan, Teknokrat, Polinela, Malahayati, Saburai). User cari kos dengan natural language query bahasa Indonesia, dengan **5 model retrieval**:

1. **Lexical baseline** &mdash; TF-IDF + cosine similarity (scikit-learn)
2. **Probabilistic lexical** &mdash; BM25 (`rank_bm25`)
3. **Neural / semantic** &mdash; MiniLM multilingual sentence embeddings (via fastembed ONNX; label kode `indobert` historis) + FAISS
4. **Hybrid** &mdash; BM25 top-50 candidate &rarr; MiniLM rerank top-10
5. **Smart (default live)** &mdash; query understanding (parser gender/harga/fasilitas/anchor) + geo ranking (gazetteer kampus terverifikasi + haversine) + BM25 fusion; hard filter dengan relaxation jujur

Representasi dokumen memakai **fielded indexing** (judul x2 + kecamatan + fasilitas + deskripsi, BM25F-lite): 21% deskripsi pemilik < 10 kata sehingga field metadata membawa sinyal terkuat. UI punya 4 tab demo: **Pencarian**, **Evaluasi Model** (metrik live + signifikansi + effect size), **Preprocessing** (visualisasi 9-stage per langkah), **Statistik** (distribusi corpus).

## Tech Stack

| Layer | Tech |
|-------|------|
| Frontend | React 18 + Vite 5 + TypeScript |
| Backend | Python 3.11 + FastAPI + Uvicorn |
| Database | PostgreSQL 16 (asyncpg + SQLAlchemy 2.0) |
| Migrations | Alembic |
| IR Libraries | Sastrawi, rank_bm25, fastembed (ONNX MiniLM), faiss-cpu, scikit-learn |
| Eval | scipy (Wilcoxon + Holm-Bonferroni), custom Kappa + IR metrics + CS@K |
| Deploy | HF Spaces (16GB, neural ON) + Render.com backup; satu Dockerfile, gated `ENABLE_NEURAL` |

## Quick Start (Local Dev)

### Prerequisites

- Python 3.11+, Node 20+, Docker Desktop, Git

### Setup

```bash
# 1. Clone repo
git clone https://github.com/DYmazeh/KozyNear.git
cd KozyNear

# 2. Start local PostgreSQL via Docker
docker compose up -d

# 3. Backend setup
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate      # macOS/Linux
pip install -r requirements.txt
cp .env.example .env

# 4. Run migrations
alembic upgrade head

# 5. (Optional, kalau scrape data sudah ada) Seed DB + preprocess
python -m scripts.seed_db --input ../data/raw/mamikos.jsonl \
    --preprocess --corpus-output ../data/processed/corpus.json

# 6. (Optional, kalau corpus.json sudah ada) Build indexes
python -m app.indexing.build --corpus ../data/processed/corpus.json \
    --output-dir ../data/indexes

# 7. Run backend
uvicorn app.main:app --reload --port 8000
# Backend: http://localhost:8000 -- docs: http://localhost:8000/docs

# 8. Frontend (terminal baru)
cd frontend
npm install
cp .env.example .env
npm run dev
# Frontend: http://localhost:5173
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Root info |
| GET | `/health` | Liveness probe (Render uses this) |
| GET | `/api/search` | IR search dengan filter (`q`, `model`, `top_k`, `harga_min`, `harga_max`, `tipe`, `kecamatan`); filter berlaku untuk semua model termasuk smart |
| GET | `/api/listings/{id}` | Detail kos by ID |
| GET | `/api/eval/summary` | Aggregate metrics (standard + pool-restricted + CS@5 + Wilcoxon/Holm), file-based |
| GET | `/api/eval/query/{query_id}` | Per-query metric breakdown |
| GET | `/api/preprocess` | Trace pipeline 9-stage per langkah |
| GET | `/api/stats` | Statistik corpus live |
| GET | `/api/status` | Diagnostics (index loaded, RAM, DB) |
| GET | `/api/docs` | Swagger UI (auto) |

## Project Structure

```
TKI-KOS/
├── frontend/                # React Vite SPA (TypeScript)
│   ├── src/
│   │   ├── api/             # API client
│   │   ├── components/      # FilterPanel, ResultCard
│   │   └── App.tsx
│   └── package.json
├── backend/                 # Python FastAPI
│   ├── app/
│   │   ├── api/             # Routes: health, search, listings, eval
│   │   ├── core/            # config, db
│   │   ├── models/          # ORM + Pydantic schemas
│   │   ├── scraper/         # Mamikos + OLX + base + utils
│   │   ├── preprocessing/   # 9-stage pipeline + 108 jargon dict
│   │   ├── indexing/        # TF-IDF + BM25 + IndoBERT + Hybrid
│   │   ├── search/          # query handling
│   │   ├── evaluation/      # P@K, MAP, NDCG, Kappa, stat tests
│   │   └── main.py          # FastAPI entry + lifespan
│   ├── alembic/             # Database migrations
│   ├── scripts/             # seed_db.py
│   ├── tests/               # pytest suite
│   └── requirements.txt
├── data/
│   ├── raw/                 # Scraped JSONL (gitignored)
│   ├── processed/           # corpus.json setelah preprocessing
│   └── indexes/             # TF-IDF/BM25/FAISS serialized
├── notebooks/               # EDA, preprocessing experiment, eval
├── eval/                    # queries.json, ground_truth.csv, results.csv
├── docs/                    # architecture, deploy guide, GT protocol
├── docker-compose.yml       # Local PG
├── render.yaml              # Render infrastructure-as-code
└── README.md
```

## Roles & Team (5 anggota)

| Role | Bidang | Module |
|------|--------|--------|
| A &mdash; Lead / Scraper | Koordinasi tim + Mamikos scraper + data quality | `backend/app/scraper/` |
| B &mdash; Preprocessing | Custom jargon dictionary + pipeline tuning | `backend/app/preprocessing/` |
| C &mdash; IR Baseline | TF-IDF + BM25 hyperparameter tuning | `backend/app/indexing/` |
| D &mdash; IR Neural | IndoBERT + FAISS + hybrid tuning | `backend/app/indexing/` |
| E &mdash; Frontend + Eval | React UI + ground truth coord + metrics | `frontend/` + `backend/app/evaluation/` |

## Timeline (4-week sprint)

| Week | Dates | Milestone |
|------|-------|-----------|
| 1 | 20&ndash;26 Mei 2026 | Repo setup, Mamikos scraper, &ge;1500 listings |
| 2 | 27 Mei&ndash;2 Jun | Preprocessing pipeline, 3 IR indexes, FastAPI scaffold |
| 3 | 3&ndash;9 Jun | React UI live, ground truth annotation (Kappa &ge;0.7), deploy Render |
| 4 | 10&ndash;16 Jun | Evaluation, statistical tests, report finalization |
| **Submit** | **17 Jun 23:59 WIB** | Upload ke vClass UNILA |

## Documentation

| Topic | File |
|-------|------|
| Architecture (komponen, data flow, deploy strategy) | [docs/architecture.md](docs/architecture.md) |
| Deploy guide step-by-step (Render.com) | [docs/deploy_guide.md](docs/deploy_guide.md) |
| Ground truth annotation protocol (3-annotator + Kappa) | [docs/ground_truth_protocol.md](docs/ground_truth_protocol.md) |
| Scraper usage + TODOs | [backend/app/scraper/README.md](backend/app/scraper/README.md) |
| Preprocessing pipeline + jargon dict | [backend/app/preprocessing/README.md](backend/app/preprocessing/README.md) |
| IR indexes (TF-IDF/BM25/IndoBERT/Hybrid) | [backend/app/indexing/README.md](backend/app/indexing/README.md) |
| Evaluation metrics + Kappa + stat tests | [backend/app/evaluation/README.md](backend/app/evaluation/README.md) |
| Data schema + acquisition methodology | [data/README.md](data/README.md) |
| Notebooks convention | [notebooks/README.md](notebooks/README.md) |

## Course Rubric Coverage

| Aspect | Weight | Status |
|--------|--------|--------|
| Theme | 15% | Kos-kosan **Bandar Lampung raya**, multi-university (UNILA, ITERA, Darmajaya, UBL, UIN, Teknokrat, Polinela, Malahayati, Saburai sebagai target user; 8 anchor kampus terverifikasi di gazetteer) |
| Dataset | 15% | **227 listing REAL Mamikos = 88.7% populasi kost-bulanan Bandar Lampung** (Mamikos: "Tersedia 256 Kost", live 12 Jun) — ceiling efektif terverifikasi: uji re-extract 72 slug → 0 listing baru (gerbang validasi + audit trail); deskripsi pemilik + koordinat asli, PII di-strip |
| Preprocessing | 15% | 9-stage pipeline + 106 domain jargon entries (>=100 rubric req) + **ablation study per-stage** + audit coverage jargon + visualisasi live per-stage |
| Model | 20% | 5 model + **fielded indexing (BM25F-lite)** — Smart teratas standard P@5 0.773 / MAP 0.359 + CS@5 0.867 vs BM25 0.527 (p=0.0001, r=0.91, n=30); + eksperimen **Learning-to-Rank**; pooling-bias dibahas jujur |
| Evaluation | 10% | 30 query, 900 annotations; P@K/MAP/NDCG/MRR + 95% CI bootstrap + Cohen's Kappa + Wilcoxon + **Holm-Bonferroni** + effect size + CS@5 lima model + kit anotasi manusia |
| System | 25% | FastAPI + React 4-tab + Supabase PG (RLS) + Docker; HF Spaces (neural ON) + Render backup + CI test + rate limit + listing cache |

## Deploy URLs

- **Live App (primary)**: https://dymazeh-kozynear.hf.space (HF Spaces 16GB, `ENABLE_NEURAL=true`: kelima model live)
- **API Docs**: `<base-url>/api/docs` (Swagger UI)
- **Health**: `<base-url>/health`

## Live Results Summary (n=30 queries × 5 models, corpus real 227, index fielded)

**Standard top-K** (IR atas seluruh corpus):

| Model | P@5 | P@10 | MAP | NDCG@10 | MRR |
|-------|-----|------|-----|---------|-----|
| **Smart (live)** | **0.773** | **0.633** | **0.359** | **0.684** | 0.863 |
| BM25 | 0.640 | 0.610 | 0.296 | 0.634 | **0.872** |
| Hybrid (α=0.9) | 0.640 | 0.600 | 0.285 | 0.627 | 0.872 |
| TF-IDF | 0.613 | 0.547 | 0.253 | 0.569 | 0.786 |
| Neural MiniLM | 0.180 | 0.150 | 0.044 | 0.154 | 0.359 |

**Pool-restricted** (fair &mdash; Neural MAP melompat 0.044 → **0.627**, membuktikan skor rendah di atas adalah *pooling bias*, bukan model buruk; angka Smart di lensa ini optimistic karena sinyal filter overlap dengan heuristik annotator, lihat LAPORAN §7.1b):

| Model | P@5 | MAP |
|-------|-----|-----|
| Smart (live) | 0.940 | 0.893 |
| **BM25** | 0.640 | **0.667** |
| Hybrid (α=0.9) | 0.633 | 0.662 |
| TF-IDF | 0.660 | 0.656 |
| Neural MiniLM | 0.580 | 0.627 |

**Constraint Satisfaction @5** (lensa bebas qrels: % top-5 yang memenuhi SEMUA kebutuhan user &mdash; gender, budget, fasilitas, radius 3 km kampus):

| Model | mean CS@5 (n=30) |
|-------|------------------|
| **Smart (live)** | **0.867** |
| BM25 | 0.527 |
| TF-IDF | 0.527 |
| Hybrid (α=0.9) | 0.513 |
| Neural MiniLM | 0.233 |

Smart vs BM25 signifikan (Wilcoxon p=0.0001, effect size r=0.91). Signifikansi pairwise MAP memakai **Holm-Bonferroni** + effect size rank-biserial: 8/10 raw → **4/10 setelah Holm** (semuanya vs Neural); keunggulan Smart vs BM25 raw-significant (p=0.039, r=0.46) tapi belum lolos koreksi keluarga. Bonus eksperimen: **Learning-to-Rank** (logreg 9 fitur, 5-fold per query) MAP pool 0.926 vs fusion manual 0.907 (setara, p=0.14) dan **ablation preprocessing per-stage** (stopword +0.007; stemming ~0; jargon di sisi dokumen malah -0.024). Detail di [LAPORAN.md](LAPORAN.md) + [eval/eval_summary.md](eval/eval_summary.md).

## License

MIT &mdash; lihat [LICENSE](LICENSE)

# Indexing Module

Tiga paradigma IR + optional hybrid:

| Index | Library | Type | Strength | Weakness |
|-------|---------|------|----------|----------|
| `TFIDFIndex` | scikit-learn `TfidfVectorizer` + cosine | Lexical | Fast, interpretable | Synonym/paraphrase miss |
| `BM25Index` | `rank_bm25` `BM25Okapi` | Lexical (probabilistic) | Term saturation | Still lexical |
| `IndoBERTIndex` | `sentence-transformers` + FAISS | Neural / semantic | Semantic, paraphrase-aware | Heavy, slower |
| `HybridIndex` | BM25 candidates -> IndoBERT rerank | Composite | Best of both | Latency tertinggi |

## Workflow

```
data/raw/mamikos.jsonl              (output Anggota A scraper)
  -> preprocessing pipeline          (Anggota B)
  -> data/processed/corpus.json      (input ke build.py)
  -> python -m app.indexing.build    (Anggota C + D)
  -> data/indexes/
      tfidf.pkl
      bm25.pkl
      indobert/
        embeddings.npy
        faiss.index
        meta.json
```

## Quick Start

```bash
cd backend
.venv\Scripts\activate  # Windows

# 1. Pastikan corpus.json sudah ada
ls ../data/processed/corpus.json

# 2. Build semua indexes
python -m app.indexing.build \
    --corpus ../data/processed/corpus.json \
    --output-dir ../data/indexes

# 3. Smoke test load (di Python REPL atau notebook)
python -c "
from pathlib import Path
from app.indexing.loader import load_all_indexes
indexes = load_all_indexes(Path('../data/indexes'))
hits = indexes['bm25'].query('kos putra dekat unila', top_k=5)
for h in hits:
    print(h)
"
```

## Corpus Format

`data/processed/corpus.json` = list of objects:

```json
[
  {
    "id": "kos-abc-123",
    "text": "kos putra eksklusif gedong meneng air conditioner kamar mandi dalam ...",
    "raw_text": "Kos Putra Exclusive Gedong Meneng dengan AC dan KMD ...",
    "metadata": {
      "judul": "Kos Putra Exclusive Gedong Meneng",
      "harga_per_bulan": 850000,
      "tipe": "putra",
      "fasilitas": ["ac", "wifi", "kamar mandi dalam"],
      "alamat": "Jl. Sumantri Brojonegoro No. 1",
      "kecamatan": "Rajabasa"
    }
  },
  ...
]
```

- `text` = output preprocessing pipeline (lowercased, stemmed, stopword-free)
- `raw_text` = original deskripsi untuk display di UI
- `metadata` = field tambahan dari schema scraper

## What's Done (oleh mentor — scaffold)

- [x] `base.py` — `Document`, `SearchHit`, `IndexBase` abstract
- [x] `tfidf.py` — `TFIDFIndex` dengan default ngram=(1,2), min_df=2,
  max_features=10000, sublinear_tf
- [x] `bm25.py` — `BM25Index` dengan k1=1.5, b=0.75 (rank_bm25)
- [x] `indobert.py` — `IndoBERTIndex` dengan MiniLM multilingual default,
  FAISS IndexFlatIP, lazy model load, `score_docs()` untuk hybrid rerun
- [x] `hybrid.py` — `HybridIndex` (BM25 top-50 -> IndoBERT rerank, min-max
  normalize + weighted combination, configurable alpha)
- [x] `build.py` — CLI builder untuk semua indexes
- [x] `loader.py` — FastAPI lifespan helper
- [x] `tests/test_indexing.py` — unit test dengan small fixture corpus

## What Tim Anggota C (IR Baseline) WAJIB Do

### Priority 1 — Tune TF-IDF + BM25 hyperparameters

File: `notebooks/03_model_comparison.ipynb`.

Experiment:
- TF-IDF: `ngram_range` in [(1,1), (1,2), (1,3)], `min_df` in [1, 2, 5],
  `max_features` in [5000, 10000, 20000, None]
- BM25: `k1` in [1.0, 1.2, 1.5, 2.0], `b` in [0.5, 0.75, 1.0]

Plot heatmap MAP per kombinasi. Pilih best, dokumentasikan kenapa di laporan.

### Priority 2 — Edge case query testing

Setelah index ready, test queries:
- 1-word query: "kos" — apakah TF-IDF/BM25 robust?
- Stopword-heavy query: "yang ada di dekat kampus" — apakah preprocessing
  handle dengan baik?
- Out-of-vocabulary: query dengan term yang gak muncul di corpus

## What Tim Anggota D (IR Neural) WAJIB Do

### Priority 1 — Compare embedding models

Default: `paraphrase-multilingual-MiniLM-L12-v2` (~118MB).
Alternative: `indobenchmark/indobert-base-p2` (~440MB, lebih bagus tapi heavy).

Test both di sample queries, compare MAP. Trade-off: model size vs accuracy
vs Render free tier RAM (512MB).

### Priority 2 — Tune Hybrid alpha

File: `hybrid.py` di constructor `alpha` parameter.

Experiment alpha in [0.0, 0.3, 0.5, 0.7] di ground truth queries. Pure rerank
(alpha=0) sering kompetitif untuk Indonesian.

### Priority 3 — Pooling strategy comparison

`sentence-transformers` default mean-pool. Bisa explore:
- CLS token pooling
- Max pooling
- Concat (CLS + mean)

Pakai `model.encode(texts, output_value="token_embeddings")` untuk custom pool.

## Anti-Patterns

- [BAD] `FAISS IndexIVFFlat` untuk corpus <=5000 docs — overhead lebih besar
  dari benefit. Pakai `IndexFlatIP` (exhaustive) sampai 10K+ docs.
- [BAD] Re-encode docs di tiap hybrid query — slow. Pakai `score_docs()`
  yang reuse embeddings.
- [BAD] Pickle FAISS index langsung — fragile across versions. Pakai
  `faiss.write_index()` + `faiss.read_index()`.
- [BAD] Cache embeddings di memory tanpa save — restart loss. Selalu save
  ke disk.

## Performance Expectations

Corpus 3000 listings @ 384-dim embeddings:

| Operation | Time |
|-----------|------|
| TF-IDF build | 1-2s |
| BM25 build | <1s |
| IndoBERT encode all 3K docs (MiniLM, CPU) | 30-60s |
| IndoBERT encode all 3K docs (IndoBERT base, CPU) | 2-3 min |
| TF-IDF query | <5ms |
| BM25 query | <10ms |
| IndoBERT query (encode + FAISS Flat) | 50-100ms (mostly encode) |
| Hybrid query | 80-150ms |

Acceptable untuk demo. Optimization optional kalau ada budget waktu di Week 4.

## Testing

```bash
cd backend
pytest tests/test_indexing.py -v
```

Test cover:
- Each index: build, query, save, load roundtrip
- Edge cases: empty corpus, top_k > corpus_size
- Hybrid: combination math, alpha=0/0.5/1
- Performance smoke: 50-doc fixture builds < 5s

## Save Format Reference

| Index | File | Format |
|-------|------|--------|
| TF-IDF | `tfidf.pkl` | pickle: `{vectorizer, doc_matrix (sparse), doc_ids, size}` |
| BM25 | `bm25.pkl` | pickle: `{k1, b, doc_ids, tokenized_corpus, size}` (BM25Okapi rebuilt on load) |
| IndoBERT | `indobert/` folder | `embeddings.npy` + `faiss.index` + `meta.json` |
| Hybrid | _none_ | Composition only — load sub-indexes, instantiate HybridIndex |

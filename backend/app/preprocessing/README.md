# Preprocessing Module

Indonesian text preprocessing pipeline untuk kos listings, dengan custom jargon
dictionary (105+ entries).

## Pipeline Order (PENTING)

```
raw text
  -> 1. HTML strip (BeautifulSoup)
  -> 2. Whitespace normalize
  -> 3. Price extraction (SEBELUM lowercase — preserve `Rp` capitalized)
  -> 4. Lowercase
  -> 5. Jargon dictionary substitution (word-boundary, longest-first)
  -> 6. Spelling correction (typo dictionary)
  -> 7. Tokenize (whitespace + punctuation)
  -> 8. Stopword removal (Sastrawi default + custom domain)
  -> 9. Stem (Sastrawi StemmerFactory, LRU-cached)
```

**Anti-pattern**: lowercase SEBELUM extract harga → regex `[Rr][Pp]` masih
match, tapi edge case `Rp850k` (tanpa space) jadi rapuh kalau pattern berubah.

## Usage

```python
from app.preprocessing import PreprocessingPipeline, PipelineConfig

# Default: semua stage aktif
pipeline = PreprocessingPipeline()
result = pipeline.process(
    "Kos Putra AC WiFi <b>Rp 850.000</b>/bulan dekat unyila"
)

print(result.processed)         # joined tokens setelah stemming
print(result.tokens)            # ['putra', 'air', 'condition', ...]
print(result.extracted_prices)  # [850000]
print(result.stages_applied)    # ['strip_html', 'normalize_whitespace', ...]

# Shorthand utility
from app.preprocessing import preprocess
text = preprocess("kos murah 500k dekat unyila")

# Experiment mode: disable stem
config = PipelineConfig(stem=False)
pipeline = PreprocessingPipeline(config)
```

## What's Done (oleh mentor — scaffold)

- [x] `pipeline.py` — orchestrator dengan stage toggles
- [x] `normalizer.py` — HTML strip, whitespace, lowercase, inline price extract
- [x] `jargon.py` — **105+ entries** (abbreviation, location, type, rules,
  payment, kos form) — `python -m app.preprocessing.jargon` cek count
- [x] `spelling.py` — typo dictionary baseline (15+ fixes)
- [x] `tokenizer.py` — whitespace + punctuation tokenizer
- [x] `stopwords.py` — Sastrawi + custom kos stopwords
- [x] `stemmer.py` — Sastrawi wrapper dengan LRU cache 10k
- [x] Tests di `backend/tests/test_preprocessing.py`

## What Tim Anggota B (Preprocessing Engineer) WAJIB Do

### Priority 1 — Tambah jargon dict ke ≥120 entries

File: `jargon.py`. Saat ini 105 entries (sudah lewat threshold 100, tapi
**lebih bagus 120+ untuk dokumentasi rubric**).

Cara discover entries baru:
1. Tunggu Anggota A scrape selesai (`data/raw/mamikos.jsonl`)
2. Random sample 50 listings, baca manual:
   ```python
   import json
   with open("data/raw/mamikos.jsonl") as f:
       sample = [json.loads(line) for line in list(f)[:50]]
   for listing in sample:
       print(listing["deskripsi"][:300])
       print("---")
   ```
3. Catat pattern abbreviation / slang yang belum di `jargon.py`
4. Tambah ke kategori sesuai (`ABBREVIATIONS`, `LOCATIONS`, etc.)
5. Run check:
   ```bash
   cd backend
   python -m app.preprocessing.jargon
   # Output: KOS_JARGON_DICT size: 120+ -- OK: meets rubric requirement
   ```

### Priority 2 — Custom stopwords dari EDA

File: `stopwords.py` — `DEFAULT_CUSTOM_STOPWORDS`.

Di notebook `01_eda.ipynb`:
```python
from collections import Counter
import json

corpus = [json.loads(line) for line in open("../data/raw/mamikos.jsonl")]
all_tokens = []
for listing in corpus:
    all_tokens.extend(listing["deskripsi"].lower().split())

# Top-50 paling sering
for token, count in Counter(all_tokens).most_common(50):
    print(f"{token}: {count}")
```

Top-50 paling sering biasanya stopword domain — tambah ke
`DEFAULT_CUSTOM_STOPWORDS` kalau memang noise (bukan informative term).

### Priority 3 — Benchmark preprocessing impact

File: `notebooks/02_preprocessing_experiment.ipynb` (Anggota B buat sendiri).

Untuk **rubric Preprocessing 15%** — wajib show preprocessing impact secara
empirik:

| Variant | MAP | P@10 | Notes |
|---------|-----|------|-------|
| BASELINE (raw text, no preprocess) | 0.XX | 0.XX | reference |
| + HTML strip + lowercase           | 0.XX | 0.XX | |
| + jargon dict                      | 0.XX | 0.XX | expect lift |
| + spelling correction              | 0.XX | 0.XX | |
| + stopword removal                 | 0.XX | 0.XX | |
| + stemming                         | 0.XX | 0.XX | |
| FULL pipeline                      | 0.XX | 0.XX | |

Insight ditulis di laporan akhir + slide.

### Priority 4 — Spelling correction yang lebih robust (optional, Week 3)

Kalau ada budget waktu di Week 2-3, upgrade `spelling.py`:
- Edit distance (Levenshtein) untuk catch unknown typos
- Bigram language model untuk context-aware correction

Libraries: `pyspellchecker`, custom n-gram model.

## Testing

```bash
cd backend
pip install -r requirements.txt  # include Sastrawi
pytest tests/test_preprocessing.py -v
```

Test coverage:
- Atomic functions (normalizer, tokenizer, jargon, spelling) — no Sastrawi needed
- Pipeline integration (stopword, stemmer) — needs Sastrawi installed
- Anti-pattern test: price extraction sebelum lowercase
- Jargon substitution dengan word boundary + longest-first ordering
- Config toggles (disable individual stages)

## Performance Notes

Untuk corpus 3000 listings × ~150 tokens avg:
- HTML strip: ~5ms/doc
- Jargon substitution: ~2ms/doc (pre-compiled regex)
- Sastrawi stem (first time): ~500ms-1s setup, ~10-50ms/doc
- Sastrawi stem (cached): ~1-5ms/doc

Total batch preprocessing 3000 docs: ~30-60 detik (acceptable, run once,
save corpus.json).

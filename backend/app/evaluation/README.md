# Evaluation Module

IR evaluation metrics, inter-annotator agreement (Cohen's Kappa), dan
statistical significance tests untuk rubric Evaluation 10%.

## Metrics

| Metric | Definisi | Range | Best |
|--------|----------|-------|------|
| P@K (K=5,10) | proportion of top-K predicted yang relevant | [0,1] | higher |
| MAP | Mean Average Precision across queries | [0,1] | higher |
| NDCG@10 | Normalized Discounted Cumulative Gain (graded relevance) | [0,1] | higher |
| MRR | Mean Reciprocal Rank (1/rank of first relevant) | [0,1] | higher |

## Inter-Annotator Agreement

- **`cohen_kappa`**: untuk 2 annotator, nominal labels
- **`weighted_kappa`**: untuk ordinal labels (0/1/2 relevance) — direkomendasikan

**Target rubric**: Kappa >= 0.7 across all annotator pairs. Kalau di bawah,
**dokumentasikan + lakuin consensus resolution** sebelum compute IR metrics.

Interpretation (Landis & Koch 1977):
- < 0.20: slight | 0.21-0.40: fair | 0.41-0.60: moderate
- 0.61-0.80: substantial | 0.81-1.00: almost perfect

## Statistical Significance

Untuk compare 2 model (e.g., BM25 vs IndoBERT) berdasar per-query metric:

- **`paired_ttest`**: kalau distribusi roughly normal (lebih powerful)
- **`wilcoxon_signed_rank`**: non-parametric, **direkomendasikan** untuk IR
  (metric in [0,1] sering skewed)

H0: "Model A == Model B". alpha = 0.05. p < alpha => significant difference.

## Workflow Penuh

### 1. Annotation Phase (Week 3)

Persiapan:
1. Tim Anggota E koordinasi 3 annotator (anggota tim, bukan AI)
2. Pre-annotation calibration: discuss 3-5 sample queries bareng, agree on
   what "relevant" / "somewhat relevant" / "not relevant" means
3. Distribute task: tiap annotator label SEMUA top-30 hits dari BM25 untuk
   setiap query (300+ judgments per annotator, ~12 queries x 30 docs)

Output: `eval/annotations_annotator_{A,B,C}.csv`
```csv
query_id,doc_id,relevance
q01,d12345,2
q01,d12346,1
...
```

### 2. Compute Kappa + Consensus

```python
import csv
from app.evaluation import cohen_kappa, weighted_kappa, interpret_kappa

# Load annotations
def load(path):
    with open(path) as f:
        return {(row["query_id"], row["doc_id"]): int(row["relevance"])
                for row in csv.DictReader(f)}

ann_a = load("eval/annotations_annotator_A.csv")
ann_b = load("eval/annotations_annotator_B.csv")
ann_c = load("eval/annotations_annotator_C.csv")

# Align by shared (query, doc) keys
shared = set(ann_a.keys()) & set(ann_b.keys()) & set(ann_c.keys())

for pair_name, (pa, pb) in [("A-B", (ann_a, ann_b)), ("A-C", (ann_a, ann_c)),
                             ("B-C", (ann_b, ann_c))]:
    a_vals = [pa[k] for k in shared]
    b_vals = [pb[k] for k in shared]
    k = weighted_kappa(a_vals, b_vals)
    print(f"{pair_name}: kappa={k:.3f} ({interpret_kappa(k)})")
```

Target: semua pair >= 0.7. Kalau ada yang <0.7:
1. Identify queries dengan disagreement tertinggi
2. Discussion session: 3 annotator bareng, agree pada labeling
3. Re-annotate problematic queries
4. Recompute kappa
5. Generate consensus: majority vote, atau (kalau 2-2 split) tertinggi/median

Output: `eval/ground_truth.csv` (consensus labels)
```csv
query_id,doc_id,relevance
q01,d12345,2
...
```

### 3. Run Evaluation (Week 4)

```bash
cd backend
python -m app.evaluation.runner \
    --queries ../eval/queries.json \
    --ground-truth ../eval/ground_truth.csv \
    --indexes-dir ../data/indexes \
    --output ../eval/results.csv
```

Output:
- `eval/results.csv`: per-model per-query metrics
- Console: aggregate MAP / P@5 / P@10 / NDCG@10 / MRR per model + pairwise
  Wilcoxon significance

### 4. Generate Report (Notebook)

File: `notebooks/04_evaluation.ipynb` (Anggota E buat).

Wajib content untuk laporan:
- Table aggregate metrics (4 model x 5 metrics)
- Per-query analysis: queries mana yang hard untuk each model + kenapa
- Statistical significance table (pairwise Wilcoxon p-values)
- Visualizations:
  - Bar chart MAP per model
  - Scatter plot: query difficulty (avg performance) vs query length/complexity
  - Heatmap: model x query (cell = MAP per query)
- Interpretation: kenapa Model X menang/kalah di query Y

## Files in Module

| File | Purpose |
|------|---------|
| `metrics.py` | P@K, MAP, NDCG@K, MRR |
| `kappa.py` | Cohen's & Weighted Kappa untuk inter-annotator |
| `statistical.py` | Paired t-test, Wilcoxon signed-rank |
| `runner.py` | CLI: run all-models x all-queries, output CSV |
| `README.md` | this file |

## What Tim Anggota E (Frontend + Eval Lead) WAJIB Do

### Priority 1 — Coordinate annotation (Week 3 start)

- Kick off 3-annotator process minggu pertama Week 3
- Use BM25 top-30 sebagai pool kandidat (cepat, lexical baseline)
- Provide annotators dengan: query text, listing snippet (judul + 200 char
  deskripsi), full deskripsi link
- Calibration session sebelum mulai (1 jam, walk through 5 sample queries)

### Priority 2 — Build evaluation notebook (Week 4)

File: `notebooks/04_evaluation.ipynb`. Template structure di README di atas.

### Priority 3 — Build /eval dashboard (optional, kalau ada budget waktu)

FastAPI route `/eval/summary` yang return aggregate metrics + per-query
breakdown sebagai JSON. Frontend tab di React app untuk dosen lihat eval
langsung di web (bonus poin System 25%).

## Anti-Patterns

- [BAD] Compute MAP tanpa Kappa check dulu — kalau Kappa < 0.7, ground truth
  unreliable, MAP angka gak meaningful
- [BAD] Annotator labels berbeda interpretation "relevant" — selalu kalibrasi
  dulu
- [BAD] Cuma 1 annotator — gak ada agreement check, gak reproducible
- [BAD] Ground truth dari AI/dosen tanpa human annotator independen
- [BAD] Report MAP without significance test — perbedaan 0.05 bisa noise

## Testing

```bash
cd backend
pytest tests/test_evaluation.py -v
```

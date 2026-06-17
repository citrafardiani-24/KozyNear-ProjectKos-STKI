# Ground Truth Annotation Protocol

Protokol untuk membuat ground truth labels yang reliable, sesuai rubric
**Evaluation 10%** (course requirement: >=12 queries x 3 annotator,
Cohen's Kappa >= 0.7).

## Scope

| Item | Target |
|------|--------|
| Queries | >= 12 dengan context diverse (lihat `eval/queries.json`) |
| Annotators | 3 (anggota tim TKI-KOS, bukan AI) |
| Documents per query | Top-30 dari BM25 baseline (sufficient untuk MAP@10 stable) |
| Total annotations | >= 12 x 3 x 30 = 1,080 judgments |
| Relevance scale | 0 / 1 / 2 (3-point ordinal) |
| Inter-annotator target | Kappa >= 0.7 (substantial agreement) |

## Relevance Scale

| Score | Label | Definisi |
|-------|-------|----------|
| **2** | Sangat relevan | Document **sangat** cocok untuk query — mencakup semua atau hampir semua constraint user |
| **1** | Sebagian relevan | Document mencakup sebagian constraint, atau secara umum kategori kos cocok tapi gak match detail |
| **0** | Tidak relevan | Document tidak cocok constraint utama (e.g., query kos putra, listing kos putri) |

### Contoh: query `"kos putra dekat unila ada wifi dan ac"`

- **Score 2**: Listing kos putra di Gedong Meneng, fasilitas AC + WiFi, dekat kampus UNILA
- **Score 1**: Listing kos putra dekat UNILA tapi tanpa info WiFi/AC eksplisit (atau hanya salah satu)
- **Score 0**: Listing kos putri (tipe mismatch), atau kos campur di luar Bandar Lampung

## Pre-Annotation Calibration (WAJIB, 1 jam)

Tujuan: pastikan 3 annotator interpret "relevant" yang sama.

1. Anggota E (Eval Lead) pilih 3 sample query dari `queries.json`
2. Tampilkan 5 sample documents per query (dari BM25 top hits)
3. Annotator label independen, lalu bandingkan
4. Discussion: untuk doc dengan disagreement, agree on guideline:
   - Berapa toleransi missing facility — kalau query minta "ac+wifi" tapi listing cuma punya ac, score 1 atau 0?
   - Bagaimana handle ambiguous location (e.g., query "dekat unila" tapi listing 5km dari kampus)?
   - Document partial vs whole match
5. Dokumentasikan hasil calibration di `eval/calibration_notes.md`

## Annotation Format

Tiap annotator save ke `eval/annotations_annotator_<NAME>.csv`:

```csv
query_id,doc_id,relevance,notes
q01,kos-abc-12345,2,kos putra wifi ac gedong meneng - perfect match
q01,kos-def-67890,1,kos putra dekat unila tapi gak ada info AC
q01,kos-xyz-54321,0,tipe putri - mismatch
q02,kos-...,2,...
```

`notes` optional tapi recommended untuk doc dengan score borderline (membantu
consensus discussion).

## Annotation Workflow

### Step 1: Generate candidate pool

Setelah scraping + indexing done:

```python
# scripts/generate_annotation_pool.py
import json
from pathlib import Path
from app.indexing.loader import load_all_indexes
from app.preprocessing import PreprocessingPipeline

indexes = load_all_indexes(Path("data/indexes"))
bm25 = indexes["bm25"]
pipeline = PreprocessingPipeline()

with open("eval/queries.json") as f:
    queries = json.load(f)["queries"]

pool = {}
for q in queries:
    qid = q["id"]
    q_text = q["query"]
    processed = pipeline.process(q_text).processed
    hits = bm25.query(processed, top_k=30)
    pool[qid] = [h.doc_id for h in hits]

with open("eval/annotation_pool.json", "w") as f:
    json.dump(pool, f, indent=2)
```

Output `eval/annotation_pool.json`:
```json
{
  "q01": ["kos-abc-12345", "kos-def-67890", ...],
  "q02": [...],
  ...
}
```

### Step 2: Distribute task

Anggota E generate UI sederhana untuk annotator (atau pakai Google Sheet):

| Query | Doc ID | Judul | Snippet | Score (0/1/2) | Notes |
|-------|--------|-------|---------|---------------|-------|
| q01: kos putra dekat unila ada wifi dan ac | kos-abc-12345 | Kos Putra Eksklusif | "Kos putra mewah di Gedong Meneng..." | _ | _ |
| ... |

Annotator isi kolom Score independen (tanpa lihat hasil annotator lain).

### Step 3: Compute Kappa

```python
# scripts/compute_kappa.py
import csv
from app.evaluation import weighted_kappa, interpret_kappa

annotations = {}
for name in ["A", "B", "C"]:
    with open(f"eval/annotations_annotator_{name}.csv") as f:
        annotations[name] = {
            (row["query_id"], row["doc_id"]): int(row["relevance"])
            for row in csv.DictReader(f)
        }

shared_keys = set(annotations["A"].keys()) & set(annotations["B"].keys()) & set(annotations["C"].keys())

for a, b in [("A", "B"), ("A", "C"), ("B", "C")]:
    a_vals = [annotations[a][k] for k in shared_keys]
    b_vals = [annotations[b][k] for k in shared_keys]
    kappa = weighted_kappa(a_vals, b_vals, labels=[0, 1, 2])
    print(f"{a} vs {b}: weighted kappa = {kappa:.3f} ({interpret_kappa(kappa)})")
```

Target output: semua pair >= 0.7. Kalau **ada pair < 0.7**:

1. Identify queries dengan disagreement tertinggi:
   ```python
   for qid in queries:
       max_diff = max(annotator scores for this qid) - min(...)
       if max_diff >= 2: # disagree by 2 levels (e.g., 0 vs 2)
           flag
   ```
2. Discussion session (30 min): 3 annotator bareng, agree pada labeling
3. Re-annotate problematic queries
4. Recompute kappa
5. Iterate sampai >= 0.7

### Step 4: Generate consensus

```python
# scripts/generate_consensus.py
import csv
from collections import Counter

consensus = {}
for key in shared_keys:
    scores = [annotations[name][key] for name in ["A", "B", "C"]]
    # Majority vote
    most_common = Counter(scores).most_common(1)[0][0]
    consensus[key] = most_common

with open("eval/ground_truth.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["query_id", "doc_id", "relevance"])
    for (qid, did), rel in sorted(consensus.items()):
        writer.writerow([qid, did, rel])
```

Untuk **2-2 split** (one annotator removed, evenly disagreement):
- Default: ambil median atau tertinggi (conservative)
- Atau: flag untuk discussion lagi

## Common Pitfalls

- [BAD] Annotator diskusi sambil label — defeats independence (biased Kappa)
- [BAD] Cuma 1-2 annotator — gak ada way to check Kappa, gak reproducible
- [BAD] AI annotation tanpa human verification — gak count untuk rubric
- [BAD] Calibration di-skip — Kappa hampir pasti rendah, butuh re-do
- [BAD] Annotate berbeda kriteria per query — bikin inconsistent labels

## Time Budget

| Task | Estimasi |
|------|----------|
| Calibration session | 1 jam |
| Annotation 360 judgments per annotator @ 10 detik/judgment | 1 jam |
| Compute Kappa + initial check | 30 menit |
| Disagreement discussion (kalau perlu) | 30-60 menit |
| Re-annotation problematic queries | 30 menit |
| Generate consensus + ground_truth.csv | 15 menit |
| **Total per round** | **3-4 jam** |

Plan untuk 1-2 round di Week 3 (5-9 Juni).

## Output Files

```
eval/
├── queries.json                  # input: 12+ queries
├── annotation_pool.json          # input: top-30 per query (dari BM25)
├── annotations_annotator_A.csv   # output annotator A
├── annotations_annotator_B.csv   # output annotator B
├── annotations_annotator_C.csv   # output annotator C
├── calibration_notes.md          # documentation calibration agreement
├── kappa_report.md               # documentation per-pair Kappa + interpretation
└── ground_truth.csv              # final consensus (input ke runner.py)
```

`ground_truth.csv` adalah satu-satunya yang dipakai `app.evaluation.runner` —
yang lain untuk dokumentasi/reproducibility/auditing.

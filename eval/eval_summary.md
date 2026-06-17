# Evaluation Summary Report

Auto-generated dari `eval/results.csv` (output `app.evaluation.runner`).
Untuk laporan akhir, copy table di sini + narasi insight.

## Setup

- Total queries: 30
- Corpus: 227 listing REAL Mamikos (lihat `data/README.md`)
- Ground truth: AI-assisted 3-annotator simulation, di-pool dari BM25 (lihat `eval/kappa_report.md`)
- Eval: standard top-K + pool-restricted (`results_pool_restricted.csv`) — lihat pooling bias di LAPORAN §8.2
- Top-K cutoff: 10

## Aggregate Metrics per Model

MAP disertai 95% CI (percentile bootstrap atas query, 10k resample).

| Model | P@5 | P@10 | MAP | MAP 95% CI | NDCG@10 | MRR |
|-------|-----|------|-----|------------|---------|-----|
| bm25 | 0.6400 | 0.6100 | 0.2958 | [0.242, 0.350] | 0.6341 | 0.8722 |
| hybrid | 0.6400 | 0.6000 | 0.2846 | [0.235, 0.336] | 0.6265 | 0.8722 |
| indobert | 0.1800 | 0.1500 | 0.0437 | [0.027, 0.063] | 0.1542 | 0.3594 |
| smart | 0.7733 | 0.6333 | 0.3594 | [0.288, 0.430] | 0.6835 | 0.8633 |
| tfidf | 0.6133 | 0.5467 | 0.2532 | [0.206, 0.302] | 0.5688 | 0.7861 |

**Ranking by MAP**: smart (0.359) > bm25 (0.296) > hybrid (0.285) > tfidf (0.253) > indobert (0.044)

## Pairwise Statistical Significance (Wilcoxon signed-rank, MAP)

H0: Model A == Model B. alpha = 0.05. Karena ada banyak pasangan diuji
sekaligus, signifikansi final memakai koreksi **Holm-Bonferroni**
(kontrol family-wise error rate); kolom raw disertakan untuk transparansi.

| Pair | Statistic | p-value | p-Holm | r (rank-biserial) | n | Sig (raw) | Sig (Holm) |
|------|-----------|---------|--------|-------------------|---|-----------|------------|
| tfidf vs bm25 | 87.00 | 0.0246 | 0.1230 | -0.504 | 30 | yes | no |
| tfidf vs indobert | 10.00 | 0.0000 | 0.0000 | 0.957 | 30 | yes | **YES** |
| tfidf vs hybrid | 127.00 | 0.1363 | 0.1708 | -0.328 | 30 | no | no |
| tfidf vs smart | 90.00 | 0.0101 | 0.0605 | -0.557 | 30 | yes | no |
| bm25 vs indobert | 3.00 | 0.0000 | 0.0000 | 0.987 | 30 | yes | **YES** |
| bm25 vs hybrid | 46.00 | 0.0854 | 0.1708 | 0.462 | 30 | no | no |
| bm25 vs smart | 103.00 | 0.0388 | 0.1530 | -0.455 | 30 | yes | no |
| indobert vs hybrid | 3.00 | 0.0000 | 0.0000 | -0.987 | 30 | yes | **YES** |
| indobert vs smart | 14.00 | 0.0000 | 0.0000 | -0.940 | 30 | yes | **YES** |
| hybrid vs smart | 112.00 | 0.0382 | 0.1530 | -0.448 | 30 | yes | no |

**8/10** pasangan signifikan tanpa koreksi; setelah Holm-Bonferroni **4/10**. Dengan n=30 query, uji non-signifikan dibaca *inconclusive*, bukan bukti dua model setara.

## Per-Query Analysis (AP per Model)

| Query | tfidf | bm25 | indobert | hybrid | smart |
|-------|---|---|---|---|---|
| `q01` *kos putra dekat unila wifi dan ac* | 0.271 | 0.516 | 0.042 | 0.513 | 0.711 |
| `q02` *kos putri murah kedaton dekat kampus* | 0.166 | 0.265 | 0.111 | 0.276 | 0.260 |
| `q03` *kos campur kamar mandi dalam parkir motor* | 0.255 | 0.527 | 0.129 | 0.447 | 0.036 |
| `q04` *kos dekat itera sukarame kamar mandi dalam* | 0.401 | 0.526 | 0.000 | 0.526 | 0.377 |
| `q05` *kos dekat teknokrat kedaton ada ac* | 0.350 | 0.137 | 0.022 | 0.139 | 0.275 |
| `q06` *kos murah dekat polinela rajabasa mahasiswa* | 0.417 | 0.253 | 0.136 | 0.253 | 0.375 |
| `q07` *kos eksklusif ac wifi kamar mandi dalam shower* | 0.258 | 0.269 | 0.022 | 0.279 | 0.385 |
| `q08` *kos putri aman bersih ada cctv* | 0.328 | 0.403 | 0.028 | 0.403 | 0.500 |
| `q09` *kos ada dapur bisa masak parkir mobil* | 0.270 | 0.270 | 0.111 | 0.270 | 0.127 |
| `q10` *kos dekat transmart way halim nyaman* | 0.600 | 0.583 | 0.200 | 0.572 | 0.433 |
| `q11` *kos putra murah rajabasa dekat unila* | 0.037 | 0.217 | 0.075 | 0.227 | 0.526 |
| `q12` *kos putri dekat unila wifi murah* | 0.256 | 0.435 | 0.022 | 0.387 | 0.435 |
| `q13` *kos dekat uin raden intan sukarame* | 0.338 | 0.455 | 0.011 | 0.455 | 0.346 |
| `q14` *kos putri way halim ac wifi* | 0.068 | 0.057 | 0.000 | 0.057 | 0.178 |
| `q15` *kos campur nyaman strategis dekat kampus* | 0.400 | 0.411 | 0.000 | 0.342 | 0.688 |
| `q16` *kos putra ac kamar mandi dalam rajabasa* | 0.366 | 0.440 | 0.000 | 0.341 | 0.610 |
| `q17` *kos putri dekat darmajaya labuhan ratu* | 0.310 | 0.311 | 0.036 | 0.311 | 0.312 |
| `q18` *kos dekat ubl wifi parkir motor* | 0.064 | 0.184 | 0.073 | 0.188 | 0.261 |
| `q19` *kos campur tanjung senang murah* | 0.334 | 0.313 | 0.000 | 0.313 | 0.381 |
| `q20` *kos putri kipas angin murah sukarame* | 0.285 | 0.344 | 0.005 | 0.338 | 0.455 |
| `q21` *kos putra wifi dapur bersama labuhan ratu* | 0.000 | 0.000 | 0.028 | 0.000 | 0.000 |
| `q22` *kos dekat mall boemi kedaton ac* | 0.206 | 0.206 | 0.013 | 0.211 | 0.394 |
| `q23` *kos putri lemari kasur way halim* | 0.124 | 0.141 | 0.033 | 0.141 | 0.333 |
| `q24` *kos campur ac wifi murah* | 0.310 | 0.274 | 0.027 | 0.269 | 0.011 |
| `q25` *kos campur tv wifi kedaton* | 0.258 | 0.290 | 0.112 | 0.249 | 0.471 |
| `q26` *kos putra labuhan ratu kamar mandi dalam* | 0.000 | 0.000 | 0.050 | 0.000 | 0.000 |
| `q27` *kos putri aman dekat itera* | 0.127 | 0.251 | 0.000 | 0.251 | 0.607 |
| `q28` *kos putri dekat teknokrat wifi* | 0.271 | 0.176 | 0.024 | 0.217 | 0.114 |
| `q29` *kos campur parkir mobil sukarame* | 0.214 | 0.227 | 0.000 | 0.186 | 0.625 |
| `q30` *kos putri bersih nyaman dekat uin sukarame* | 0.312 | 0.390 | 0.000 | 0.376 | 0.555 |

## Constraint Satisfaction @5 (lensa kedua — kebutuhan user)

CS@5 = proporsi top-5 yang memenuhi SEMUA constraint query (gender + harga + fasilitas + radius 3 km dari anchor). Metric ini mengukur apa yang dioptimalkan smart pipeline dan TIDAK bergantung qrels (bebas pooling bias).

| Model | mean CS@5 (n=30) |
|-------|------------|
| **smart** | **0.8667** |
| bm25 | 0.5267 |
| tfidf | 0.5267 |
| hybrid | 0.5133 |
| indobert | 0.2333 |

Per-query detail: `eval/results_constraints.csv`.

## Insights & Discussion (template untuk laporan)

1. **BM25 best lexical baseline**: menang di queries lexical-heavy (exact match nama universitas/fasilitas di deskripsi pemilik). Selisih vs TF-IDF tidak signifikan.

2. **IndoBERT (MiniLM) = pooling bias, bukan model buruk**: skor rendah di standard top-K karena GT di-pool dari kandidat lexical (hasil semantic unik tidak ter-judge). Pada pool-restricted eval dia kompetitif. Lihat `results_pool_restricted.csv` + LAPORAN §8.2.

3. **Smart (model live)**: smart = BM25 + query understanding + geo + hard filter. Unggul di MAP/P@5/MRR standard dan dominan di CS@5 (selisih vs BM25 signifikan); selisih MAP standard vs BM25 belum signifikan. P@10 smart bisa di bawah BM25: hard filter memangkas kandidat dan dokumen hasil geo-augment yang belum ter-annotate dihitung rel=0 (pooling bias juga menekan smart).

4. **Multiple comparison**: signifikansi final pakai Holm-Bonferroni (lihat tabel). Semua pasangan yang tetap signifikan melibatkan indobert standard top-K — konsisten dengan cerita pooling bias.

5. **Sample size**: n=30 query (dinaikkan dari 15); uji non-signifikan = inconclusive. Hyperparameter dipilih di data eval yang sama (disclosed sebagai mild selection bias; future: held-out set).

6. **Data real terse**: deskripsi pemilik median ~23 kata menurunkan absolute metric tapi otentik. Future: multi-model pooling untuk hilangkan pooling bias. Lihat `data/README.md`.

## Files

- `eval/queries.json`: 30 query set (v1.1 real-data)
- `eval/ground_truth.csv`: 900 consensus annotations
- `eval/kappa_report.md`: inter-annotator agreement
- `eval/results.csv`: per-model per-query (standard top-K, termasuk smart)
- `eval/results_pool_restricted.csv`: pool-restricted (fair) eval
- `eval/results_constraints.csv`: Constraint Satisfaction @5 (smart vs bm25)
- `eval/significance_map.csv`: Wilcoxon semua pasangan + Holm
- `eval/eval_summary.md`: ini (aggregate + significance + analysis)
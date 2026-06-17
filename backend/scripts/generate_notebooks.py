"""Generate 4 Jupyter notebooks untuk laporan analysis.

Output:
- notebooks/01_eda.ipynb           — corpus statistics + distribusi
- notebooks/02_preprocessing.ipynb — pipeline impact benchmark
- notebooks/03_model_comparison.ipynb — qualitative per-query analysis
- notebooks/04_evaluation.ipynb    — final viz (bar charts + heatmap)

Usage:
    cd backend
    python -m scripts.generate_notebooks
"""

from __future__ import annotations

import json
from pathlib import Path


def md(*sources: str) -> dict:
    """Markdown cell — accepts variadic lines (auto-newline join)."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": "\n".join(sources).splitlines(keepends=True),
    }


def code(*sources: str) -> dict:
    """Code cell — empty outputs (run jupyter to populate)."""
    return {
        "cell_type": "code",
        "metadata": {},
        "source": "\n".join(sources).splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
    }


def make_nb(cells: list[dict], title: str = "Notebook") -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
            "title": title,
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def save(nb: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"[wrote] {path}")


# =============================================================================
# Notebook 1: EDA
# =============================================================================
def nb_01_eda() -> dict:
    return make_nb([
        md(
            "# 01 — Exploratory Data Analysis (EDA)",
            "",
            "**Corpus**: 227 listing kos REAL hasil scrape Mamikos.com (deskripsi pemilik + koordinat asli), 18 kecamatan Bandar Lampung.",
            "",
            "**Tujuan**: pahami distribusi data sebelum preprocessing + indexing untuk informasi Dataset 15% rubric.",
        ),
        md("## Setup"),
        code(
            "import json",
            "from collections import Counter",
            "from pathlib import Path",
            "",
            "import pandas as pd",
            "import matplotlib.pyplot as plt",
            "",
            "plt.style.use('seaborn-v0_8-whitegrid')",
            "# Sumber real committed (raw listing fields). Untuk corpus terindeks: data/processed/corpus.json",
            "DATA_PATH = Path('../data/raw/mamikos_real_v2.jsonl')",
        ),
        md("## Load Corpus"),
        code(
            "with open(DATA_PATH, encoding='utf-8') as f:",
            "    data = [json.loads(line) for line in f if line.strip()]",
            "",
            "df = pd.DataFrame(data)",
            "print(f'Total listings: {len(df)}')",
            "df.head(3)",
        ),
        md("## 1. Word Count Distribution (deskripsi pemilik asli — terse)"),
        code(
            "df['deskripsi'] = df['deskripsi'].fillna('')",
            "df['word_count'] = df['deskripsi'].str.split().str.len()",
            "print(df['word_count'].describe())",
            "",
            "fig, ax = plt.subplots(figsize=(8, 4))",
            "ax.hist(df['word_count'], bins=25, color='#2563eb', edgecolor='black')",
            "ax.axvline(df['word_count'].median(), color='red', linestyle='--', label=f'Median: {int(df[\"word_count\"].median())} kata')",
            "ax.set_xlabel('Word count per deskripsi (cerita pemilik)')",
            "ax.set_ylabel('Listings')",
            "ax.set_title(f'Distribusi panjang deskripsi real (n={len(df)})')",
            "ax.legend()",
            "plt.tight_layout()",
            "plt.savefig('charts/01_word_count.png', dpi=100, bbox_inches='tight')",
            "plt.show()",
        ),
        md("## 2. Price Distribution"),
        code(
            "print(df['harga_per_bulan'].describe())",
            "",
            "fig, ax = plt.subplots(figsize=(10, 4))",
            "ax.hist(df['harga_per_bulan'], bins=30, color='#047857', edgecolor='black')",
            "ax.axvline(df['harga_per_bulan'].median(), color='red', linestyle='--', label=f'Median: Rp {int(df[\"harga_per_bulan\"].median()):,}')",
            "ax.set_xlabel('Harga per bulan (IDR)')",
            "ax.set_ylabel('Listings')",
            "ax.set_title('Distribusi harga sewa kos real (Rp 300k - 6jt)')",
            "ax.legend()",
            "ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x/1000)}k'))",
            "plt.tight_layout()",
            "plt.savefig('charts/01_price.png', dpi=100, bbox_inches='tight')",
            "plt.show()",
        ),
        md("## 3. Tipe Distribution"),
        code(
            "tipe_counts = df['tipe'].value_counts()",
            "print(tipe_counts)",
            "",
            "fig, ax = plt.subplots(figsize=(7, 4))",
            "colors = {'putra': '#2563eb', 'putri': '#ec4899', 'campur': '#8b5cf6'}",
            "tipe_counts.plot(kind='bar', ax=ax, color=[colors.get(t, '#999') for t in tipe_counts.index])",
            "ax.set_title('Distribusi tipe kos real (campur/putri/putra)')",
            "ax.set_ylabel('Count')",
            "ax.set_xlabel('Tipe')",
            "plt.xticks(rotation=0)",
            "plt.tight_layout()",
            "plt.savefig('charts/01_tipe.png', dpi=100, bbox_inches='tight')",
            "plt.show()",
        ),
        md("## 4. Kecamatan Coverage (18 kecamatan dengan data real)"),
        code(
            "kec_counts = df['kecamatan'].value_counts()",
            "print(f'Total kecamatan unique: {len(kec_counts)}')",
            "",
            "fig, ax = plt.subplots(figsize=(8, 8))",
            "kec_counts.sort_values().plot(kind='barh', ax=ax, color='#0891b2')",
            "ax.set_title('Distribusi listings per kecamatan Bandar Lampung')",
            "ax.set_xlabel('Count')",
            "plt.tight_layout()",
            "plt.savefig('charts/01_kecamatan.png', dpi=100, bbox_inches='tight')",
            "plt.show()",
        ),
        md("## 5. Fasilitas Frequency"),
        code(
            "fasilitas_flat = [f for fs in df['fasilitas'] for f in fs]",
            "fac_counts = Counter(fasilitas_flat)",
            "print('Top 10 fasilitas:')",
            "for fac, cnt in fac_counts.most_common(10):",
            "    print(f'  {fac}: {cnt}')",
            "",
            "top_fac = pd.DataFrame(fac_counts.most_common(15), columns=['fasilitas', 'count'])",
            "fig, ax = plt.subplots(figsize=(8, 6))",
            "ax.barh(top_fac['fasilitas'][::-1], top_fac['count'][::-1], color='#7c3aed')",
            "ax.set_title('Top 15 fasilitas')",
            "ax.set_xlabel('Listings yang punya fasilitas ini')",
            "plt.tight_layout()",
            "plt.savefig('charts/01_fasilitas.png', dpi=100, bbox_inches='tight')",
            "plt.show()",
        ),
        md("## 6. Universitas Mentions (Multi-University Coverage)"),
        code(
            "UNIV_ALIASES = {",
            "    'UNILA': ['UNILA', 'Universitas Lampung', 'unyila', 'Unila'],",
            "    'Polinela': ['Polinela', 'Politeknik Negeri Lampung'],",
            "    'IBI Darmajaya': ['IBI Darmajaya', 'Darmajaya'],",
            "    'UBL': ['UBL', 'Universitas Bandar Lampung'],",
            "    'UIN Raden Intan': ['UIN Raden Intan', 'UIN RIL', 'UIN Lampung'],",
            "    'Teknokrat': ['Teknokrat', 'Universitas Teknokrat'],",
            "    'Malahayati': ['Malahayati', 'Universitas Malahayati'],",
            "    'ITERA': ['ITERA', 'Institut Teknologi Sumatera'],",
            "    'Saburai': ['Saburai', 'Universitas Saburai'],",
            "}",
            "",
            "univ_counts = {}",
            "for univ, aliases in UNIV_ALIASES.items():",
            "    count = df['deskripsi'].apply(lambda d: any(a in d for a in aliases)).sum()",
            "    univ_counts[univ] = count",
            "",
            "univ_df = pd.Series(univ_counts).sort_values(ascending=True)",
            "fig, ax = plt.subplots(figsize=(8, 5))",
            "univ_df.plot(kind='barh', ax=ax, color='#dc2626')",
            "ax.set_title('Mentions universitas di deskripsi listing')",
            "ax.set_xlabel('Count listings')",
            "plt.tight_layout()",
            "plt.savefig('charts/01_universities.png', dpi=100, bbox_inches='tight')",
            "plt.show()",
        ),
        md(
            "## Insights (data real)",
            "",
            "1. **18 kecamatan** ter-cover; dominasi Sukarame/Kedaton/Rajabasa karena density kos Mamikos tertinggi di sekitar UNILA/UIN/ITERA/Polinela — mencerminkan pasar nyata.",
            "2. **Harga real**: median ~Rp 800k, range Rp 300k - 6jt.",
            "3. **Tipe**: campur 50%, putri 37%, putra 13% (tidak ada pasutri — Mamikos gender 0/1/2 saja).",
            "4. **Deskripsi terse**: median ~23 kata, cerita pemilik asli (bukan padded). Lebih pendek dari synthetic tapi otentik.",
            "5. **Universitas mentions**: UNILA/Teknokrat/Polinela/UIN paling sering disebut pemilik karena memang area kos terpadat.",
            "",
            "Detail di [LAPORAN.md](../LAPORAN.md) section 3.",
        ),
    ], title="01 EDA")


# =============================================================================
# Notebook 2: Preprocessing Experiment
# =============================================================================
def nb_02_preprocessing() -> dict:
    return make_nb([
        md(
            "# 02 — Preprocessing Pipeline Impact",
            "",
            "**Tujuan**: benchmark IR metric (MAP via BM25) untuk setiap variant preprocessing.",
            "Untuk rubric Preprocessing 15% — wajib show empirical impact.",
        ),
        md("## Setup"),
        code(
            "import sys",
            "from pathlib import Path",
            "",
            "sys.path.insert(0, '../backend')",
            "",
            "import json",
            "import pandas as pd",
            "from rank_bm25 import BM25Okapi",
            "",
            "from app.preprocessing import PipelineConfig, PreprocessingPipeline",
            "from app.evaluation import average_precision, mean_average_precision",
        ),
        md("## Load Data"),
        code(
            "with open('../data/raw/mamikos_real_v2.jsonl', encoding='utf-8') as f:",
            "    raw_listings = [json.loads(line) for line in f if line.strip()]",
            "raw_listings = [r for r in raw_listings if (r.get('deskripsi') or '').strip()]",
            "print(f'Loaded {len(raw_listings)} real listings')",
            "",
            "with open('../eval/queries.json', encoding='utf-8') as f:",
            "    queries = json.load(f)['queries']",
            "",
            "with open('../eval/ground_truth.csv') as f:",
            "    import csv",
            "    gt_rows = list(csv.DictReader(f))",
            "",
            "# Build relevant set per query (relevance >= 1)",
            "relevant_per_query = {}",
            "for row in gt_rows:",
            "    qid = row['query_id']",
            "    if int(row['relevance']) >= 1:",
            "        relevant_per_query.setdefault(qid, set()).add(row['doc_id'])",
        ),
        md("## Benchmark Function"),
        code(
            "def benchmark_config(config: PipelineConfig, label: str) -> dict:",
            "    \"\"\"Run pipeline, build BM25, compute MAP.\"\"\"",
            "    pipeline = PreprocessingPipeline(config)",
            "    ",
            "    # Preprocess corpus",
            "    docs = [pipeline.process(l['deskripsi']).processed for l in raw_listings]",
            "    doc_ids = [l['id'] for l in raw_listings]",
            "    ",
            "    # Build BM25",
            "    tokenized = [d.split() for d in docs]",
            "    bm25 = BM25Okapi(tokenized, k1=1.5, b=0.75)",
            "    ",
            "    # Run queries",
            "    predicted_per_query = {}",
            "    for q in queries:",
            "        q_processed = pipeline.process(q['query']).processed",
            "        scores = bm25.get_scores(q_processed.split())",
            "        # Top-10",
            "        import numpy as np",
            "        top_idx = np.argsort(-scores)[:10]",
            "        predicted_per_query[q['id']] = [doc_ids[i] for i in top_idx]",
            "    ",
            "    map_score = mean_average_precision(predicted_per_query, relevant_per_query)",
            "    return {'config': label, 'map': map_score}",
        ),
        md(
            "## Run Benchmark per Variant",
            "",
            "Toggle individual stages off untuk lihat impact masing-masing.",
            "",
            "WARNING: tiap variant butuh ~30 detik (preprocessing + BM25 build).",
        ),
        code(
            "variants = [",
            "    ('BASELINE (no preprocessing)', PipelineConfig(",
            "        strip_html=False, normalize_whitespace=False, extract_prices=False,",
            "        lowercase=False, apply_jargon_dict=False, correct_spelling=False,",
            "        tokenize=False, remove_stopwords=False, stem=False,",
            "    )),",
            "    ('+ lowercase only', PipelineConfig(",
            "        strip_html=True, normalize_whitespace=True, extract_prices=False,",
            "        lowercase=True, apply_jargon_dict=False, correct_spelling=False,",
            "        tokenize=True, remove_stopwords=False, stem=False,",
            "    )),",
            "    ('+ jargon dict (108 entries)', PipelineConfig(",
            "        strip_html=True, normalize_whitespace=True, extract_prices=False,",
            "        lowercase=True, apply_jargon_dict=True, correct_spelling=False,",
            "        tokenize=True, remove_stopwords=False, stem=False,",
            "    )),",
            "    ('+ spelling correction', PipelineConfig(",
            "        strip_html=True, normalize_whitespace=True, extract_prices=False,",
            "        lowercase=True, apply_jargon_dict=True, correct_spelling=True,",
            "        tokenize=True, remove_stopwords=False, stem=False,",
            "    )),",
            "    ('+ stopword removal', PipelineConfig(",
            "        strip_html=True, normalize_whitespace=True, extract_prices=False,",
            "        lowercase=True, apply_jargon_dict=True, correct_spelling=True,",
            "        tokenize=True, remove_stopwords=True, stem=False,",
            "    )),",
            "    ('FULL pipeline (+ stem)', PipelineConfig()),",
            "]",
            "",
            "results = []",
            "for label, cfg in variants:",
            "    print(f'Running: {label}')",
            "    results.append(benchmark_config(cfg, label))",
            "",
            "results_df = pd.DataFrame(results)",
            "results_df",
        ),
        md("## Visualize Impact"),
        code(
            "import matplotlib.pyplot as plt",
            "plt.style.use('seaborn-v0_8-whitegrid')",
            "",
            "fig, ax = plt.subplots(figsize=(10, 5))",
            "ax.barh(results_df['config'], results_df['map'], color='#2563eb')",
            "ax.set_xlabel('MAP (BM25, top-10)')",
            "ax.set_title('Preprocessing Pipeline Stage Impact pada BM25 MAP')",
            "for i, v in enumerate(results_df['map']):",
            "    ax.text(v + 0.005, i, f'{v:.4f}', va='center')",
            "plt.tight_layout()",
            "plt.savefig('charts/02_preprocessing_impact.png', dpi=100, bbox_inches='tight')",
            "plt.show()",
        ),
        md(
            "## Discussion (template untuk laporan)",
            "",
            "1. **BASELINE** (raw text) sebagai reference",
            "2. **+ lowercase** memberikan boost karena query case insensitive",
            "3. **+ jargon dict** memberikan boost paling besar — handle abbreviation domain (kmd, ac, gdg meneng)",
            "4. **+ spelling correction** lebih berdampak di data real (typo asli pemilik: 'kreta', 'belakanga')",
            "5. **+ stopword** boost kecil — kos/kamar terlalu sering muncul",
            "6. **+ stem** kompromise — beberapa kata jadi terlalu pendek (gedong neng) tapi consolidate morphology",
            "",
            "**Anti-pattern surface**: lowercase sebelum extract harga akan break `Rp 850.000` matching, walaupun regex case-insensitive masih handle.",
        ),
    ], title="02 Preprocessing")


# =============================================================================
# Notebook 3: Model Comparison (Qualitative)
# =============================================================================
def nb_03_model_comparison() -> dict:
    return make_nb([
        md(
            "# 03 — Model Comparison (Qualitative)",
            "",
            "**Tujuan**: bandingkan output top-5 dari 4 model (TF-IDF, BM25, IndoBERT, Hybrid) untuk sample queries.",
            "Untuk rubric Model 20% — wajib show qualitative per-query analysis.",
        ),
        md("## Setup"),
        code(
            "import sys",
            "import json",
            "from pathlib import Path",
            "",
            "sys.path.insert(0, '../backend')",
            "",
            "from app.indexing.bm25 import BM25Index",
            "from app.indexing.tfidf import TFIDFIndex",
            "from app.indexing.indobert import IndoBERTIndex",
            "from app.indexing.hybrid import HybridIndex",
            "from app.preprocessing import PreprocessingPipeline",
        ),
        md("## Load Indexes + Corpus"),
        code(
            "INDEXES_DIR = Path('../data/indexes')",
            "tfidf = TFIDFIndex.load(INDEXES_DIR / 'tfidf.pkl')",
            "bm25 = BM25Index.load(INDEXES_DIR / 'bm25.pkl')",
            "indobert = IndoBERTIndex.load(INDEXES_DIR / 'indobert')",
            "hybrid = HybridIndex(bm25, indobert, bm25_top_k=50, alpha=0.3)",
            "",
            "pipeline = PreprocessingPipeline()",
            "",
            "with open('../data/processed/corpus.json', encoding='utf-8') as f:",
            "    corpus = {item['id']: item for item in json.load(f)}",
            "",
            "print(f'Loaded 4 indexes, {len(corpus)} docs in corpus')",
        ),
        md("## Helper: Show Top-5 Side-by-Side"),
        code(
            "def show_query(q: str, top_k: int = 5):",
            "    processed = pipeline.process(q).processed",
            "    print(f'Query:     {q!r}')",
            "    print(f'Processed: {processed!r}')",
            "    print('=' * 80)",
            "    ",
            "    for index, name in [(tfidf, 'TF-IDF'), (bm25, 'BM25'), (indobert, 'IndoBERT'), (hybrid, 'Hybrid')]:",
            "        hits = index.query(processed, top_k=top_k)",
            "        print(f'\\n[{name}]')",
            "        for h in hits:",
            "            meta = corpus[h.doc_id]['metadata']",
            "            print(f'  rank{h.rank} score={h.score:.4f}')",
            "            print(f'    {meta[\"judul\"]}')",
            "            print(f'    Rp {meta[\"harga_per_bulan\"]:,} / {meta[\"tipe\"]} / {meta[\"kecamatan\"]}')",
        ),
        md("## Query Analysis"),
        md("### Query 1: Multi-Constraint UNILA"),
        code("show_query('kos putra dekat unila wifi dan ac')"),
        md("### Query 2: Multi-University ITERA"),
        code("show_query('kos dekat itera sukarame kamar mandi dalam')"),
        md("### Query 3: Fasilitas + Tipe"),
        code("show_query('kos campur kamar mandi dalam parkir motor')"),
        md("### Query 4: Premium / Eksklusif"),
        code("show_query('kos eksklusif ac wifi kamar mandi dalam shower')"),
        md("### Query 5: Kualitas / Keamanan"),
        code("show_query('kos putri aman bersih ada cctv')"),
        md(
            "## Insights per Model",
            "",
            "- **BM25** menang di queries dengan exact-match nama kampus / fasilitas yang muncul di deskripsi",
            "- **IndoBERT** kuat di pool-restricted (semantic mengelompokkan kos sejenis), tertekan di standard karena GT lexical-pooled",
            "- **Hybrid** #2 di pool-restricted (MRR tertinggi) — kombinasi bernilai saat eval fair",
            "- **TF-IDF** kompetitif di corpus kecil 227 dengan term jarang (selisih vs BM25 tidak signifikan)",
        ),
    ], title="03 Model Comparison")


# =============================================================================
# Notebook 4: Evaluation Visualization
# =============================================================================
def nb_04_evaluation() -> dict:
    return make_nb([
        md(
            "# 04 — Evaluation Final Visualization",
            "",
            "**Source**: `eval/results.csv` (output dari `app.evaluation.runner`)",
            "",
            "**Untuk laporan akhir**: charts di sini bisa di-save dan di-embed ke PDF report.",
        ),
        md("## Setup"),
        code(
            "import pandas as pd",
            "import matplotlib.pyplot as plt",
            "import numpy as np",
            "",
            "plt.style.use('seaborn-v0_8-whitegrid')",
        ),
        md("## Load Results"),
        code(
            "df = pd.read_csv('../eval/results.csv')",
            "print(f'Total rows: {len(df)}')",
            "print(f'Models: {df[\"model\"].unique().tolist()}')",
            "print(f'Queries: {df[\"query_id\"].nunique()}')",
            "df.head()",
        ),
        md("## Aggregate Metrics per Model"),
        code(
            "agg = df.groupby('model').agg(",
            "    p_at_5=('p_at_5', 'mean'),",
            "    p_at_10=('p_at_10', 'mean'),",
            "    map=('ap', 'mean'),",
            "    ndcg_at_10=('ndcg_at_10', 'mean'),",
            "    mrr=('rr', 'mean'),",
            ").sort_values('map', ascending=False)",
            "agg",
        ),
        md("## Chart 1: Bar Chart MAP per Model"),
        code(
            "model_order = agg.index.tolist()",
            "colors = {'bm25': '#2563eb', 'hybrid': '#7c3aed', 'tfidf': '#0891b2', 'indobert': '#dc2626'}",
            "",
            "fig, ax = plt.subplots(figsize=(8, 4))",
            "bars = ax.bar(model_order, agg['map'], color=[colors.get(m, '#888') for m in model_order])",
            "ax.set_ylabel('MAP')",
            "ax.set_title(f'Mean Average Precision per IR Model (n={df[\"query_id\"].nunique()} queries, corpus real)')",
            "ax.set_ylim(0, max(agg['map']) * 1.15)",
            "for bar, val in zip(bars, agg['map']):",
            "    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005, f'{val:.4f}',",
            "            ha='center', fontweight='bold')",
            "plt.tight_layout()",
            "plt.savefig('charts/04_map_bars.png', dpi=100, bbox_inches='tight')",
            "plt.show()",
        ),
        md("## Chart 2: All Metrics Comparison"),
        code(
            "fig, ax = plt.subplots(figsize=(10, 5))",
            "metrics = ['p_at_5', 'p_at_10', 'map', 'ndcg_at_10', 'mrr']",
            "x = np.arange(len(metrics))",
            "width = 0.2",
            "",
            "for i, model in enumerate(model_order):",
            "    vals = [agg.loc[model, m] for m in metrics]",
            "    ax.bar(x + i * width, vals, width, label=model, color=colors.get(model, '#888'))",
            "",
            "ax.set_xticks(x + width * 1.5)",
            "ax.set_xticklabels(['P@5', 'P@10', 'MAP', 'NDCG@10', 'MRR'])",
            "ax.set_ylabel('Score')",
            "ax.set_title('All Metrics — 4 IR Models Compared')",
            "ax.legend()",
            "plt.tight_layout()",
            "plt.savefig('charts/04_all_metrics.png', dpi=100, bbox_inches='tight')",
            "plt.show()",
        ),
        md("## Chart 3: Per-Query Heatmap (AP)"),
        code(
            "pivot = df.pivot(index='query_id', columns='model', values='ap')",
            "pivot = pivot[model_order]  # reorder columns",
            "",
            "fig, ax = plt.subplots(figsize=(8, 8))",
            "im = ax.imshow(pivot.values, aspect='auto', cmap='YlGnBu', vmin=0, vmax=1)",
            "ax.set_xticks(range(len(model_order)))",
            "ax.set_xticklabels(model_order)",
            "ax.set_yticks(range(len(pivot.index)))",
            "ax.set_yticklabels(pivot.index)",
            "ax.set_title('Average Precision per Query × Model')",
            "for i in range(len(pivot.index)):",
            "    for j in range(len(model_order)):",
            "        val = pivot.values[i, j]",
            "        color = 'white' if val > 0.5 else 'black'",
            "        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=8)",
            "fig.colorbar(im, ax=ax, label='AP')",
            "plt.tight_layout()",
            "plt.savefig('charts/04_heatmap.png', dpi=100, bbox_inches='tight')",
            "plt.show()",
        ),
        md("## Chart 4: Query Difficulty (Mean AP across all models)"),
        code(
            "query_difficulty = df.groupby('query_id')['ap'].mean().sort_values()",
            "query_text = df.groupby('query_id')['query'].first()",
            "",
            "fig, ax = plt.subplots(figsize=(10, 6))",
            "colors_difficulty = ['#dc2626' if v < 0.1 else '#f59e0b' if v < 0.3 else '#10b981' for v in query_difficulty]",
            "ax.barh(query_difficulty.index, query_difficulty.values, color=colors_difficulty)",
            "ax.set_xlabel('Mean AP across all 4 models')",
            "ax.set_title('Query Difficulty (sorted ascending)')",
            "ax.axvline(0.1, color='red', linestyle='--', alpha=0.5, label='Hard (<0.1)')",
            "ax.axvline(0.3, color='orange', linestyle='--', alpha=0.5, label='Medium (<0.3)')",
            "ax.legend()",
            "plt.tight_layout()",
            "plt.savefig('charts/04_difficulty.png', dpi=100, bbox_inches='tight')",
            "plt.show()",
            "",
            "print('\\nHARD queries (mean AP < 0.1):')",
            "for qid in query_difficulty[query_difficulty < 0.1].index:",
            "    print(f'  {qid}: {query_text[qid]!r}')",
        ),
        md("## Statistical Significance Summary"),
        code(
            "from scipy import stats",
            "",
            "print('Pairwise Wilcoxon signed-rank on AP per query (alpha=0.05):')",
            "print('=' * 70)",
            "models = list(model_order)",
            "for i, ma in enumerate(models):",
            "    for mb in models[i+1:]:",
            "        a_aps = df[df['model'] == ma].sort_values('query_id')['ap'].values",
            "        b_aps = df[df['model'] == mb].sort_values('query_id')['ap'].values",
            "        try:",
            "            stat, p = stats.wilcoxon(a_aps, b_aps)",
            "            sig = 'SIGNIFICANT' if p < 0.05 else 'not sig'",
            "            print(f'  {ma:10s} vs {mb:10s} | stat={stat:6.2f} | p={p:.4f} | {sig}')",
            "        except Exception as e:",
            "            print(f'  {ma} vs {mb}: ERROR {e}')",
        ),
        md(
            "## Final Insights (corpus real 227, 15 queries)",
            "",
            "1. **BM25 winner** (MAP 0.319, P@5 0.613) — 4/6 pairs SIGNIFICANT (signifikan vs IndoBERT & Hybrid; selisih vs TF-IDF tidak signifikan).",
            "2. **IndoBERT bukan buruk — pooling bias**: MAP standard 0.053 → pool-restricted 0.527. GT di-pool dari BM25 sehingga hasil semantic unik tidak ter-judge. Bandingkan `results.csv` (standard) vs `results_pool_restricted.csv`.",
            "3. **Hybrid #2 di pool-restricted** (MAP 0.594, MRR 0.783) — kombinasi bernilai saat eval fair.",
            "4. **Data real terse**: deskripsi pemilik median ~23 kata menurunkan absolute metric tapi otentik.",
            "5. **Sample n=15**: cukup untuk Wilcoxon (4/6 significant); 30+ ideal untuk future work.",
            "",
            "> **Penting**: untuk evaluasi fair model non-lexical, lihat juga pool-restricted (`scripts/eval_pool_restricted.py`).",
        ),
    ], title="04 Evaluation")


def main() -> int:
    out_dir = Path(__file__).resolve().parent.parent.parent / "notebooks"
    charts_dir = out_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    notebooks = [
        ("01_eda.ipynb", nb_01_eda()),
        ("02_preprocessing.ipynb", nb_02_preprocessing()),
        ("03_model_comparison.ipynb", nb_03_model_comparison()),
        ("04_evaluation.ipynb", nb_04_evaluation()),
    ]

    for name, nb in notebooks:
        save(nb, out_dir / name)

    print(f"\n[done] {len(notebooks)} notebooks -> {out_dir}")
    return 0


if __name__ == "__main__":
    main()

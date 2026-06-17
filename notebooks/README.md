# Notebooks

Jupyter notebook untuk analisis & experiment yang dipresentasikan di laporan akhir.

| Notebook | Week | Content |
|----------|------|---------|
| `01_eda.ipynb` | 1 | Exploratory Data Analysis: distribusi harga, panjang deskripsi, kecamatan, tipe kos, fasilitas frequency |
| `02_preprocessing_experiment.ipynb` | 2 | Benchmark metric BEFORE vs AFTER tiap step preprocessing (HTML strip, jargon dict, Sastrawi stemming, stopword) |
| `03_model_comparison.ipynb` | 2-3 | Side-by-side run TF-IDF / BM25 / IndoBERT / Hybrid di sample queries, qualitative analysis |
| `04_evaluation.ipynb` | 4 | Full eval: metric per model per query, statistical significance test, visualizations (bar chart, scatter difficulty), insights untuk laporan |

## Setup

```bash
# Dari root project
cd backend
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Mac/Linux

# Install jupyter
pip install jupyter ipykernel
python -m ipykernel install --user --name=tki-kos --display-name="TKI-KOS"

# Run
cd ../notebooks
jupyter notebook
```

Pilih kernel `TKI-KOS` di Jupyter untuk auto-pakai venv.

## Convention

- Tiap notebook standalone — bisa di-run dari awal tanpa dependency notebook lain
- Output disimpan di notebook (komit dengan output untuk reproducibility/laporan)
- Pakai relative path: `../data/processed/corpus.json`
- Markdown narrative wajib di tiap section — ini bagian dari course assessment

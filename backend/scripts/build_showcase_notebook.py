"""Bangun notebook showcase model (kode + teks + output nyata) lalu eksekusi
+ render ke HTML untuk ditampilkan di tab "Notebook" aplikasi.

Output:
- notebooks/05_model_showcase.ipynb  (executed; bisa dibuka di HF Files tab)
- frontend/public/notebook.html       (render Jupyter, di-embed React tab)

Notebook self-contained: tiap eksekusi nyatanya menyentuh index + eval CSV
yang sudah ada (cepat & deterministik). Jalankan ulang tiap update model:
    cd backend && python -m scripts.build_showcase_notebook
"""
from __future__ import annotations
import sys
from pathlib import Path

import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
NB_OUT = ROOT / "notebooks" / "05_model_showcase.ipynb"
HTML_OUT = ROOT / "frontend" / "public" / "notebook.html"


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(src: str):
    return nbf.v4.new_code_cell(src)


SETUP = f'''
import sys, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(r"{ROOT.as_posix()}")
sys.path.insert(0, str(ROOT / "backend"))
import pandas as pd
pd.set_option("display.max_colwidth", 46)

from app.indexing.loader import load_all_indexes
from app.indexing.hybrid import HybridIndex
from app.preprocessing import PreprocessingPipeline
from app.search.gazetteer import Gazetteer
from app.search.pipeline import smart_rank
from scripts.eval_smart import load_listings

idx = load_all_indexes(ROOT / "data" / "indexes", include_neural=True)
bm25, tfidf, neural = idx["bm25"], idx["tfidf"], idx["indobert"]
pipe = PreprocessingPipeline()
pre = lambda s: pipe.process(s).processed
hybrid = HybridIndex(bm25, neural, query_preprocessor=pre)
gz = Gazetteer.load()
listings = load_listings()
print(f"Corpus: {{len(listings)}} listing | vocab BM25: {{len(bm25.bm25.idf)}} term")
print(f"Index siap: {{', '.join(idx.keys())}} + smart + hybrid")
'''.strip()

PREP = '''
# Pipeline preprocessing 9-stage pada contoh teks (judul + deskripsi pemilik)
contoh = "Kost Putri AC KM Dlm dkt UNILA 800rb/bln, wifi kenceng"
res = pipe.process(contoh, trace=True)
print("INPUT :", res.raw)
for s in res.trace:
    out = s["output"] if isinstance(s["output"], str) else " | ".join(map(str, s["output"]))
    print(f"  {s['stage']:<20} -> {out[:70]}")
print("HASIL :", res.processed)
print("Harga terdeteksi:", res.extracted_prices)
'''.strip()

QUERY = '''
# Bandingkan lima model pada satu query natural language
q = "kos putri dekat unila wifi murah"

def top(model, n=5):
    if model == "bm25":   ids = [(h.doc_id, h.score) for h in bm25.query(pre(q), top_k=n)]
    elif model == "tfidf":ids = [(h.doc_id, h.score) for h in tfidf.query(pre(q), top_k=n)]
    elif model == "neural":ids = [(h.doc_id, h.score) for h in neural.query(q, top_k=n)]
    elif model == "hybrid":ids = [(h.doc_id, h.score) for h in hybrid.query(q, top_k=n)]
    else: ids = smart_rank(q, bm25, listings, gz, top_k=n, preprocess=pre)[0]
    return [listings[i].judul for i, _ in ids]

pd.DataFrame({m: top(m) for m in ["bm25","tfidf","neural","hybrid","smart"]})
'''.strip()

SMART = '''
# Smart pipeline: query understanding + geo + fusion (model live)
res, understood, relaxed = (lambda r: (r[0], r[1], r[2]))(
    smart_rank(q, bm25, listings, gz, top_k=5, preprocess=pre))
print("Yang dipahami sistem dari query:")
for k, v in understood.items():
    if v: print(f"  {k:<10}: {v}")
print("\\nTop-5 smart:")
for did, score in res:
    r = listings[did]
    print(f"  [{r.tipe:<6} Rp{r.harga_per_bulan:>8} {(r.kecamatan or '-'):<14}] {r.judul[:44]}")
'''.strip()

EVAL = '''
# Hasil evaluasi 30 query (dibaca dari CSV hasil eksperimen)
df = pd.read_csv(ROOT / "eval" / "results.csv")
agg = (df.groupby("model")[["p_at_5","p_at_10","ap","ndcg_at_10","rr"]]
         .mean().round(3).sort_values("ap", ascending=False))
agg.columns = ["P@5","P@10","MAP","NDCG@10","MRR"]
agg
'''.strip()

CHART = '''
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 3.4))
order = agg.sort_values("MAP")
colors = ["#2563eb" if m == "smart" else "#94a3b8" for m in order.index]
ax.barh(order.index, order["MAP"], color=colors)
ax.set_xlabel("MAP (standard top-K, n=30)"); ax.set_title("Perbandingan MAP per model — smart (biru) = model live")
for i, v in enumerate(order["MAP"]): ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=9)
plt.tight_layout(); plt.show()
'''.strip()

CONSTRAINT = '''
# Constraint Satisfaction @5: % top-5 yang penuhi SEMUA kebutuhan user
# (gender + budget + fasilitas + radius 3km kampus) — bebas pooling bias
cs = pd.read_csv(ROOT / "eval" / "results_constraints.csv")
cols = [c for c in cs.columns if c.startswith("cs_at_5_")]
cs_mean = cs[cols].mean().round(3).sort_values(ascending=False)
cs_mean.index = [c.replace("cs_at_5_","") for c in cs_mean.index]
cs_mean.to_frame("mean CS@5 (n=30)")
'''.strip()

EXPERIMENTS = '''
# Ringkasan eksperimen lain (dibaca dari artefak eval)
import json
abl = pd.read_csv(ROOT / "eval" / "preprocess_ablation.csv")[["config","map","delta_map"]]
pb = json.loads((ROOT / "eval" / "explore_pooling_bias.json").read_text())["per_model"]
print("Ablation preprocessing (delta MAP saat stage dimatikan):")
print(abl.to_string(index=False))
print("\\nPooling bias — MAP saat pool adil (5 model) vs BM25-only:")
for m, d in pb.items():
    print(f"  {m:<8} {d['map_bm25pool']:.3f} -> {d['map_unionpool']:.3f}  (delta {d['delta']:+.3f})")
'''.strip()


def main() -> int:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# KozyNear — Showcase Model IR\n\n"
           "Notebook reproducible: lima model retrieval (TF-IDF, BM25, Neural "
           "MiniLM, Hybrid, **Smart**) di corpus **227 listing kos REAL** "
           "Bandar Lampung. Tiap sel dieksekusi sungguhan; output di bawah "
           "adalah hasil nyata, bukan tangkapan layar.\n\n"
           "Mata Kuliah Temu Kembali Informasi — Universitas Lampung."),
        md("## 1. Setup: muat corpus + index + pipeline"),
        code(SETUP),
        md("## 2. Preprocessing 9-stage\n\nJargon domain (`KM Dlm`→`kamar mandi "
           "dalam`), ekstraksi harga, stemming Sastrawi — langkah demi langkah."),
        code(PREP),
        md("## 3. Lima model pada satu query\n\n"
           "`\"kos putri dekat unila wifi murah\"` — bandingkan judul top-5 "
           "tiap model."),
        code(QUERY),
        md("## 4. Smart pipeline (model live)\n\nMemecah query jadi constraint "
           "terstruktur (gender/harga/fasilitas/anchor), lalu fusi teks + geo "
           "+ atribut dengan hard filter."),
        code(SMART),
        md("## 5. Evaluasi 30 query — metrik standard\n\nMAP, P@K, NDCG, MRR "
           "per model."),
        code(EVAL),
        code(CHART),
        md("## 6. Constraint Satisfaction @5 (lensa kebutuhan user)\n\n"
           "Bebas pooling bias: mengukur apakah hasil benar-benar memenuhi "
           "gender + budget + fasilitas + jarak kampus."),
        code(CONSTRAINT),
        md("## 7. Eksperimen pendukung\n\nAblation preprocessing & kuantifikasi "
           "pooling bias."),
        code(EXPERIMENTS),
        md("## Kesimpulan\n\n**Smart** unggul di MAP standard (0.359) dan "
           "dominan di CS@5 (0.867 vs BM25 0.527, p=0.0001) tanpa model neural "
           "di runtime. Skor standard cenderung meremehkan smart karena "
           "*pooling bias* (lihat sel 7): saat pool dibuat adil, jarak smart "
           "vs BM25 makin lebar. Detail metodologi di `LAPORAN.md`."),
    ]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}

    print("[execute] menjalankan notebook (load neural model, butuh ~1-2 menit)...")
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(BACKEND)}})

    NB_OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(NB_OUT))
    print(f"[saved] {NB_OUT}")

    exporter = HTMLExporter()
    exporter.exclude_input_prompt = False
    body, _ = exporter.from_notebook_node(nb)
    HTML_OUT.parent.mkdir(parents=True, exist_ok=True)
    HTML_OUT.write_text(body, encoding="utf-8")
    print(f"[saved] {HTML_OUT} ({len(body)//1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

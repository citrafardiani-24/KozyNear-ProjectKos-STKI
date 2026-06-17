"""Buat lembar anotasi MANUSIA dengan pool multi-model.

Mengobati dua cacat ilmiah terbesar evaluasi saat ini sekaligus:
1. GT simulasi (3 "annotator" = 1 heuristik + noise) -> diganti manusia.
2. Pooling bias (pool dari BM25 saja) -> pool = UNION top-10 LIMA model
   (bm25, tfidf, neural, hybrid, smart), dokumen yang hanya ditemukan model
   semantic/geo ikut ter-judge.

Output: eval/annotation_sheet.csv — satu baris per (query, dokumen), urutan
dokumen DIACAK per query (mengaburkan model asal supaya annotator tidak bias
posisi). Kolom `relevance` dikosongkan untuk diisi 0/1/2:
  0 = tidak relevan; 1 = sebagian relevan (memenuhi sebagian kebutuhan);
  2 = sangat relevan (layak direkomendasikan untuk query itu).

Cara pakai (1-3 annotator):
  1. python -m scripts.make_annotation_sheet
  2. Copy eval/annotation_sheet.csv menjadi annotation_<NAMA>.csv per orang
     (buka di Excel/Sheets, isi kolom relevance, JANGAN ubah kolom lain)
  3. python -m scripts.ingest_human_annotations --sheets eval/annotation_A.csv [eval/annotation_B.csv ...]
  4. Jalankan ulang eval dengan --ground-truth eval/ground_truth_human.csv

Estimasi beban: ~30-45 dokumen x 30 query ~ 2-4 jam per annotator.
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.indexing.hybrid import HybridIndex  # noqa: E402
from app.indexing.loader import load_all_indexes  # noqa: E402
from app.preprocessing import PreprocessingPipeline  # noqa: E402
from app.search.gazetteer import Gazetteer  # noqa: E402
from app.search.pipeline import smart_rank  # noqa: E402
from scripts.eval_smart import load_listings  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "eval" / "annotation_sheet.csv"
TOP_K_PER_MODEL = 10


def main() -> int:
    idx = load_all_indexes(ROOT / "data" / "indexes", include_neural=True)
    bm25, tfidf, neural = idx["bm25"], idx["tfidf"], idx["indobert"]
    hybrid = HybridIndex(bm25, neural)
    pipeline = PreprocessingPipeline()
    pre = lambda s: pipeline.process(s).processed  # noqa: E731
    hybrid.query_preprocessor = pre
    gz = Gazetteer.load()
    listings = load_listings()
    queries = json.loads(
        (ROOT / "eval" / "queries.json").read_text(encoding="utf-8"))["queries"]

    rng = random.Random(42)
    rows = []
    pool_sizes = []
    for q in queries:
        q_text = q["query"]
        pool: set[str] = set()
        pool |= {h.doc_id for h in bm25.query(pre(q_text), top_k=TOP_K_PER_MODEL)}
        pool |= {h.doc_id for h in tfidf.query(pre(q_text), top_k=TOP_K_PER_MODEL)}
        pool |= {h.doc_id for h in neural.query(q_text, top_k=TOP_K_PER_MODEL)}
        pool |= {h.doc_id for h in hybrid.query(q_text, top_k=TOP_K_PER_MODEL)}
        ranked, _, _ = smart_rank(q_text, bm25, listings, gz,
                                  top_k=TOP_K_PER_MODEL, preprocess=pre)
        pool |= {d for d, _ in ranked}

        docs = sorted(pool)
        rng.shuffle(docs)  # acak urutan: samarkan model asal
        pool_sizes.append(len(docs))
        for did in docs:
            r = listings.get(did)
            if r is None:
                continue
            fasilitas = ", ".join((r.fasilitas or [])[:8])
            rows.append({
                "query_id": q["id"],
                "query": q_text,
                "doc_id": did,
                "judul": r.judul,
                "tipe": r.tipe or "",
                "harga_per_bulan": r.harga_per_bulan or "",
                "kecamatan": r.kecamatan or "",
                "fasilitas": fasilitas,
                "deskripsi": (r.deskripsi or "").replace("\n", " ")[:400],
                "relevance": "",  # <- DIISI ANNOTATOR: 0 / 1 / 2
            })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8-sig", newline="") as f:  # BOM utk Excel
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    avg = sum(pool_sizes) / len(pool_sizes)
    print(f"[saved] {OUT}")
    print(f"[pool] {len(rows)} (query, doc) pairs, "
          f"rata-rata {avg:.1f} dokumen/query (min {min(pool_sizes)}, max {max(pool_sizes)})")
    print("Langkah berikutnya: copy jadi annotation_<NAMA>.csv per annotator, "
          "isi kolom relevance (0/1/2), lalu jalankan scripts.ingest_human_annotations")
    return 0


if __name__ == "__main__":
    sys.exit(main())

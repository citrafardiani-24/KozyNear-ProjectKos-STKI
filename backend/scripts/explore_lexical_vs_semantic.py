"""Eksplorasi 1: kapan semantic (neural) menang atas lexical (BM25)?

Temuan sebelumnya: setelah fielded indexing, MiniLM tidak menambah nilai
(alpha optimal -> 1.0). Hipotesis: itu karena query eval BERBAGI kosakata
dengan dokumen, jadi BM25 sudah cukup. Uji: query PARAPHRASE yang sengaja
menghindari kata di listing, dengan relevansi didefinisikan via ORACLE
METADATA (gender/geo/fasilitas/harga) yang tidak bergantung kata.

Kalau neural unggul di paraphrase tapi BM25 unggul di literal -> terbukti
batas lexical vs semantic, dengan bukti dari data sendiri.

Usage: cd backend && python -m scripts.explore_lexical_vs_semantic
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.indexing.loader import load_all_indexes  # noqa: E402
from app.preprocessing import PreprocessingPipeline  # noqa: E402
from app.search.gazetteer import haversine_km  # noqa: E402
from scripts.eval_smart import load_listings  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

# Tiap kasus: query LITERAL (berbagi kata corpus) vs SEMANTIC (paraphrase
# menghindari kata corpus), dengan oracle relevansi yang SAMA.
# oracle: ("gender", "putri") | ("geo", (lat,lng,km)) | ("fac","ac") | ("price",max)
CASES = [
    {"intent": "kos perempuan", "literal": "kos putri",
     "semantic": "hunian khusus muslimah", "oracle": ("gender", "putri")},
    {"intent": "kos laki-laki", "literal": "kos putra",
     "semantic": "indekos untuk pria lajang", "oracle": ("gender", "putra")},
    {"intent": "dekat ITERA", "literal": "kos dekat itera",
     "semantic": "tinggal dekat institut teknologi sumatera",
     "oracle": ("geo", (-5.3668, 105.3149, 3.0))},
    {"intent": "dekat UNILA", "literal": "kos dekat unila",
     "semantic": "akomodasi sekitar universitas negeri lampung",
     "oracle": ("geo", (-5.3645, 105.2434, 3.0))},
    {"intent": "ada AC", "literal": "kos ac",
     "semantic": "kamar dengan pendingin ruangan sejuk", "oracle": ("fac", "ac")},
    {"intent": "ada wifi", "literal": "kos wifi",
     "semantic": "tersedia koneksi internet nirkabel", "oracle": ("fac", "wifi")},
    {"intent": "murah", "literal": "kos murah",
     "semantic": "sewa kamar ramah kantong ekonomis", "oracle": ("price", 600000)},
    {"intent": "kamar mandi dalam", "literal": "kos kamar mandi dalam",
     "semantic": "toilet pribadi di dalam kamar", "oracle": ("fac", "mandi")},
]


def relevant(row, oracle) -> bool:
    kind, val = oracle
    if kind == "gender":
        return row.tipe == val
    if kind == "geo":
        if row.koordinat_lat is None:
            return False
        lat, lng, km = val
        return haversine_km(float(row.koordinat_lat), float(row.koordinat_lng), lat, lng) <= km
    if kind == "fac":
        return any(val in str(f).lower() for f in (row.fasilitas or []))
    if kind == "price":
        return (row.harga_per_bulan or 10**9) <= val
    return False


def p_at_k(ranked_ids, listings, oracle, k=5) -> float:
    top = ranked_ids[:k]
    if not top:
        return 0.0
    return sum(1 for d in top if d in listings and relevant(listings[d], oracle)) / len(top)


def main() -> int:
    idx = load_all_indexes(ROOT / "data" / "indexes", include_neural=True)
    bm25, tfidf, neural = idx["bm25"], idx["tfidf"], idx["indobert"]
    pipe = PreprocessingPipeline()
    pre = lambda s: pipe.process(s).processed  # noqa: E731
    listings = load_listings()

    def bm25_ids(q): return [h.doc_id for h in bm25.query(pre(q), top_k=5)]
    def tfidf_ids(q): return [h.doc_id for h in tfidf.query(pre(q), top_k=5)]
    def neural_ids(q): return [h.doc_id for h in neural.query(q, top_k=5)]

    agg = {"literal": {"bm25": [], "neural": [], "tfidf": []},
           "semantic": {"bm25": [], "neural": [], "tfidf": []}}
    print(f"{'intent':<22} {'mode':<9} {'BM25':>5} {'TFIDF':>6} {'Neural':>7}")
    for c in CASES:
        for mode in ("literal", "semantic"):
            q = c[mode]
            pb = p_at_k(bm25_ids(q), listings, c["oracle"])
            pt = p_at_k(tfidf_ids(q), listings, c["oracle"])
            pn = p_at_k(neural_ids(q), listings, c["oracle"])
            agg[mode]["bm25"].append(pb); agg[mode]["tfidf"].append(pt); agg[mode]["neural"].append(pn)
            print(f"{c['intent']:<22} {mode:<9} {pb:>5.2f} {pt:>6.2f} {pn:>7.2f}")

    print("\n=== RATA-RATA P@5 ===")
    print(f"{'mode':<10} {'BM25':>6} {'TFIDF':>6} {'Neural':>7}")
    out = {}
    for mode in ("literal", "semantic"):
        n = len(agg[mode]["bm25"])
        mb = sum(agg[mode]["bm25"]) / n
        mt = sum(agg[mode]["tfidf"]) / n
        mn = sum(agg[mode]["neural"]) / n
        out[mode] = {"bm25": mb, "tfidf": mt, "neural": mn}
        print(f"{mode:<10} {mb:>6.3f} {mt:>6.3f} {mn:>7.3f}")
    gap_lit = out["literal"]["neural"] - out["literal"]["bm25"]
    gap_sem = out["semantic"]["neural"] - out["semantic"]["bm25"]
    print(f"\nGap neural-BM25 @ literal : {gap_lit:+.3f}")
    print(f"Gap neural-BM25 @ semantic: {gap_sem:+.3f}")
    print("INTERPRETASI:", "neural unggul saat kosakata MISMATCH (semantic)"
          if gap_sem > gap_lit else "neural tidak menolong walau paraphrase")
    (ROOT / "eval" / "explore_lexical_vs_semantic.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

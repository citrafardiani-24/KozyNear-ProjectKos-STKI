"""Eksplorasi 2: error analysis query dengan AP Smart terendah.

Baca results.csv, ambil N query AP smart terburuk, dump top-5 hasil smart
+ bm25 berikut label GT-nya, supaya bisa dibaca manual: kenapa gagal?
(parser kelewatan / geo meleset / dokumen keyword-match tapi tak relevan /
label GT yang justru salah).

Usage: cd backend && python -m scripts.explore_error_analysis [--n 5]
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.indexing.bm25 import BM25Index  # noqa: E402
from app.preprocessing import PreprocessingPipeline  # noqa: E402
from app.search.gazetteer import Gazetteer  # noqa: E402
from app.search.pipeline import smart_rank  # noqa: E402
from app.search.query_parser import parse  # noqa: E402
from scripts.eval_smart import load_listings  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    bm25 = BM25Index.load(ROOT / "data" / "indexes" / "bm25.pkl")
    pipe = PreprocessingPipeline()
    pre = lambda s: pipe.process(s).processed  # noqa: E731
    gz = Gazetteer.load()
    listings = load_listings()

    queries = {q["id"]: q["query"] for q in json.loads(
        (ROOT / "eval" / "queries.json").read_text(encoding="utf-8"))["queries"]}
    gt: dict[str, dict[str, int]] = {}
    with open(ROOT / "eval" / "ground_truth.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gt.setdefault(row["query_id"], {})[row["doc_id"]] = int(row["relevance"])

    smart_ap = []
    with open(ROOT / "eval" / "results.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["model"] == "smart":
                smart_ap.append((row["query_id"], float(row["ap"])))
    worst = sorted(smart_ap, key=lambda t: t[1])[:args.n]

    def fmt(did):
        r = listings.get(did)
        if r is None:
            return f"{did} (TIDAK DI CORPUS)"
        rel = gt.get(qid, {}).get(did, "?")
        return (f"  [GT={rel}] {r.tipe or '-':<6} Rp{(r.harga_per_bulan or 0):>8} "
                f"{(r.kecamatan or '-'):<16} {r.judul[:42]}")

    for qid, ap_val in worst:
        q = queries[qid]
        parsed = parse(q, gz)
        print("=" * 78)
        print(f"{qid} AP={ap_val:.3f} | query: \"{q}\"")
        print(f"  understood: {parsed.understood}")
        n_rel = sum(1 for v in gt.get(qid, {}).values() if v >= 1)
        print(f"  total relevan di GT: {n_rel}")
        ranked, _, _ = smart_rank(q, bm25, listings, gz, top_k=5, preprocess=pre)
        print(" SMART top-5:")
        for did, _ in ranked:
            print(fmt(did))
        print(" BM25 top-5:")
        for h in bm25.query(pre(q), top_k=5):
            print(fmt(h.doc_id))
    return 0


if __name__ == "__main__":
    sys.exit(main())

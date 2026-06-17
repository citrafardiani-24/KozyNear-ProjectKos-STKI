"""Eksplorasi 3: apakah kos populer = kos relevan?

view_count & available_room ter-scrape tapi tak dipakai model. Uji:
(a) korelasi Spearman view_count/available_room dengan label GT (relevan?),
(b) re-rank smart dengan blend popularitas, lihat efek CS@5 + MAP-pool.

Kalau korelasi ~0 -> popularitas bukan sinyal relevansi (jangan dipakai).

Usage: cd backend && python -m scripts.explore_popularity_signal
"""
from __future__ import annotations
import csv, json, math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scipy.stats import spearmanr  # noqa: E402

from app.evaluation.metrics import average_precision, constraint_satisfaction_at_k  # noqa: E402
from app.indexing.bm25 import BM25Index  # noqa: E402
from app.preprocessing import PreprocessingPipeline  # noqa: E402
from app.search.gazetteer import Gazetteer  # noqa: E402
from app.search.pipeline import smart_rank  # noqa: E402
from scripts.eval_smart import load_listings  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    bm25 = BM25Index.load(ROOT / "data" / "indexes" / "bm25.pkl")
    pipe = PreprocessingPipeline()
    pre = lambda s: pipe.process(s).processed  # noqa: E731
    gz = Gazetteer.load()
    listings = load_listings()
    views = {}
    avail = {}
    for line in open(ROOT / "data" / "raw" / "mamikos_real_v2.jsonl", encoding="utf-8"):
        d = json.loads(line)
        views[d["id"]] = d.get("view_count") or 0
        avail[d["id"]] = d.get("available_room") or 0

    gt: dict[str, dict[str, int]] = {}
    with open(ROOT / "eval" / "ground_truth.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gt.setdefault(row["query_id"], {})[row["doc_id"]] = int(row["relevance"])
    queries = json.loads((ROOT / "eval" / "queries.json").read_text(encoding="utf-8"))["queries"]
    cqueries = json.loads((ROOT / "eval" / "queries_constraints.json").read_text(encoding="utf-8"))

    # (a) korelasi popularitas vs relevansi (semua pasangan judged)
    vc, av, rel = [], [], []
    for qid, docs in gt.items():
        for did, r in docs.items():
            if did in views:
                vc.append(views[did]); av.append(avail[did]); rel.append(r)
    rho_v, p_v = spearmanr(vc, rel)
    rho_a, p_a = spearmanr(av, rel)
    print(f"[korelasi vs label GT] n={len(rel)}")
    print(f"  view_count     : Spearman rho={rho_v:+.3f} (p={p_v:.3f})")
    print(f"  available_room : Spearman rho={rho_a:+.3f} (p={p_a:.3f})")

    # (b) re-rank smart dengan blend popularitas: skor' = (1-w)*rank + w*pop
    def run(weight: float):
        cs_vals, ap_vals = [], []
        # CS@5
        for cq in cqueries:
            c = dict(cq["constraints"])
            if c.get("anchor"):
                c["anchor"] = tuple(c["anchor"])
            ranked = smart_rank(cq["query"], bm25, listings, gz, top_k=20, preprocess=pre)[0]
            if weight > 0 and ranked:
                mx = max(math.log1p(views.get(d, 0)) for d, _ in ranked) or 1
                ranked = sorted(ranked, key=lambda t: -((1 - weight) * t[1] + weight * (math.log1p(views.get(t[0], 0)) / mx)))
            docs = [{"tipe": listings[d].tipe, "harga_per_bulan": listings[d].harga_per_bulan,
                     "fasilitas": listings[d].fasilitas, "lat": listings[d].koordinat_lat,
                     "lng": listings[d].koordinat_lng} for d, _ in ranked[:5] if d in listings]
            cs_vals.append(constraint_satisfaction_at_k(docs, c, k=5))
        # MAP pool
        for q in queries:
            rd = gt.get(q["id"], {})
            pool = {d: listings[d] for d in rd if d in listings}
            if not pool:
                continue
            ranked = smart_rank(q["query"], bm25, pool, gz, top_k=len(pool), preprocess=pre)[0]
            if weight > 0 and ranked:
                mx = max(math.log1p(views.get(d, 0)) for d, _ in ranked) or 1
                ranked = sorted(ranked, key=lambda t: -((1 - weight) * t[1] + weight * (math.log1p(views.get(t[0], 0)) / mx)))
            pred = [d for d, _ in ranked]
            rel_set = {d for d, r in rd.items() if r >= 1}
            ap_vals.append(average_precision(pred, rel_set))
        return sum(cs_vals) / len(cs_vals), sum(ap_vals) / len(ap_vals)

    print("\n[efek blend popularitas ke smart]")
    print(f"{'weight_pop':>10} {'CS@5':>7} {'MAP_pool':>9}")
    for w in (0.0, 0.1, 0.2, 0.3):
        cs, mp = run(w)
        print(f"{w:>10.1f} {cs:>7.3f} {mp:>9.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

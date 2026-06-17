"""Eksplorasi 5: kuantifikasi pooling bias (operasionalisasi temuan #2).

Eksplorasi 2 membuktikan skor standard smart ditekan karena GT di-pool dari
BM25 saja -> jawaban benar smart sering UNJUDGED (dihitung 0). Di sini kita
ukur besarnya bias: bangun GT diagnostik dengan POOL GABUNGAN 5 model
(bm25 ∪ tfidf ∪ neural ∪ hybrid ∪ smart, top-15 masing-masing), pakai
heuristik annotator yang SAMA, lalu bandingkan skor standard tiap model
di GT-BM25-pool vs GT-union-pool.

Ini versi OTOMATIS dari kit anotasi manusia (tetap pakai heuristik, jadi
caveat sirkularitas heuristik≈filter-smart tetap berlaku; bukan headline,
murni diagnostik seberapa besar pooling bias menekan tiap model).

Output: eval/ground_truth_unionpool.csv + eval/explore_pooling_bias.json

Usage: cd backend && python -m scripts.explore_pooling_bias
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
from random import Random
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.evaluation.metrics import average_precision, precision_at_k  # noqa: E402
from app.indexing.hybrid import HybridIndex  # noqa: E402
from app.indexing.loader import load_all_indexes  # noqa: E402
from app.preprocessing import PreprocessingPipeline  # noqa: E402
from app.search.gazetteer import Gazetteer  # noqa: E402
from app.search.pipeline import smart_rank  # noqa: E402
from scripts.eval_smart import load_listings  # noqa: E402
from scripts.generate_ground_truth import annotate, majority_vote  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
POOL_TOPK = 15


def load_gt(path: Path) -> dict[str, dict[str, int]]:
    gt: dict[str, dict[str, int]] = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            gt.setdefault(r["query_id"], {})[r["doc_id"]] = int(r["relevance"])
    return gt


def main() -> int:
    idx = load_all_indexes(ROOT / "data" / "indexes", include_neural=True)
    bm25, tfidf, neural = idx["bm25"], idx["tfidf"], idx["indobert"]
    pipe = PreprocessingPipeline()
    pre = lambda s: pipe.process(s).processed  # noqa: E731
    hybrid = HybridIndex(bm25, neural, query_preprocessor=pre)
    gz = Gazetteer.load()
    listings = load_listings()
    corpus = {d["id"]: d for d in json.loads(
        (ROOT / "data" / "processed" / "corpus.json").read_text(encoding="utf-8"))}
    queries = json.loads((ROOT / "eval" / "queries.json").read_text(encoding="utf-8"))["queries"]

    # ranker per model -> top-k doc_ids
    def rank(model, q, k):
        if model == "bm25":   return [h.doc_id for h in bm25.query(pre(q), top_k=k)]
        if model == "tfidf":  return [h.doc_id for h in tfidf.query(pre(q), top_k=k)]
        if model == "neural": return [h.doc_id for h in neural.query(q, top_k=k)]
        if model == "hybrid": return [h.doc_id for h in hybrid.query(q, top_k=k)]
        if model == "smart":  return [d for d, _ in smart_rank(q, bm25, listings, gz, top_k=k, preprocess=pre)[0]]
    models = ["bm25", "tfidf", "neural", "hybrid", "smart"]

    # 1. Pool gabungan 5 model
    cpq: dict[str, list[str]] = {}
    for q in queries:
        pool: set[str] = set()
        for m in models:
            pool |= set(rank(m, q["query"], POOL_TOPK))
        cpq[q["id"]] = sorted(pool)
    avg_pool = sum(len(v) for v in cpq.values()) / len(cpq)

    # 2. Heuristik annotator yang sama, konsensus majority
    a = annotate(queries, cpq, corpus, bias="strict")
    b = annotate(queries, cpq, corpus, bias="lenient")
    c = annotate(queries, cpq, corpus, bias="noisy", rng=Random(42))
    consensus = majority_vote(a, b, c)
    out = ROOT / "eval" / "ground_truth_unionpool.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["query_id", "doc_id", "relevance"])
        for (qid, did), rel in sorted(consensus.items()):
            w.writerow([qid, did, rel])

    gt_union: dict[str, dict[str, int]] = {}
    for (qid, did), rel in consensus.items():
        gt_union.setdefault(qid, {})[did] = rel
    gt_bm25 = load_gt(ROOT / "eval" / "ground_truth.csv")

    # 3. Skor standard tiap model di kedua GT
    def evaluate(gt):
        res = {}
        for m in models:
            aps, p5s = [], []
            for q in queries:
                pred = rank(m, q["query"], 10)
                rel_set = {d for d, r in gt.get(q["id"], {}).items() if r >= 1}
                aps.append(average_precision(pred, rel_set))
                p5s.append(precision_at_k(pred, rel_set, 5))
            n = len(queries)
            res[m] = {"MAP": round(sum(aps) / n, 4), "P@5": round(sum(p5s) / n, 4)}
        return res

    bm25pool = evaluate(gt_bm25)
    unionpool = evaluate(gt_union)

    print(f"pool union rata-rata {avg_pool:.1f} dok/query "
          f"(BM25-pool lama ~30)\n")
    print(f"{'model':<9} {'MAP@bm25pool':>13} {'MAP@unionpool':>14} {'delta':>7}")
    summary = {}
    for m in models:
        d = unionpool[m]["MAP"] - bm25pool[m]["MAP"]
        summary[m] = {"map_bm25pool": bm25pool[m]["MAP"],
                      "map_unionpool": unionpool[m]["MAP"], "delta": round(d, 4)}
        print(f"{m:<9} {bm25pool[m]['MAP']:>13.4f} {unionpool[m]['MAP']:>14.4f} {d:>+7.4f}")

    (ROOT / "eval" / "explore_pooling_bias.json").write_text(
        json.dumps({"avg_pool_size": round(avg_pool, 1), "per_model": summary},
                   indent=2), encoding="utf-8")
    print(f"\n[saved] eval/ground_truth_unionpool.csv + explore_pooling_bias.json")
    print("Interpretasi: delta positif besar = model itu paling ditekan "
          "pooling bias BM25 (jawaban benarnya banyak yang tadinya unjudged).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

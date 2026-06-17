"""Pool-restricted evaluation: kurangi bias pooling untuk IndoBERT/Hybrid.

Standard top-K evaluation menyalahkan IndoBERT/Hybrid karena pool annotation
dibangun dari BM25/TF-IDF candidates — IndoBERT-specific results (yang belum
di-annotate) otomatis di-treat rel=0 walau secara semantik mungkin relevan.

Pool-restricted eval:
1. Untuk tiap query, ambil semua doc_ids yang DI-ANNOTATE di ground_truth.
2. Score subset itu pakai tiap model.
3. Sort by score, compute metrics on the ranking within pool.

Hasil: fair comparison — semua model dinilai berdasar kemampuan ranking pool
yang sama. Cocok dilaporkan side-by-side dengan standard top-K.

Usage:
    cd backend
    python -m scripts.eval_pool_restricted
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.evaluation.metrics import (  # noqa: E402
    average_precision, ndcg_at_k, precision_at_k, reciprocal_rank,
)
from app.indexing.bm25 import BM25Index  # noqa: E402
from app.indexing.indobert import IndoBERTIndex  # noqa: E402
from app.indexing.tfidf import TFIDFIndex  # noqa: E402
from app.preprocessing import PreprocessingPipeline  # noqa: E402


def rank_pool_with_bm25(bm25: BM25Index, q_processed: str, pool: list[str]) -> list[str]:
    import numpy as np
    q_tokens = q_processed.split()
    scores = bm25.bm25.get_scores(q_tokens)
    id_to_idx = {d: i for i, d in enumerate(bm25.doc_ids)}
    pool_scored = [(d, scores[id_to_idx[d]]) for d in pool if d in id_to_idx]
    pool_scored.sort(key=lambda x: -x[1])
    return [d for d, _ in pool_scored]


def rank_pool_with_tfidf(tfidf: TFIDFIndex, q_processed: str, pool: list[str]) -> list[str]:
    from sklearn.metrics.pairwise import cosine_similarity
    q_vec = tfidf.vectorizer.transform([q_processed])
    scores = cosine_similarity(q_vec, tfidf.doc_matrix).flatten()
    id_to_idx = {d: i for i, d in enumerate(tfidf.doc_ids)}
    pool_scored = [(d, scores[id_to_idx[d]]) for d in pool if d in id_to_idx]
    pool_scored.sort(key=lambda x: -x[1])
    return [d for d, _ in pool_scored]


def rank_pool_with_indobert(indobert: IndoBERTIndex, q_raw: str, pool: list[str]) -> list[str]:
    q_emb = indobert.encode_query(q_raw)
    pairs = indobert.score_docs(q_emb, pool)
    pairs.sort(key=lambda x: -x[1])
    return [d for d, _ in pairs]


def rank_pool_with_hybrid(
    bm25: BM25Index, indobert: IndoBERTIndex,
    q_raw: str, q_processed: str, pool: list[str], alpha: float = 0.3,
) -> list[str]:
    import numpy as np
    # Filter pool to docs present in both indexes so both score arrays stay aligned
    bm25_id_to_idx = {d: i for i, d in enumerate(bm25.doc_ids)}
    pool_filtered = [d for d in pool if d in bm25_id_to_idx]

    # BM25 score filtered subset
    q_tokens = q_processed.split()
    bm25_scores_full = bm25.bm25.get_scores(q_tokens)
    bm25_pool_scores = np.array([bm25_scores_full[bm25_id_to_idx[d]] for d in pool_filtered])

    # IndoBERT score filtered subset
    q_emb = indobert.encode_query(q_raw)
    ib_pairs = dict(indobert.score_docs(q_emb, pool_filtered))
    ib_pool_scores = np.array([ib_pairs.get(d, 0.0) for d in pool_filtered])

    # Min-max normalize
    def norm(x):
        if x.size == 0 or x.max() - x.min() < 1e-9:
            return np.zeros_like(x)
        return (x - x.min()) / (x.max() - x.min())
    combined = alpha * norm(bm25_pool_scores) + (1 - alpha) * norm(ib_pool_scores)
    order = np.argsort(-combined)
    return [pool_filtered[i] for i in order]


def main() -> int:
    ROOT = Path(__file__).resolve().parents[2]
    indexes_dir = ROOT / "data" / "indexes"

    print("[load] indexes + pipeline...")
    tfidf = TFIDFIndex.load(indexes_dir / "tfidf.pkl")
    bm25 = BM25Index.load(indexes_dir / "bm25.pkl")
    indobert = IndoBERTIndex.load(indexes_dir / "indobert")
    pipeline = PreprocessingPipeline()

    queries = json.loads((ROOT / "eval" / "queries.json").read_text(encoding="utf-8"))["queries"]

    # Load ground truth
    gt: dict[str, dict[str, int]] = {}
    with open(ROOT / "eval" / "ground_truth.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gt.setdefault(row["query_id"], {})[row["doc_id"]] = int(row["relevance"])

    print(f"[loaded] {len(queries)} queries")

    out_rows = []
    aggregates: dict[str, dict[str, list[float]]] = {
        m: {"p5": [], "p10": [], "ap": [], "ndcg": [], "rr": []}
        for m in ["tfidf", "bm25", "indobert", "hybrid"]
    }

    for q in queries:
        qid = q["id"]
        q_raw = q["query"]
        q_processed = pipeline.process(q_raw).processed
        pool = list(gt.get(qid, {}).keys())
        rel_dict = gt.get(qid, {})
        rel_set = {d for d, r in rel_dict.items() if r >= 1}
        if not pool:
            continue

        rankings = {
            "tfidf": rank_pool_with_tfidf(tfidf, q_processed, pool),
            "bm25": rank_pool_with_bm25(bm25, q_processed, pool),
            "indobert": rank_pool_with_indobert(indobert, q_raw, pool),
            "hybrid": rank_pool_with_hybrid(bm25, indobert, q_raw, q_processed, pool, alpha=0.9),
        }

        for model, ranking in rankings.items():
            p5 = precision_at_k(ranking, rel_set, 5)
            p10 = precision_at_k(ranking, rel_set, 10)
            ap = average_precision(ranking, rel_set)
            ndcg = ndcg_at_k(ranking, rel_dict, 10)
            rr = reciprocal_rank(ranking, rel_set)
            aggregates[model]["p5"].append(p5)
            aggregates[model]["p10"].append(p10)
            aggregates[model]["ap"].append(ap)
            aggregates[model]["ndcg"].append(ndcg)
            aggregates[model]["rr"].append(rr)
            out_rows.append([model, qid, q_raw, p5, p10, ap, ndcg, rr])

    # Write per-query CSV
    out = ROOT / "eval" / "results_pool_restricted.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "query_id", "query", "p_at_5", "p_at_10", "ap", "ndcg_at_10", "rr"])
        w.writerows(out_rows)

    # Print aggregate
    print("\n=== POOL-RESTRICTED METRICS (fair across models) ===")
    print(f"{'model':10s} {'P@5':>8s} {'P@10':>8s} {'MAP':>8s} {'NDCG@10':>8s} {'MRR':>8s}")
    for m, agg in aggregates.items():
        n = len(agg["p5"]) or 1
        print(f"{m:10s} {sum(agg['p5'])/n:>8.4f} {sum(agg['p10'])/n:>8.4f} "
              f"{sum(agg['ap'])/n:>8.4f} {sum(agg['ndcg'])/n:>8.4f} {sum(agg['rr'])/n:>8.4f}")
    print(f"\n[saved] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

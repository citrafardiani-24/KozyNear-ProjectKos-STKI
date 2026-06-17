"""Grid search hyperparameter BM25 (k1, b) — kriteria pool-restricted MAP.

k1 mengatur saturasi term frequency, b mengatur normalisasi panjang dokumen.
Default literatur (1.5, 0.75) belum tentu pas untuk dokumen kos fielded yang
pendek. Standard eval tidak dipakai sebagai kriteria karena GT-nya di-pool
dari BM25 default (bias ke konfigurasi pool); pool-restricted me-ranking
dokumen judged yang sama sehingga adil antar konfigurasi.

Output: eval/bm25_params_grid.csv + rekomendasi di console.

Usage:
    cd backend
    python -m scripts.experiment_bm25_params
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rank_bm25 import BM25Okapi  # noqa: E402

from app.evaluation.metrics import average_precision, ndcg_at_k, precision_at_k  # noqa: E402
from app.indexing.bm25 import BM25Index  # noqa: E402
from app.preprocessing import PreprocessingPipeline  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "eval" / "bm25_params_grid.csv"

K1_GRID = [0.6, 0.9, 1.2, 1.5, 1.8, 2.1]
B_GRID = [0.3, 0.5, 0.75, 0.9, 1.0]


def main() -> int:
    base = BM25Index.load(ROOT / "data" / "indexes" / "bm25.pkl")
    tokenized = base.tokenized_corpus
    id_to_idx = {d: i for i, d in enumerate(base.doc_ids)}
    pipeline = PreprocessingPipeline()
    queries = json.loads((ROOT / "eval" / "queries.json").read_text(encoding="utf-8"))["queries"]

    gt: dict[str, dict[str, int]] = {}
    with open(ROOT / "eval" / "ground_truth.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gt.setdefault(row["query_id"], {})[row["doc_id"]] = int(row["relevance"])

    processed = {q["id"]: pipeline.process(q["query"]).processed.split() for q in queries}

    rows = []
    for k1 in K1_GRID:
        for b in B_GRID:
            engine = BM25Okapi(tokenized, k1=k1, b=b)
            aps, p5s, nds = [], [], []
            for q in queries:
                rel_dict = gt.get(q["id"], {})
                pool = [d for d in rel_dict if d in id_to_idx]
                if not pool:
                    continue
                scores = engine.get_scores(processed[q["id"]])
                ranked = sorted(pool, key=lambda d: -scores[id_to_idx[d]])
                rel_set = {d for d, r in rel_dict.items() if r >= 1}
                aps.append(average_precision(ranked, rel_set))
                p5s.append(precision_at_k(ranked, rel_set, 5))
                nds.append(ndcg_at_k(ranked, rel_dict, 10))
            n = len(aps)
            rows.append({
                "k1": k1, "b": b,
                "map_pool": round(sum(aps) / n, 4),
                "p5_pool": round(sum(p5s) / n, 4),
                "ndcg10_pool": round(sum(nds) / n, 4),
            })

    rows.sort(key=lambda r: -r["map_pool"])
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["k1", "b", "map_pool", "p5_pool", "ndcg10_pool"])
        w.writeheader(); w.writerows(rows)

    default = next(r for r in rows if r["k1"] == 1.5 and r["b"] == 0.75)
    print(f"[saved] {OUT} ({len(rows)} kombinasi)")
    print("Top 5 by pool-restricted MAP:")
    for r in rows[:5]:
        print(f"  k1={r['k1']:>4} b={r['b']:>4}  MAP={r['map_pool']}  P@5={r['p5_pool']}  NDCG={r['ndcg10_pool']}")
    print(f"Default (1.5, 0.75): MAP={default['map_pool']}  P@5={default['p5_pool']}  NDCG={default['ndcg10_pool']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

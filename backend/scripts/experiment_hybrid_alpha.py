"""Grid search Hybrid alpha untuk find optimal BM25+IndoBERT weight.

Hypothesis: alpha~0.7-0.9 (BM25-dominant) akan beat BM25 alone, karena
BM25 udah strong di lexical match, IndoBERT cuma tie-break di semantic
cases yang BM25 underranks.

Alpha range:
- 0.0 = pure IndoBERT rerank (current Hybrid behavior dengan alpha=0.3 kasih
  70% weight ke IndoBERT, yang mana lemah)
- 1.0 = pure BM25 (equivalent ke BM25 baseline)
- 0.3 = current production
- 0.5 = balanced
- 0.7-0.9 = BM25-dominant

Output:
- eval/hybrid_alpha_grid.csv: alpha vs metrics
- eval/hybrid_alpha_grid.png: bar chart

Usage:
    cd backend
    python -m scripts.experiment_hybrid_alpha
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.evaluation import (  # noqa: E402
    average_precision, mean_average_precision,
    mean_reciprocal_rank, ndcg_at_k, precision_at_k, reciprocal_rank,
)
from app.indexing.bm25 import BM25Index  # noqa: E402
from app.indexing.hybrid import HybridIndex  # noqa: E402
from app.indexing.indobert import IndoBERTIndex  # noqa: E402
from app.preprocessing import PreprocessingPipeline  # noqa: E402


def relevant_set(rel_dict: dict[str, int], threshold: int = 1) -> set[str]:
    return {did for did, rel in rel_dict.items() if rel >= threshold}


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Grid search alpha hybrid")
    parser.add_argument(
        "--pool-restricted", action="store_true",
        help=(
            "Sweep di dalam pool annotated (fair lens). Standard sweep "
            "monoton naik ke alpha=1.0 karena GT lexical-pooled (pooling "
            "bias), jadi keputusan alpha diambil dari mode ini."
        ),
    )
    args = parser.parse_args()

    ROOT = Path(__file__).resolve().parents[2]
    indexes_dir = ROOT / "data" / "indexes"
    queries_path = ROOT / "eval" / "queries.json"
    gt_path = ROOT / "eval" / "ground_truth.csv"
    suffix = "_pool" if args.pool_restricted else ""
    out_csv = ROOT / "eval" / f"hybrid_alpha_grid{suffix}.csv"
    out_png = ROOT / "eval" / f"hybrid_alpha_grid{suffix}.png"

    # Load resources
    print("[load] indexes + pipeline...")
    bm25 = BM25Index.load(indexes_dir / "bm25.pkl")
    indobert = IndoBERTIndex.load(indexes_dir / "indobert")
    pipeline = PreprocessingPipeline()

    with open(queries_path, encoding="utf-8") as f:
        queries = json.load(f)["queries"]

    # Build relevance dicts
    relevant_per_query: dict[str, set[str]] = {}
    relevance_scores_per_query: dict[str, dict[str, int]] = {}
    with open(gt_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            qid = row["query_id"]
            did = row["doc_id"]
            rel = int(row["relevance"])
            relevance_scores_per_query.setdefault(qid, {})[did] = rel
            if rel >= 1:
                relevant_per_query.setdefault(qid, set()).add(did)

    print(f"[loaded] {len(queries)} queries, {sum(len(v) for v in relevant_per_query.values())} relevant judgments")

    # Grid search
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results: list[dict] = []

    from scripts.eval_pool_restricted import rank_pool_with_hybrid

    for alpha in alphas:
        hybrid = HybridIndex(
            bm25, indobert, bm25_top_k=None, alpha=alpha,  # None = exhaustive
            query_preprocessor=lambda q: pipeline.process(q).processed,
        )

        per_q_ap: list[float] = []
        per_q_rr: list[float] = []
        per_q_p5: list[float] = []
        per_q_p10: list[float] = []
        per_q_ndcg: list[float] = []

        for q in queries:
            qid = q["id"]
            rel_scores_all = relevance_scores_per_query.get(qid, {})
            if args.pool_restricted:
                pool = list(rel_scores_all.keys())
                if not pool:
                    continue
                predicted = rank_pool_with_hybrid(
                    bm25, indobert, q["query"],
                    pipeline.process(q["query"]).processed, pool, alpha=alpha,
                )
            else:
                # Hybrid akan preprocess internal untuk BM25; IndoBERT pakai raw.
                hits = hybrid.query(q["query"], top_k=10)
                predicted = [h.doc_id for h in hits]

            rel_set = relevant_per_query.get(qid, set())
            rel_scores = rel_scores_all

            per_q_ap.append(average_precision(predicted, rel_set))
            per_q_rr.append(reciprocal_rank(predicted, rel_set))
            per_q_p5.append(precision_at_k(predicted, rel_set, 5))
            per_q_p10.append(precision_at_k(predicted, rel_set, 10))
            per_q_ndcg.append(ndcg_at_k(predicted, rel_scores, 10))

        n = len(queries)
        result = {
            "alpha": alpha,
            "map": sum(per_q_ap) / n,
            "mrr": sum(per_q_rr) / n,
            "p_at_5": sum(per_q_p5) / n,
            "p_at_10": sum(per_q_p10) / n,
            "ndcg_at_10": sum(per_q_ndcg) / n,
        }
        results.append(result)
        print(f"  alpha={alpha:.1f}  MAP={result['map']:.4f}  P@10={result['p_at_10']:.4f}  NDCG={result['ndcg_at_10']:.4f}")

    # Save CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["alpha", "map", "mrr", "p_at_5", "p_at_10", "ndcg_at_10"])
        writer.writeheader()
        for r in results:
            writer.writerow({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in r.items()})
    print(f"\n[saved] {out_csv}")

    # Find best
    best_map = max(results, key=lambda r: r["map"])
    best_ndcg = max(results, key=lambda r: r["ndcg_at_10"])
    print(f"\n=== OPTIMAL ALPHA ===")
    print(f"  best MAP:     alpha={best_map['alpha']:.1f} -> MAP={best_map['map']:.4f}")
    print(f"  best NDCG@10: alpha={best_ndcg['alpha']:.1f} -> NDCG={best_ndcg['ndcg_at_10']:.4f}")
    bm25_alone = next((r for r in results if r["alpha"] == 1.0), results[-1])
    prod_alpha = next((r for r in results if r["alpha"] == 0.3), None)
    print(f"\n  vs BM25 alone (alpha=1.0): MAP={bm25_alone['map']:.4f}")
    if prod_alpha:
        print(f"  vs current production alpha=0.3: MAP={prod_alpha['map']:.4f}")

    # Visualization
    try:
        import matplotlib.pyplot as plt

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: MAP curve
        ax = axes[0]
        ax.plot([r["alpha"] for r in results], [r["map"] for r in results],
                marker="o", linewidth=2, color="#2563eb", label="MAP")
        ax.plot([r["alpha"] for r in results], [r["ndcg_at_10"] for r in results],
                marker="s", linewidth=2, color="#dc2626", label="NDCG@10")
        ax.plot([r["alpha"] for r in results], [r["mrr"] for r in results],
                marker="^", linewidth=2, color="#10b981", label="MRR")
        ax.axvline(0.3, color="gray", linestyle="--", alpha=0.5, label="current alpha=0.3")
        ax.axvline(best_map["alpha"], color="red", linestyle=":", alpha=0.7, label=f"best MAP alpha={best_map['alpha']:.1f}")
        ax.set_xlabel("Hybrid alpha (BM25 weight)")
        ax.set_ylabel("Metric")
        ax.set_title(f"Hybrid alpha grid search (n={len(queries)} queries)")
        ax.legend(loc="best")
        ax.set_xticks(alphas)

        # Plot 2: Bar chart MAP comparison
        ax = axes[1]
        bars = ax.bar([f"α={r['alpha']:.1f}" for r in results],
                     [r["map"] for r in results], color="#2563eb")
        # Highlight best + current
        for bar, r in zip(bars, results):
            if r["alpha"] == best_map["alpha"]:
                bar.set_color("#10b981")
            elif r["alpha"] == 0.3:
                bar.set_color("#dc2626")
        ax.axhline(bm25_alone["map"], color="orange", linestyle="--", alpha=0.7,
                   label=f"BM25 alone MAP={bm25_alone['map']:.4f}")
        ax.set_xlabel("Alpha")
        ax.set_ylabel("MAP")
        ax.set_title("Hybrid MAP per alpha (green=best, red=current production)")
        ax.legend()
        plt.xticks(rotation=45)

        plt.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=120, bbox_inches="tight")
        print(f"[saved] {out_png}")
    except ImportError:
        print("[skip] matplotlib not installed, no plot")

    return 0


if __name__ == "__main__":
    sys.exit(main())

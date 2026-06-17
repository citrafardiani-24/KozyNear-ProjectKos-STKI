"""CLI runner untuk evaluation: run semua model x query, compute metrics.

Workflow:
    # 1. Pastikan: corpus + indexes + ground truth siap
    #    - data/processed/corpus.json
    #    - data/indexes/{tfidf.pkl, bm25.pkl, indobert/}
    #    - eval/queries.json
    #    - eval/ground_truth.csv

    # 2. Run evaluation
    python -m app.evaluation.runner \\
        --queries ../eval/queries.json \\
        --ground-truth ../eval/ground_truth.csv \\
        --indexes-dir ../data/indexes \\
        --output ../eval/results.csv

    # 3. Output:
    #    eval/results.csv -- per-model per-query metrics
    #    Console summary: aggregate MAP, P@5/10, NDCG@10, MRR per model
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from .kappa import interpret_kappa
from .metrics import (
    mean_average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
)
from .statistical import paired_ttest, wilcoxon_signed_rank


def load_queries(path: Path) -> list[dict[str, Any]]:
    """Load eval/queries.json -> list of {id, query, context, ...}."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("queries", data)  # support raw list atau {queries: [...]}


def load_ground_truth(path: Path) -> dict[str, dict[str, int]]:
    """Load ground_truth.csv -> {query_id: {doc_id: consensus_relevance}}.

    Expected columns: query_id, doc_id, relevance (consensus)
    """
    gt: dict[str, dict[str, int]] = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row["query_id"]
            did = row["doc_id"]
            rel = int(row["relevance"])
            gt.setdefault(qid, {})[did] = rel
    return gt


def relevant_set(rel_dict: dict[str, int], threshold: int = 1) -> set[str]:
    """Convert graded relevance ke binary relevant set (rel >= threshold)."""
    return {doc_id for doc_id, rel in rel_dict.items() if rel >= threshold}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run IR evaluation")
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--indexes-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["tfidf", "bm25", "indobert", "hybrid"],
    )
    args = parser.parse_args()

    # Load eval data
    queries = load_queries(args.queries)
    ground_truth = load_ground_truth(args.ground_truth)
    logger.info(
        f"[eval] {len(queries)} queries, "
        f"{sum(len(v) for v in ground_truth.values())} GT annotations"
    )

    # Load preprocessing pipeline (untuk preprocess queries)
    from app.preprocessing import PreprocessingPipeline

    pipeline = PreprocessingPipeline()

    # Load indexes
    from app.indexing.hybrid import HybridIndex
    from app.indexing.loader import load_all_indexes

    indexes = load_all_indexes(args.indexes_dir)
    if "bm25" in indexes and "indobert" in indexes and "hybrid" in args.models:
        indexes["hybrid"] = HybridIndex(
            indexes["bm25"],
            indexes["indobert"],
            query_preprocessor=lambda q: pipeline.process(q).processed,
        )

    # Run all model x all queries
    # Structure: results[model][query_id] = {p5, p10, ap, ndcg10, rr}
    results: dict[str, dict[str, dict[str, float]]] = {}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(
            ["model", "query_id", "query", "p_at_5", "p_at_10", "ap",
             "ndcg_at_10", "rr"]
        )

        for model_name in args.models:
            if model_name not in indexes:
                logger.warning(f"[skip] index {model_name} gak ditemukan")
                continue
            index = indexes[model_name]
            results[model_name] = {}

            for q in queries:
                qid = q["id"]
                q_text = q["query"]
                processed = pipeline.process(q_text).processed
                # BM25/TF-IDF: lexical, butuh ter-stem. IndoBERT/Hybrid:
                # semantic, perlu natural language (raw query).
                if model_name in ("indobert", "hybrid"):
                    search_q = q_text
                else:
                    search_q = processed
                hits = index.query(search_q, top_k=args.top_k)
                predicted = [h.doc_id for h in hits]

                rel_dict = ground_truth.get(qid, {})
                rel_set = relevant_set(rel_dict, threshold=1)

                p5 = precision_at_k(predicted, rel_set, 5)
                p10 = precision_at_k(predicted, rel_set, 10)
                from .metrics import average_precision, reciprocal_rank

                ap = average_precision(predicted, rel_set)
                nd10 = ndcg_at_k(predicted, rel_dict, 10)
                rr = reciprocal_rank(predicted, rel_set)

                results[model_name][qid] = {
                    "p5": p5, "p10": p10, "ap": ap, "ndcg10": nd10, "rr": rr,
                }
                writer.writerow([model_name, qid, q_text, p5, p10, ap, nd10, rr])

            logger.info(f"[done] model={model_name}")

    # Console summary
    logger.info("\n=== AGGREGATE METRICS ===")
    for model_name, per_query in results.items():
        n = len(per_query)
        if n == 0:
            continue
        avg_p5 = sum(v["p5"] for v in per_query.values()) / n
        avg_p10 = sum(v["p10"] for v in per_query.values()) / n
        avg_ap = sum(v["ap"] for v in per_query.values()) / n
        avg_ndcg = sum(v["ndcg10"] for v in per_query.values()) / n
        avg_rr = sum(v["rr"] for v in per_query.values()) / n
        logger.info(
            f"{model_name:10s} P@5={avg_p5:.4f} P@10={avg_p10:.4f} "
            f"MAP={avg_ap:.4f} NDCG@10={avg_ndcg:.4f} MRR={avg_rr:.4f}"
        )

    # Pairwise statistical significance (Wilcoxon)
    model_names = [m for m in args.models if m in results]
    if len(model_names) >= 2:
        logger.info("\n=== PAIRWISE WILCOXON (MAP) ===")
        for i, ma in enumerate(model_names):
            for mb in model_names[i + 1 :]:
                a_aps = [results[ma][qid]["ap"] for qid in results[ma]]
                b_aps = [results[mb][qid]["ap"] for qid in results[mb]]
                try:
                    test = wilcoxon_signed_rank(a_aps, b_aps)
                    logger.info(f"{ma} vs {mb}: {test}")
                except Exception as e:
                    logger.warning(f"{ma} vs {mb}: {e}")

    logger.info(f"\n[done] per-query metrics saved to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

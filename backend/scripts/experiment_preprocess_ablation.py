"""Ablation study preprocessing: matikan satu stage, ukur dampak ke BM25.

Menjawab "stage mana yang benar-benar menyumbang?" secara empiris, bukan
checklist. Untuk tiap stage yang bisa ditoggle, corpus fielded + query
di-preprocess ulang dengan stage itu OFF (yang lain ON), BM25 dibangun
in-memory, lalu dievaluasi standard (MAP/P@5) terhadap GT yang sama.

Catatan metodologis: GT di-pool dari BM25 full-pipeline, jadi delta yang
dilaporkan condong MENGUNTUNGKAN konfigurasi full (judged docs berasal dari
representasi full). Tetap informatif untuk arah dan magnitudo.

Output: eval/preprocess_ablation.csv + tabel console.

Usage:
    cd backend
    python -m scripts.experiment_preprocess_ablation
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rank_bm25 import BM25Okapi  # noqa: E402

from app.evaluation.metrics import average_precision, precision_at_k  # noqa: E402
from app.preprocessing import PipelineConfig, PreprocessingPipeline  # noqa: E402
from app.preprocessing.doc_text import compose_lexical_text  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "eval" / "preprocess_ablation.csv"

# Stage yang ditoggle (tokenize selalu on; extract_prices non-destructive)
STAGES = [
    "strip_html", "normalize_whitespace", "lowercase",
    "apply_jargon_dict", "correct_spelling", "remove_stopwords", "stem",
]


def evaluate(config: PipelineConfig, listings, queries, gt) -> tuple[float, float]:
    pipeline = PreprocessingPipeline(config)
    tokenized = [
        pipeline.process(compose_lexical_text(l)).processed.split()
        for l in listings
    ]
    doc_ids = [l["id"] for l in listings]
    engine = BM25Okapi(tokenized)
    aps, p5s = [], []
    for q in queries:
        q_tokens = pipeline.process(q["query"]).processed.split()
        scores = engine.get_scores(q_tokens)
        order = sorted(range(len(doc_ids)), key=lambda i: -scores[i])[:10]
        predicted = [doc_ids[i] for i in order]
        rel_set = {d for d, r in gt.get(q["id"], {}).items() if r >= 1}
        aps.append(average_precision(predicted, rel_set))
        p5s.append(precision_at_k(predicted, rel_set, 5))
    n = len(queries)
    return sum(aps) / n, sum(p5s) / n


def main() -> int:
    listings = [
        json.loads(l)
        for l in open(ROOT / "data" / "raw" / "kozynear_combined.jsonl",
                      encoding="utf-8") if l.strip()
    ]
    queries = json.loads(
        (ROOT / "eval" / "queries.json").read_text(encoding="utf-8"))["queries"]
    gt: dict[str, dict[str, int]] = {}
    with open(ROOT / "eval" / "ground_truth.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gt.setdefault(row["query_id"], {})[row["doc_id"]] = int(row["relevance"])

    print(f"[load] {len(listings)} listings, {len(queries)} queries")
    t0 = time.perf_counter()
    base_map, base_p5 = evaluate(PipelineConfig(), listings, queries, gt)
    print(f"[full pipeline] MAP={base_map:.4f} P@5={base_p5:.4f} "
          f"({time.perf_counter() - t0:.0f}s)")

    rows = [{"config": "full (semua stage ON)", "map": round(base_map, 4),
             "p_at_5": round(base_p5, 4), "delta_map": 0.0}]
    for stage in STAGES:
        cfg = PipelineConfig(**{stage: False})
        m, p5 = evaluate(cfg, listings, queries, gt)
        rows.append({
            "config": f"tanpa {stage}",
            "map": round(m, 4), "p_at_5": round(p5, 4),
            "delta_map": round(m - base_map, 4),
        })
        print(f"  tanpa {stage:<22} MAP={m:.4f} (delta {m - base_map:+.4f}) P@5={p5:.4f}")

    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config", "map", "p_at_5", "delta_map"])
        w.writeheader(); w.writerows(rows)
    print(f"[saved] {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

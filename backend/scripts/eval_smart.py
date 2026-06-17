"""Evaluasi model `smart` (pipeline live) memakai kode serving yang sama.

Tiga lensa:
1. Standard top-K vs ground_truth.csv -> update baris model=smart di
   eval/results.csv (baris model lain tidak disentuh).
2. Pool-restricted (ranking di dalam pool annotated) -> update
   eval/results_pool_restricted.csv.
3. Constraint-Satisfaction@5 (eval/queries_constraints.json, 15 query):
   % top-5 yang memenuhi SEMUA constraint user (gender/harga/fasilitas/
   radius 3km dari anchor) -> tulis eval/results_constraints.csv,
   bandingkan smart vs bm25.

Plus: pairwise Wilcoxon (AP) semua model di results.csv DENGAN koreksi
Holm-Bonferroni -> eval/significance_map.csv.

Listing di-load dari data/raw/mamikos_real_v2.jsonl (source of truth DB),
jadi eval tidak butuh Postgres. smart_rank() = fungsi yang sama dengan
endpoint /api/search?model=smart.

Usage:
    cd backend
    python -m scripts.eval_smart
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger  # noqa: E402

from app.evaluation.metrics import (  # noqa: E402
    average_precision,
    constraint_satisfaction_at_k,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
)
from app.evaluation.statistical import (  # noqa: E402
    holm_bonferroni,
    rank_biserial,
    wilcoxon_signed_rank,
)
from app.indexing.bm25 import BM25Index  # noqa: E402
from app.preprocessing import PreprocessingPipeline  # noqa: E402
from app.search.gazetteer import Gazetteer  # noqa: E402
from app.search.pipeline import smart_rank  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "eval"
RESULTS_CSV = EVAL_DIR / "results.csv"
POOL_CSV = EVAL_DIR / "results_pool_restricted.csv"
CONSTRAINTS_JSON = EVAL_DIR / "queries_constraints.json"
CONSTRAINTS_CSV = EVAL_DIR / "results_constraints.csv"
SIGNIFICANCE_CSV = EVAL_DIR / "significance_map.csv"
CSV_HEADER = ["model", "query_id", "query", "p_at_5", "p_at_10", "ap", "ndcg_at_10", "rr"]


def load_listings() -> dict[str, SimpleNamespace]:
    """JSONL -> adapter dengan atribut yang sama dengan ORM Listing.

    Difilter ke id yang ada di corpus.json (227): jsonl mentah berisi 240,
    13 di antaranya deskripsi kosong dan DIBUANG seed_db saat seeding DB.
    Eval harus melihat populasi listing yang sama dengan serving.
    """
    corpus = json.loads(
        (ROOT / "data" / "processed" / "corpus.json").read_text(encoding="utf-8"))
    corpus_ids = {d["id"] for d in corpus}

    rows: dict[str, SimpleNamespace] = {}
    with open(ROOT / "data" / "raw" / "mamikos_real_v2.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["id"] not in corpus_ids:
                continue
            koord = d.get("koordinat") or [None, None]
            rows[d["id"]] = SimpleNamespace(
                id=d["id"], judul=d.get("judul", ""), deskripsi=d.get("deskripsi", ""),
                harga_per_bulan=d.get("harga_per_bulan"), tipe=d.get("tipe"),
                fasilitas=d.get("fasilitas") or [], alamat=d.get("alamat"),
                kecamatan=d.get("kecamatan"),
                koordinat_lat=koord[0], koordinat_lng=koord[1],
            )
    assert len(rows) == len(corpus_ids), (
        f"listing eval {len(rows)} != corpus {len(corpus_ids)}")
    return rows


def load_ground_truth(path: Path | None = None) -> dict[str, dict[str, int]]:
    gt: dict[str, dict[str, int]] = {}
    with open(path or (EVAL_DIR / "ground_truth.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gt.setdefault(row["query_id"], {})[row["doc_id"]] = int(row["relevance"])
    return gt


def replace_model_rows(csv_path: Path, model: str, new_rows: list[list]) -> None:
    """Ganti semua baris `model` di CSV dengan new_rows; baris lain utuh."""
    existing: list[list] = []
    if csv_path.exists():
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == CSV_HEADER, f"{csv_path}: header tak terduga {header}"
            existing = [r for r in reader if r and r[0] != model]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        w.writerows(existing)
        w.writerows(new_rows)


def metric_row(model: str, qid: str, q: str, predicted: list[str],
               rel_dict: dict[str, int]) -> list:
    rel_set = {d for d, r in rel_dict.items() if r >= 1}
    return [
        model, qid, q,
        precision_at_k(predicted, rel_set, 5),
        precision_at_k(predicted, rel_set, 10),
        average_precision(predicted, rel_set),
        ndcg_at_k(predicted, rel_dict, 10),
        reciprocal_rank(predicted, rel_set),
    ]


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Eval smart 3 lensa")
    parser.add_argument(
        "--ground-truth", type=Path, default=None,
        help="Path GT alternatif (mis. ground_truth_human.csv)")
    parser.add_argument(
        "--suffix", default="",
        help="Suffix nama file output (mis. _human) supaya tidak menimpa hasil simulasi")
    args = parser.parse_args()

    global RESULTS_CSV, POOL_CSV, SIGNIFICANCE_CSV
    if args.suffix:
        RESULTS_CSV = EVAL_DIR / f"results{args.suffix}.csv"
        POOL_CSV = EVAL_DIR / f"results_pool_restricted{args.suffix}.csv"
        SIGNIFICANCE_CSV = EVAL_DIR / f"significance_map{args.suffix}.csv"

    logger.info("[load] bm25 + pipeline + gazetteer + listings...")
    bm25 = BM25Index.load(ROOT / "data" / "indexes" / "bm25.pkl")
    pipeline = PreprocessingPipeline()
    preprocess = lambda s: pipeline.process(s).processed  # noqa: E731
    gz = Gazetteer.load()
    listings = load_listings()
    gt = load_ground_truth(args.ground_truth)
    queries = json.loads((EVAL_DIR / "queries.json").read_text(encoding="utf-8"))["queries"]
    logger.info(f"[load] {len(listings)} listings, {len(queries)} queries")

    # ------------------------------------------------------------------
    # 1. Standard top-K
    # ------------------------------------------------------------------
    smart_rows = []
    for q in queries:
        ranked, _, _ = smart_rank(
            q["query"], bm25, listings, gz, top_k=10, preprocess=preprocess)
        predicted = [doc_id for doc_id, _ in ranked]
        smart_rows.append(metric_row("smart", q["id"], q["query"], predicted,
                                     gt.get(q["id"], {})))
    replace_model_rows(RESULTS_CSV, "smart", smart_rows)
    logger.info(f"[standard] smart rows -> {RESULTS_CSV}")

    # ------------------------------------------------------------------
    # 2. Pool-restricted: ranking di dalam pool annotated per query
    # ------------------------------------------------------------------
    pool_rows = []
    for q in queries:
        rel_dict = gt.get(q["id"], {})
        pool = list(rel_dict.keys())
        if not pool:
            continue
        pool_listings = {d: listings[d] for d in pool if d in listings}
        ranked, _, _ = smart_rank(
            q["query"], bm25, pool_listings, gz,
            top_k=len(pool), preprocess=preprocess)
        predicted = [doc_id for doc_id, _ in ranked]
        # Pool doc yang ke-drop hard filter tetap dihitung: taruh di buntut
        # (urut bm25 raw) supaya AP membandingkan ranking penuh seperti model lain.
        missing = [d for d in pool if d not in set(predicted) and d in listings]
        if missing:
            import numpy as np
            scores = bm25.bm25.get_scores(preprocess(q["query"]).split())
            idx_of = {d: i for i, d in enumerate(bm25.doc_ids)}
            missing.sort(key=lambda d: -scores[idx_of[d]] if d in idx_of else 0.0)
            predicted = predicted + missing
        pool_rows.append(metric_row("smart", q["id"], q["query"], predicted, rel_dict))
    replace_model_rows(POOL_CSV, "smart", pool_rows)
    logger.info(f"[pool-restricted] smart rows -> {POOL_CSV}")

    # ------------------------------------------------------------------
    # 3. Constraint-Satisfaction@5: SEMUA model (lensa kebutuhan user)
    # ------------------------------------------------------------------
    cqueries = json.loads(CONSTRAINTS_JSON.read_text(encoding="utf-8"))

    def to_dict(doc_id: str) -> dict:
        r = listings[doc_id]
        return {
            "tipe": r.tipe, "harga_per_bulan": r.harga_per_bulan,
            "fasilitas": r.fasilitas, "lat": r.koordinat_lat, "lng": r.koordinat_lng,
        }

    # Ranker per model: callable(query) -> list doc_id top-5
    from app.indexing.loader import load_all_indexes

    idx = load_all_indexes(ROOT / "data" / "indexes", include_neural=True)
    rankers: dict[str, callable] = {
        "smart": lambda q: [d for d, _ in smart_rank(
            q, bm25, listings, gz, top_k=5, preprocess=preprocess)[0]],
        "bm25": lambda q: [h.doc_id for h in bm25.query(preprocess(q), top_k=5)],
    }
    if "tfidf" in idx:
        rankers["tfidf"] = lambda q: [
            h.doc_id for h in idx["tfidf"].query(preprocess(q), top_k=5)]
    if "indobert" in idx:
        rankers["indobert"] = lambda q: [
            h.doc_id for h in idx["indobert"].query(q, top_k=5)]
        from app.indexing.hybrid import HybridIndex

        hybrid = HybridIndex(bm25, idx["indobert"], query_preprocessor=preprocess)
        rankers["hybrid"] = lambda q: [h.doc_id for h in hybrid.query(q, top_k=5)]

    model_order = [m for m in ("smart", "bm25", "tfidf", "indobert", "hybrid")
                   if m in rankers]
    cs_rows = []
    cs_agg: dict[str, list[float]] = {m: [] for m in model_order}
    for cq in cqueries:
        constraints = dict(cq["constraints"])
        if "anchor" in constraints and constraints["anchor"] is not None:
            constraints["anchor"] = tuple(constraints["anchor"])
        row = [cq.get("id", ""), cq["query"]]
        for m in model_order:
            docs = [to_dict(d) for d in rankers[m](cq["query"]) if d in listings]
            cs = constraint_satisfaction_at_k(docs, constraints, k=5)
            cs_agg[m].append(cs)
            row.append(f"{cs:.4f}")
        cs_rows.append(row)

    with open(CONSTRAINTS_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "query"] + [f"cs_at_5_{m}" for m in model_order])
        w.writerows(cs_rows)
    means = {m: sum(v) / len(v) for m, v in cs_agg.items()}
    logger.info(
        "[constraint] CS@5 " +
        " ".join(f"{m}={means[m]:.4f}" for m in model_order) +
        f" (n={len(cs_rows)}) -> {CONSTRAINTS_CSV}")
    try:
        cs_test = wilcoxon_signed_rank(cs_agg["smart"], cs_agg["bm25"])
        r_cs = rank_biserial(cs_agg["smart"], cs_agg["bm25"])
        logger.info(f"[constraint] smart vs bm25 (CS@5): {cs_test} r={r_cs:.3f}")
    except ValueError as e:
        logger.warning(f"[constraint] wilcoxon skip: {e}")

    # ------------------------------------------------------------------
    # 4. Pairwise Wilcoxon (AP, standard) semua model + Holm-Bonferroni
    # ------------------------------------------------------------------
    per_model: dict[str, dict[str, float]] = {}
    with open(RESULTS_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            per_model.setdefault(row["model"], {})[row["query_id"]] = float(row["ap"])

    models = sorted(per_model)
    qids = sorted(set.intersection(*(set(v) for v in per_model.values())))
    raw_tests: list[tuple[str, float]] = []
    stats_by_pair: dict[str, tuple[float, int, float]] = {}
    for i, ma in enumerate(models):
        for mb in models[i + 1:]:
            a = [per_model[ma][qid] for qid in qids]
            b = [per_model[mb][qid] for qid in qids]
            try:
                t = wilcoxon_signed_rank(a, b)
                r = rank_biserial(a, b)
                raw_tests.append((f"{ma} vs {mb}", t.p_value))
                stats_by_pair[f"{ma} vs {mb}"] = (t.statistic, t.n, r)
            except ValueError as e:
                logger.warning(f"{ma} vs {mb}: {e}")

    holm = holm_bonferroni(raw_tests, alpha=0.05)
    with open(SIGNIFICANCE_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair", "statistic", "n", "p_value", "p_holm",
                    "r_rank_biserial", "significant_raw", "significant_holm"])
        for entry in holm:
            stat, n, r = stats_by_pair[entry.label]
            w.writerow([
                entry.label, f"{stat:.2f}", n, f"{entry.p_value:.4f}",
                f"{entry.p_adjusted:.4f}", f"{r:.3f}",
                "yes" if entry.p_value < 0.05 else "no",
                "yes" if entry.significant else "no",
            ])
    n_raw = sum(1 for e in holm if e.p_value < 0.05)
    n_holm = sum(1 for e in holm if e.significant)
    logger.info(
        f"[significance] {len(holm)} pasangan: {n_raw} signifikan (raw) -> "
        f"{n_holm} setelah Holm -> {SIGNIFICANCE_CSV}")

    # Aggregate ringkas smart untuk console
    n = len(smart_rows)
    logger.info(
        "[smart standard] P@5={:.4f} P@10={:.4f} MAP={:.4f} NDCG@10={:.4f} MRR={:.4f}".format(
            sum(float(r[3]) for r in smart_rows) / n,
            sum(float(r[4]) for r in smart_rows) / n,
            sum(float(r[5]) for r in smart_rows) / n,
            sum(float(r[6]) for r in smart_rows) / n,
            sum(float(r[7]) for r in smart_rows) / n,
        ))
    n = len(pool_rows)
    logger.info(
        "[smart pool]     P@5={:.4f} P@10={:.4f} MAP={:.4f} NDCG@10={:.4f} MRR={:.4f}".format(
            sum(float(r[3]) for r in pool_rows) / n,
            sum(float(r[4]) for r in pool_rows) / n,
            sum(float(r[5]) for r in pool_rows) / n,
            sum(float(r[6]) for r in pool_rows) / n,
            sum(float(r[7]) for r in pool_rows) / n,
        ))
    return 0


if __name__ == "__main__":
    sys.exit(main())

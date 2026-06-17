"""Evaluation dashboard endpoints (file-based).

Sumber data: CSV hasil eval yang di-COMMIT di repo (eval/results.csv,
results_pool_restricted.csv, results_constraints.csv, significance_map.csv).
Sebelumnya endpoint ini membaca tabel `eval_results` yang TIDAK PERNAH diisi
oleh siapa pun (runner hanya menulis CSV) sehingga selalu kosong di production.
File-based = nol langkah deploy ekstra dan hasil identik dengan laporan.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/eval", tags=["evaluation"])

# repo root: backend/app/api/eval.py -> parents[3]
_EVAL_DIR = Path(__file__).resolve().parents[3] / "eval"


class ModelMetrics(BaseModel):
    model: str
    p_at_5: float
    p_at_10: float
    map: float
    ndcg_at_10: float
    mrr: float
    n_queries: int


class ConstraintSummary(BaseModel):
    n_queries: int
    mean_cs_at_5: dict[str, float]


class SignificanceRow(BaseModel):
    pair: str
    p_value: float
    p_holm: float
    r_rank_biserial: float | None = None
    significant_raw: bool
    significant_holm: bool


class EvalSummary(BaseModel):
    standard: list[ModelMetrics]
    pool_restricted: list[ModelMetrics]
    constraints: ConstraintSummary | None
    significance: list[SignificanceRow]
    total_queries: int
    note: str


class QueryResult(BaseModel):
    query_id: str
    query_text: str
    model: str
    p_at_5: float
    p_at_10: float
    average_precision: float
    ndcg_at_10: float
    reciprocal_rank: float


def _aggregate_results_csv(path: Path) -> tuple[list[ModelMetrics], int]:
    """results.csv -> aggregate per model + jumlah query."""
    if not path.exists():
        return [], 0
    per_model: dict[str, list[dict]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            per_model[row["model"]].append(row)
    out: list[ModelMetrics] = []
    n_queries = 0
    for model, rows in sorted(per_model.items()):
        n = len(rows)
        n_queries = max(n_queries, n)
        out.append(ModelMetrics(
            model=model,
            p_at_5=sum(float(r["p_at_5"]) for r in rows) / n,
            p_at_10=sum(float(r["p_at_10"]) for r in rows) / n,
            map=sum(float(r["ap"]) for r in rows) / n,
            ndcg_at_10=sum(float(r["ndcg_at_10"]) for r in rows) / n,
            mrr=sum(float(r["rr"]) for r in rows) / n,
            n_queries=n,
        ))
    out.sort(key=lambda m: -m.map)
    return out, n_queries


def _load_constraints(path: Path) -> ConstraintSummary | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    # Kolom model dinamis: cs_at_5_<model>
    model_cols = [c for c in rows[0].keys() if c.startswith("cs_at_5_")]
    return ConstraintSummary(
        n_queries=len(rows),
        mean_cs_at_5={
            col.removeprefix("cs_at_5_"):
                sum(float(r[col]) for r in rows) / len(rows)
            for col in model_cols
        },
    )


def _load_significance(path: Path) -> list[SignificanceRow]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [
            SignificanceRow(
                pair=r["pair"],
                p_value=float(r["p_value"]),
                p_holm=float(r["p_holm"]),
                r_rank_biserial=(
                    float(r["r_rank_biserial"])
                    if r.get("r_rank_biserial") not in (None, "")
                    else None
                ),
                significant_raw=r["significant_raw"] == "yes",
                significant_holm=r["significant_holm"] == "yes",
            )
            for r in csv.DictReader(f)
        ]


@router.get("/summary", response_model=EvalSummary)
async def eval_summary() -> EvalSummary:
    """Aggregate metrics per IR model: standard, pool-restricted, CS@5, Wilcoxon+Holm."""
    standard, n_queries = _aggregate_results_csv(_EVAL_DIR / "results.csv")
    pool, _ = _aggregate_results_csv(_EVAL_DIR / "results_pool_restricted.csv")
    return EvalSummary(
        standard=standard,
        pool_restricted=pool,
        constraints=_load_constraints(_EVAL_DIR / "results_constraints.csv"),
        significance=_load_significance(_EVAL_DIR / "significance_map.csv"),
        total_queries=n_queries,
        note=(
            "Standard top-K memakai qrels yang di-pool dari kandidat lexical "
            "(pooling bias menekan model semantic & smart geo-augment); "
            "pool-restricted = ranking di dalam pool annotated (fair); "
            "CS@5 = % top-5 yang memenuhi semua constraint user (bebas qrels). "
            "Signifikansi: Wilcoxon signed-rank + koreksi Holm-Bonferroni."
        ),
    )


@router.get("/query/{query_id}", response_model=list[QueryResult])
async def eval_per_query(query_id: str) -> list[QueryResult]:
    """Per-model metric untuk satu query (drill-down dari results.csv)."""
    path = _EVAL_DIR / "results.csv"
    if not path.exists():
        raise HTTPException(404, "results.csv belum ada — jalankan evaluasi dulu")
    out: list[QueryResult] = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["query_id"] == query_id:
                out.append(QueryResult(
                    query_id=r["query_id"],
                    query_text=r["query"],
                    model=r["model"],
                    p_at_5=float(r["p_at_5"]),
                    p_at_10=float(r["p_at_10"]),
                    average_precision=float(r["ap"]),
                    ndcg_at_10=float(r["ndcg_at_10"]),
                    reciprocal_rank=float(r["rr"]),
                ))
    if not out:
        raise HTTPException(404, f"Query '{query_id}' tidak ada di results.csv")
    out.sort(key=lambda x: x.model)
    return out

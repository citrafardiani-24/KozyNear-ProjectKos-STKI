"""IR evaluation metrics: P@K, MAP, NDCG@K, MRR.

Convention:
- `predicted` = ordered list of doc_ids dari index query (rank 1 = first)
- `relevant` = set of doc_ids yang TRULY relevant (dari ground truth)
- `relevance_scores` = dict {doc_id: int} untuk graded relevance (0/1/2)

Semua metric in [0, 1] dengan 1 = perfect.
"""

from __future__ import annotations

import math
from collections.abc import Iterable


# =============================================================================
# Precision @ K
# =============================================================================
def precision_at_k(
    predicted: list[str], relevant: Iterable[str], k: int
) -> float:
    """P@K = proportion of top-K predicted yang relevant.

    Example:
        predicted = ['a', 'b', 'c', 'd', 'e']
        relevant = {'a', 'c', 'x'}
        k = 5  -> 2/5 = 0.4
    """
    if k <= 0:
        return 0.0
    relevant_set = set(relevant)
    top_k = predicted[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc_id in top_k if doc_id in relevant_set)
    return hits / k


# =============================================================================
# Average Precision (AP) per query
# =============================================================================
def average_precision(
    predicted: list[str], relevant: Iterable[str]
) -> float:
    """Average Precision per query.

    AP = mean of P@k untuk setiap relevant doc yang muncul di predicted.

    Example:
        predicted = ['a', 'b', 'c', 'd']
        relevant = {'a', 'c'}
        -> P@1 = 1/1 (a hit), P@3 = 2/3 (c hit)
        -> AP = (1.0 + 0.667) / 2 = 0.833
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0

    precisions: list[float] = []
    hits = 0
    for i, doc_id in enumerate(predicted, start=1):
        if doc_id in relevant_set:
            hits += 1
            precisions.append(hits / i)

    if not precisions:
        return 0.0
    # Divide by number of relevant docs (kalau gak semua relevant ada di predicted,
    # bagi tetap pake count of relevant — penalti untuk missed relevant)
    return sum(precisions) / len(relevant_set)


def mean_average_precision(
    predicted_per_query: dict[str, list[str]],
    relevant_per_query: dict[str, Iterable[str]],
) -> float:
    """MAP = mean of AP across queries."""
    if not predicted_per_query:
        return 0.0
    aps: list[float] = []
    for query_id, predicted in predicted_per_query.items():
        relevant = relevant_per_query.get(query_id, [])
        aps.append(average_precision(predicted, relevant))
    return sum(aps) / len(aps)


# =============================================================================
# NDCG @ K (Normalized Discounted Cumulative Gain)
# =============================================================================
def _dcg_at_k(gains: list[float], k: int) -> float:
    """DCG@K = sum gain_i / log2(i+1), 1-indexed rank."""
    return sum(g / math.log2(i + 1) for i, g in enumerate(gains[:k], start=1))


def ndcg_at_k(
    predicted: list[str],
    relevance_scores: dict[str, int],
    k: int = 10,
) -> float:
    """NDCG@K dengan graded relevance.

    Args:
        predicted: ordered doc_ids dari IR model
        relevance_scores: {doc_id: int} -- typical 0/1/2 atau 0-3 scale
        k: cutoff rank

    Score in [0, 1]. Penalize relevant docs di posisi rendah.
    """
    if k <= 0 or not predicted:
        return 0.0

    # Actual DCG: pakai relevance_scores
    gains = [relevance_scores.get(doc_id, 0) for doc_id in predicted[:k]]
    dcg = _dcg_at_k(gains, k)

    # Ideal DCG: sort all relevant docs by score desc
    ideal_gains = sorted(relevance_scores.values(), reverse=True)
    idcg = _dcg_at_k(ideal_gains, k)

    if idcg == 0:
        return 0.0
    return dcg / idcg


# =============================================================================
# Mean Reciprocal Rank (MRR)
# =============================================================================
def reciprocal_rank(
    predicted: list[str], relevant: Iterable[str]
) -> float:
    """1 / rank dari relevant pertama di predicted. 0 kalau gak ada relevant."""
    relevant_set = set(relevant)
    for i, doc_id in enumerate(predicted, start=1):
        if doc_id in relevant_set:
            return 1.0 / i
    return 0.0


def mean_reciprocal_rank(
    predicted_per_query: dict[str, list[str]],
    relevant_per_query: dict[str, Iterable[str]],
) -> float:
    """MRR = mean of reciprocal_rank across queries."""
    if not predicted_per_query:
        return 0.0
    rrs: list[float] = []
    for query_id, predicted in predicted_per_query.items():
        relevant = relevant_per_query.get(query_id, [])
        rrs.append(reciprocal_rank(predicted, relevant))
    return sum(rrs) / len(rrs)


# =============================================================================
# Constraint Satisfaction @ K (lensa kedua: ukur kebutuhan user, bukan teks)
# =============================================================================
def constraint_satisfaction_at_k(
    results: list[dict],
    constraints: dict,
    k: int = 5,
    max_km: float = 3.0,
) -> float:
    """Rasio top-K hasil yang memenuhi SEMUA konstrain query yang ada.

    Berbeda dari P@K/MAP (yang mengukur relevansi teks dari qrels), metric ini
    mengukur hal yang dioptimalkan smart pipeline: gender benar + harga sesuai +
    fasilitas ada + dalam radius dari anchor. Tidak butuh qrels.

    Args:
        results: list dict listing TERURUT, key: tipe, harga_per_bulan,
                 fasilitas (list[str]), lat, lng.
        constraints: {gender, harga_max, fasilitas: list[str], anchor: (lat, lng)|None}.
        k: cutoff. max_km: ambang jarak "dekat" dari anchor.
    """
    from app.search.gazetteer import haversine_km

    topk = results[:k]
    if not topk:
        return 0.0

    ok = 0
    for r in topk:
        good = True
        if constraints.get("gender") and r.get("tipe") != constraints["gender"]:
            good = False
        if constraints.get("harga_max") and (r.get("harga_per_bulan") or 0) > constraints["harga_max"]:
            good = False
        have = [str(f).lower() for f in (r.get("fasilitas") or [])]
        for kw in constraints.get("fasilitas", []):
            if not any(kw in f for f in have):
                good = False
        anchor = constraints.get("anchor")
        if anchor and r.get("lat") is not None and r.get("lng") is not None:
            if haversine_km(float(r["lat"]), float(r["lng"]), anchor[0], anchor[1]) > max_km:
                good = False
        ok += int(good)
    return ok / len(topk)

"""Evaluation metrics + statistical tests untuk IR experiment.

Public API:
    from app.evaluation import (
        precision_at_k, average_precision, mean_average_precision,
        ndcg_at_k, reciprocal_rank, mean_reciprocal_rank,
        cohen_kappa, weighted_kappa,
        paired_ttest, wilcoxon_signed_rank,
    )
"""

from .kappa import cohen_kappa, weighted_kappa
from .metrics import (
    average_precision,
    mean_average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
)
from .statistical import paired_ttest, wilcoxon_signed_rank

__all__ = [
    # Metrics
    "precision_at_k",
    "average_precision",
    "mean_average_precision",
    "ndcg_at_k",
    "reciprocal_rank",
    "mean_reciprocal_rank",
    # Inter-annotator agreement
    "cohen_kappa",
    "weighted_kappa",
    # Statistical significance
    "paired_ttest",
    "wilcoxon_signed_rank",
]

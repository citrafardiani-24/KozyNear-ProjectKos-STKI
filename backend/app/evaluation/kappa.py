"""Cohen's Kappa untuk inter-annotator agreement.

Course requirement (rubric Evaluation 10%): target Kappa >= 0.7 across
all annotator pairs. Kalau di bawah 0.7, dokumentasikan + lakuin
consensus resolution sebelum compute metric IR.

Interpretation (Landis & Koch 1977):
- < 0.20: slight agreement
- 0.21 - 0.40: fair
- 0.41 - 0.60: moderate
- 0.61 - 0.80: substantial
- 0.81 - 1.00: almost perfect
"""

from __future__ import annotations

import numpy as np


def cohen_kappa(
    annotator_a: list[int],
    annotator_b: list[int],
    labels: list[int] | None = None,
) -> float:
    """Cohen's Kappa untuk 2 annotator pada same items.

    Args:
        annotator_a: labels dari annotator A (urutan harus match B)
        annotator_b: labels dari annotator B
        labels: list of possible label values (default: {0, 1, 2})

    Returns:
        Kappa value in [-1, 1]. 1 = perfect agreement, 0 = chance level.
    """
    if len(annotator_a) != len(annotator_b):
        raise ValueError(
            f"Length mismatch: A={len(annotator_a)}, B={len(annotator_b)}"
        )
    n = len(annotator_a)
    if n == 0:
        return 0.0

    if labels is None:
        labels = sorted(set(annotator_a) | set(annotator_b))

    # Confusion matrix
    label_to_idx = {label: i for i, label in enumerate(labels)}
    k = len(labels)
    matrix = np.zeros((k, k))
    for a, b in zip(annotator_a, annotator_b):
        matrix[label_to_idx[a]][label_to_idx[b]] += 1

    # Observed agreement (Po)
    po = np.trace(matrix) / n

    # Expected agreement by chance (Pe)
    row_marginals = matrix.sum(axis=1) / n
    col_marginals = matrix.sum(axis=0) / n
    pe = float(np.sum(row_marginals * col_marginals))

    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0

    return float((po - pe) / (1 - pe))


def weighted_kappa(
    annotator_a: list[int],
    annotator_b: list[int],
    labels: list[int] | None = None,
    weight_type: str = "linear",
) -> float:
    """Weighted Cohen's Kappa untuk ordinal labels (0/1/2 relevance).

    Lebih appropriate dari plain kappa untuk graded relevance: disagreement
    1 vs 2 di-penalize lebih ringan dari 0 vs 2.

    Args:
        weight_type: 'linear' atau 'quadratic' (lebih tegas)
    """
    if len(annotator_a) != len(annotator_b):
        raise ValueError("Length mismatch")
    n = len(annotator_a)
    if n == 0:
        return 0.0

    if labels is None:
        labels = sorted(set(annotator_a) | set(annotator_b))
    k = len(labels)
    label_to_idx = {label: i for i, label in enumerate(labels)}

    # Confusion matrix
    matrix = np.zeros((k, k))
    for a, b in zip(annotator_a, annotator_b):
        matrix[label_to_idx[a]][label_to_idx[b]] += 1

    # Weight matrix
    weights = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            diff = abs(i - j)
            if weight_type == "linear":
                weights[i][j] = diff / (k - 1) if k > 1 else 0
            elif weight_type == "quadratic":
                weights[i][j] = (diff / (k - 1)) ** 2 if k > 1 else 0
            else:
                raise ValueError(f"Unknown weight_type: {weight_type}")

    # Expected matrix (outer product of marginals)
    row_sum = matrix.sum(axis=1)
    col_sum = matrix.sum(axis=0)
    expected = np.outer(row_sum, col_sum) / n

    weighted_observed = np.sum(weights * matrix)
    weighted_expected = np.sum(weights * expected)

    if weighted_expected == 0:
        return 1.0 if weighted_observed == 0 else 0.0
    return float(1 - weighted_observed / weighted_expected)


def interpret_kappa(kappa: float) -> str:
    """Kategorisasi kappa per Landis & Koch (1977)."""
    if kappa < 0.0:
        return "less than chance"
    elif kappa < 0.20:
        return "slight"
    elif kappa < 0.40:
        return "fair"
    elif kappa < 0.60:
        return "moderate"
    elif kappa < 0.80:
        return "substantial"
    else:
        return "almost perfect"

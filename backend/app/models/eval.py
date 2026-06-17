"""ORM + Pydantic schemas untuk eval data (queries, ground truth, results).

Tables di-create via Alembic migration 001_initial_schema.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict
from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Numeric,
    SmallInteger,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base


# =============================================================================
# ORM models
# =============================================================================
class Query(Base):
    """Query set untuk evaluation."""
    __tablename__ = "queries"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    context: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    expected_tipe: Mapped[Optional[str]] = mapped_column(
        String(20), nullable=True
    )


class GroundTruth(Base):
    """Raw annotation per (query, listing, annotator)."""
    __tablename__ = "ground_truth"

    query_id: Mapped[str] = mapped_column(
        String, ForeignKey("queries.id"), primary_key=True
    )
    listing_id: Mapped[str] = mapped_column(
        String, ForeignKey("listings.id"), primary_key=True
    )
    annotator: Mapped[str] = mapped_column(String(50), primary_key=True)
    relevance: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    __table_args__ = (
        CheckConstraint(
            "relevance IN (0, 1, 2)", name="ck_relevance_0_1_2"
        ),
    )


class GroundTruthConsensus(Base):
    """Consensus labels (post-discussion). Input untuk evaluation runner."""
    __tablename__ = "ground_truth_consensus"

    query_id: Mapped[str] = mapped_column(
        String, ForeignKey("queries.id"), primary_key=True
    )
    listing_id: Mapped[str] = mapped_column(
        String, ForeignKey("listings.id"), primary_key=True
    )
    relevance: Mapped[int] = mapped_column(SmallInteger, nullable=False)

    __table_args__ = (
        CheckConstraint(
            "relevance IN (0, 1, 2)", name="ck_consensus_relevance"
        ),
    )


class EvalResult(Base):
    """Per-model per-query evaluation metrics."""
    __tablename__ = "eval_results"

    model: Mapped[str] = mapped_column(String(20), primary_key=True)
    query_id: Mapped[str] = mapped_column(
        String, ForeignKey("queries.id"), primary_key=True
    )
    precision_at_5: Mapped[Optional[float]] = mapped_column(
        Numeric(5, 4), nullable=True
    )
    precision_at_10: Mapped[Optional[float]] = mapped_column(
        Numeric(5, 4), nullable=True
    )
    average_precision: Mapped[Optional[float]] = mapped_column(
        Numeric(5, 4), nullable=True
    )
    ndcg_at_10: Mapped[Optional[float]] = mapped_column(
        Numeric(5, 4), nullable=True
    )
    reciprocal_rank: Mapped[Optional[float]] = mapped_column(
        Numeric(5, 4), nullable=True
    )
    computed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, server_default=func.now(), nullable=True
    )


# =============================================================================
# Pydantic schemas (API)
# =============================================================================
class ModelMetrics(BaseModel):
    """Aggregate metrics per IR model."""
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    model: str
    p_at_5: float
    p_at_10: float
    map: float
    ndcg_at_10: float
    mrr: float
    n_queries: int


class EvalSummary(BaseModel):
    """Response untuk /eval/summary."""
    models: list[ModelMetrics]
    total_queries: int
    total_listings_indexed: int


class QueryResult(BaseModel):
    """Per-query metric breakdown untuk /eval/query/{id}."""
    model_config = ConfigDict(protected_namespaces=())

    query_id: str
    query_text: str
    model: str
    p_at_5: float
    p_at_10: float
    average_precision: float
    ndcg_at_10: float
    reciprocal_rank: float

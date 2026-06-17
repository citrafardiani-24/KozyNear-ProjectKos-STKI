"""Pydantic schemas untuk /search endpoint request/response."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.models.listing import ListingRead


# Allowed IR model names (sinkron dengan IndexBase.name di app.indexing.*)
Model = Literal["tfidf", "bm25", "indobert", "hybrid", "smart"]


class SearchResponse(BaseModel):
    """Response untuk GET /search."""
    query: str
    model: Model
    top_k: int
    took_ms: int = Field(..., description="Latency total search dalam millisecond")
    results: list[ListingRead]
    understood: dict = Field(
        default_factory=dict, description="Atribut terdeteksi dari query (mode smart)"
    )
    relaxed: list[str] = Field(
        default_factory=list, description="Filter yang dilonggarkan (mode smart)"
    )

"""Listing ORM model + Pydantic schemas.

Analogi Laravel:
- Listing ORM = Eloquent Model (database mapping)
- ListingRead/Detail Pydantic = Form Request / API Resource (serialization)

ORM untuk persistence, Pydantic untuk API I/O. Konversi lewat
`ListingRead.model_validate(listing_orm)` atau `from_attributes=True`.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import ARRAY, Date, DateTime, Integer, Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base


# =============================================================================
# ORM model
# =============================================================================
class Listing(Base):
    """Table `listings` — raw scraped + processed kos data."""

    __tablename__ = "listings"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    judul: Mapped[str] = mapped_column(Text, nullable=False)
    deskripsi: Mapped[str] = mapped_column(Text, nullable=False)
    deskripsi_processed: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    harga_per_bulan: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    tipe: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    fasilitas: Mapped[Optional[list[str]]] = mapped_column(
        ARRAY(String), nullable=True
    )

    alamat: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    kecamatan: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    koordinat_lat: Mapped[Optional[float]] = mapped_column(
        Numeric(8, 5), nullable=True
    )
    koordinat_lng: Mapped[Optional[float]] = mapped_column(
        Numeric(8, 5), nullable=True
    )
    jarak_kampus_km: Mapped[Optional[float]] = mapped_column(
        Numeric(5, 2), nullable=True
    )

    url_source: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    scrape_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    source: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    inserted_at: Mapped[Optional[date]] = mapped_column(
        DateTime, server_default=func.now(), nullable=True
    )


# =============================================================================
# Pydantic schemas (API I/O)
# =============================================================================
class ListingBase(BaseModel):
    """Shared fields antara list & detail."""
    model_config = ConfigDict(from_attributes=True)

    id: str
    judul: str
    harga_per_bulan: Optional[int] = None
    tipe: Optional[str] = None
    fasilitas: Optional[list[str]] = None
    alamat: Optional[str] = None
    kecamatan: Optional[str] = None


class ListingRead(ListingBase):
    """Schema untuk hasil search (deskripsi present, score dari IR model)."""
    deskripsi: str = Field(..., description="Original deskripsi; UI truncate ~200 chars")
    score: float = Field(..., description="Relevance score dari IR model")
    koordinat: Optional[list[float]] = Field(
        None, description="[lat, lng] untuk pin Google Maps di frontend"
    )


class ListingDetail(ListingBase):
    """Schema untuk endpoint /listings/{id} (full detail)."""
    deskripsi: str
    deskripsi_processed: Optional[str] = None
    koordinat: Optional[list[float]] = Field(
        None, description="[lat, lng] format"
    )
    jarak_kampus_km: Optional[float] = None
    url_source: Optional[str] = None
    scrape_date: Optional[date] = None
    source: Optional[str] = None

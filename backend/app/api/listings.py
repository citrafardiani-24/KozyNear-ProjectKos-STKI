"""Listing detail endpoint: GET /listings/{id}."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_session
from app.models.listing import Listing, ListingDetail


router = APIRouter()


@router.get(
    "/listings/{listing_id}",
    response_model=ListingDetail,
    tags=["listings"],
)
async def get_listing(
    listing_id: str,
    session: AsyncSession = Depends(get_session),
) -> ListingDetail:
    """Get detail listing by ID. Return 404 kalau gak ditemukan."""
    result = await session.execute(
        select(Listing).where(Listing.id == listing_id)
    )
    listing = result.scalar_one_or_none()
    if not listing:
        raise HTTPException(
            status_code=404,
            detail=f"Listing '{listing_id}' not found",
        )

    coordinates = None
    if listing.koordinat_lat is not None and listing.koordinat_lng is not None:
        coordinates = [
            float(listing.koordinat_lat),
            float(listing.koordinat_lng),
        ]

    return ListingDetail(
        id=listing.id,
        judul=listing.judul,
        deskripsi=listing.deskripsi,
        deskripsi_processed=listing.deskripsi_processed,
        harga_per_bulan=listing.harga_per_bulan,
        tipe=listing.tipe,
        fasilitas=listing.fasilitas,
        alamat=listing.alamat,
        kecamatan=listing.kecamatan,
        koordinat=coordinates,
        jarak_kampus_km=(
            float(listing.jarak_kampus_km)
            if listing.jarak_kampus_km is not None
            else None
        ),
        url_source=listing.url_source,
        scrape_date=listing.scrape_date,
        source=listing.source,
    )

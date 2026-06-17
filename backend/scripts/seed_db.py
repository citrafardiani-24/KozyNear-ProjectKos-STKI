"""Seed PostgreSQL listings table dari scraper output (JSONL).

Usage:
    cd backend
    # Default: UPSERT (idempotent, TIDAK menghapus row lama)
    python -m scripts.seed_db --input ../data/raw/mamikos_real_v2.jsonl

    # Reconcile DB ke source (hapus row yang tak ada di input jsonl):
    python -m scripts.seed_db --input ../data/raw/mamikos_real_v2.jsonl --truncate

    # Preview dulu tanpa menulis apa pun (aman buat production):
    python -m scripts.seed_db --input ../data/raw/mamikos_real_v2.jsonl --truncate --dry-run

Pipeline:
1. Load JSONL listings (output dari app.scraper.runner)
2. Filter: deskripsi non-kosong + ada judul (lihat MIN_DESKRIPSI_WORDS)
3. Tulis ke listings table:
   - default    : UPSERT (INSERT ON CONFLICT DO UPDATE), tidak menghapus apa pun
   - --truncate : TRUNCATE CASCADE lalu reseed, DB == source persis
4. Optional: run preprocessing + save corpus.json untuk indexing
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import date as date_type
from pathlib import Path
from typing import Any

from loguru import logger
from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert

from app.core.db import async_session_factory, engine, init_db
from app.models.eval import GroundTruth, GroundTruthConsensus
from app.models.listing import Listing


# Real Mamikos descriptions terse (median ~23 kata, owner-written) — threshold
# lama 100 (warisan synthetic yang di-pad) akan drop ~99% data real. Keep semua
# listing dengan deskripsi non-kosong, selaras dengan corpus.json (drop empty only).
MIN_DESKRIPSI_WORDS = 1


def parse_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL file ke list of dicts."""
    listings: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                listings.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"[parse] line {line_no} invalid JSON: {e}")
    return listings


def filter_listings(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter listings sesuai brief requirements."""
    filtered = []
    for item in raw:
        deskripsi = item.get("deskripsi", "")
        word_count = len(deskripsi.split())
        if word_count < MIN_DESKRIPSI_WORDS:
            logger.debug(f"[skip <100 words] {item.get('id')} (got {word_count})")
            continue
        if not item.get("judul"):
            logger.debug(f"[skip no judul] {item.get('id')}")
            continue
        filtered.append(item)
    logger.info(
        f"[filter] {len(filtered)}/{len(raw)} listings pass quality bar"
    )
    return filtered


def to_orm_dict(item: dict[str, Any]) -> dict[str, Any]:
    """Convert raw listing dict ke dict cocok untuk Listing ORM."""
    koordinat = item.get("koordinat")
    lat, lng = (None, None)
    if isinstance(koordinat, (list, tuple)) and len(koordinat) == 2:
        lat, lng = koordinat

    scrape_date_raw = item.get("scrape_date")
    scrape_date = None
    if scrape_date_raw:
        try:
            scrape_date = date_type.fromisoformat(scrape_date_raw)
        except ValueError:
            pass

    return {
        "id": item["id"],
        "judul": item["judul"],
        "deskripsi": item["deskripsi"],
        "harga_per_bulan": item.get("harga_per_bulan"),
        "tipe": item.get("tipe"),
        "fasilitas": item.get("fasilitas") or [],
        "alamat": item.get("alamat"),
        "kecamatan": item.get("kecamatan"),
        "koordinat_lat": lat,
        "koordinat_lng": lng,
        "jarak_kampus_km": item.get("jarak_kampus_km"),
        "url_source": item.get("url_source"),
        "scrape_date": scrape_date,
        "source": item.get("source", "unknown"),
    }


def dedupe_by_id(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Dedupe ORM dicts by 'id' (keep last occurrence).

    TRUNCATE+INSERT butuh id unik dalam satu batch (Postgres tolak id ganda di
    satu VALUES). Real source harusnya sudah unik; ini jaga-jaga.
    """
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_id[row["id"]] = row
    return list(by_id.values())


def rows_digest(rows: "list[dict[str, Any]]") -> str:
    """Digest deterministik isi listing (id|harga|tipe|fasilitas, urut, sha256).

    Dipakai --skip-if-synced: kalau digest DB == digest source, isi tabel
    sudah sinkron dan TRUNCATE+reseed di-skip (hemat waktu cold start).
    Sengaja menyertakan field konten (bukan id saja) supaya perubahan data
    tanpa ganti id (mis. pembersihan fasilitas) tetap memicu reseed di
    deploy berikutnya.
    """
    import hashlib

    lines = sorted(
        "{}|{}|{}|{}".format(
            r.get("id"),
            r.get("harga_per_bulan"),
            r.get("tipe"),
            ",".join(r.get("fasilitas") or []),
        )
        for r in rows
    )
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def build_truncate_plan(
    current_listings: int,
    keep_count: int,
    ground_truth_rows: int,
    consensus_rows: int,
) -> dict[str, int]:
    """Ringkas dampak TRUNCATE listings CASCADE + reseed keep_count rows.

    TRUNCATE CASCADE mengosongkan SELURUH ground_truth + ground_truth_consensus
    (keduanya FK ke listings), bukan cuma baris milik listing yang dibuang.
    """
    return {
        "current_listings": current_listings,
        "final_listings": keep_count,
        "listings_removed_net": max(current_listings - keep_count, 0),
        "ground_truth_cascade_deleted": ground_truth_rows,
        "consensus_cascade_deleted": consensus_rows,
    }


async def upsert_listings(listings_data: list[dict[str, Any]]) -> int:
    """Upsert via PostgreSQL ON CONFLICT DO UPDATE. Return rows affected."""
    if not listings_data:
        return 0

    async with async_session_factory() as session:
        stmt = insert(Listing).values(listings_data)
        # Update semua field selain id kalau conflict
        update_cols = {col: getattr(stmt.excluded, col) for col in listings_data[0] if col != "id"}
        stmt = stmt.on_conflict_do_update(index_elements=["id"], set_=update_cols)
        await session.execute(stmt)
        await session.commit()
    return len(listings_data)


async def collect_counts() -> tuple[int, int, int]:
    """Return (listings, ground_truth, ground_truth_consensus) row counts."""
    async with async_session_factory() as session:
        listings = (
            await session.execute(select(func.count(Listing.id)))
        ).scalar()
        gt = (
            await session.execute(select(func.count()).select_from(GroundTruth))
        ).scalar()
        gtc = (
            await session.execute(
                select(func.count()).select_from(GroundTruthConsensus)
            )
        ).scalar()
    return int(listings or 0), int(gt or 0), int(gtc or 0)


async def truncate_and_seed(listings_data: list[dict[str, Any]]) -> int:
    """TRUNCATE listings (CASCADE) lalu insert ulang, dalam 1 transaksi (atomic).

    Postgres TRUNCATE CASCADE ikut mengosongkan ground_truth +
    ground_truth_consensus (FK ke listings) walau schema tak set ON DELETE
    CASCADE. Aman di production (tabel eval kosong), destruktif di lokal.
    Refuse kalau seed set kosong supaya DB tidak ter-wipe jadi 0.
    """
    if not listings_data:
        raise ValueError("refusing TRUNCATE: seed set kosong (cek --input)")
    async with async_session_factory() as session:
        async with session.begin():
            await session.execute(text("TRUNCATE TABLE listings CASCADE"))
            await session.execute(insert(Listing).values(listings_data))
    return len(listings_data)


def run_preprocessing(
    listings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run preprocessing pipeline on each listing's deskripsi.

    Returns list of {id, text, raw_text, metadata} -- ready untuk indexing.
    """
    from app.preprocessing import PreprocessingPipeline
    from app.preprocessing.doc_text import (
        compose_lexical_text,
        compose_natural_text,
    )

    pipeline = PreprocessingPipeline()
    corpus: list[dict[str, Any]] = []
    for i, listing in enumerate(listings):
        # Fielded text (judul x2 + kecamatan + fasilitas + deskripsi) —
        # konsisten dengan scripts/preprocess_corpus.py
        result = pipeline.process(compose_lexical_text(listing))
        corpus.append(
            {
                "id": listing["id"],
                "text": result.processed,  # for indexing
                "raw_text": compose_natural_text(listing),  # untuk neural
                "metadata": {
                    "judul": listing["judul"],
                    "harga_per_bulan": listing.get("harga_per_bulan"),
                    "tipe": listing.get("tipe"),
                    "fasilitas": listing.get("fasilitas", []),
                    "alamat": listing.get("alamat"),
                    "kecamatan": listing.get("kecamatan"),
                    # v2 real-data fields (extra IR signal)
                    "owner_name": listing.get("owner_name"),
                    "verified": listing.get("verified"),
                    "kampus_terdekat": listing.get("kampus_terdekat"),
                    "url_source": listing.get("url_source"),
                },
            }
        )
        if (i + 1) % 100 == 0:
            logger.info(f"[preprocess] {i+1}/{len(listings)}")
    return corpus


async def main_async(args: argparse.Namespace) -> int:
    logger.info(f"[load] {args.input}")
    raw = parse_jsonl(args.input)
    logger.info(f"[load] {len(raw)} listings raw")

    filtered = filter_listings(raw)
    orm_data = dedupe_by_id([to_orm_dict(item) for item in filtered])
    keep_count = len(orm_data)

    # Ensure schema exists (dev only, prod via Alembic)
    if args.init_db:
        logger.info("[init_db] running create_all (dev only)")
        await init_db()

    # Snapshot DB sekarang (untuk dry-run report + warning pre-truncate)
    current_listings, gt_rows, gtc_rows = await collect_counts()

    # Fast path cold start: kalau DB sudah sinkron per-id dengan source,
    # skip TRUNCATE+reseed (hemat beberapa detik tiap container bangun).
    if args.skip_if_synced and current_listings == keep_count:
        async with async_session_factory() as session:
            db_rows = [
                {
                    "id": row[0], "harga_per_bulan": row[1],
                    "tipe": row[2], "fasilitas": row[3],
                }
                for row in (await session.execute(
                    select(Listing.id, Listing.harga_per_bulan,
                           Listing.tipe, Listing.fasilitas)
                )).all()
            ]
        if rows_digest(db_rows) == rows_digest(orm_data):
            logger.info(
                f"[skip-if-synced] DB sudah sinkron ({keep_count} listings, "
                "digest konten sama) — seed di-skip"
            )
            await engine.dispose()
            return 0
        logger.info("[skip-if-synced] count sama tapi digest beda — lanjut seed")

    if args.dry_run:
        if args.truncate:
            plan = build_truncate_plan(
                current_listings, keep_count, gt_rows, gtc_rows
            )
            logger.info(f"[dry-run] mode=truncate-then-seed plan={plan}")
            if plan["ground_truth_cascade_deleted"] or plan[
                "consensus_cascade_deleted"
            ]:
                logger.warning(
                    f"[dry-run] TRUNCATE CASCADE akan menghapus {gt_rows} "
                    f"ground_truth + {gtc_rows} consensus rows"
                )
        else:
            logger.info(
                f"[dry-run] mode=upsert (non-destructive): current="
                f"{current_listings}, would upsert {keep_count} rows, deletes none"
            )
        logger.info("[dry-run] NO changes written")
        await engine.dispose()
        return 0

    if args.truncate:
        if not orm_data:
            logger.error("[truncate] seed set kosong, ABORT (DB tidak disentuh)")
            await engine.dispose()
            return 1
        logger.warning(
            f"[truncate] wiping {current_listings} listings + CASCADE {gt_rows} "
            f"ground_truth, {gtc_rows} consensus; reseed {keep_count}"
        )
        await truncate_and_seed(orm_data)
        logger.info(f"[truncate] reseeded {keep_count} listings")
    else:
        inserted = await upsert_listings(orm_data)
        logger.info(f"[insert] {inserted} listings upserted")

    # Post-verify: konfirmasi listings_count akhir
    final_listings, _, _ = await collect_counts()
    if final_listings == keep_count:
        logger.info(
            f"[verify] listings_count = {final_listings} (target {keep_count}) OK"
        )
    else:
        logger.warning(
            f"[verify] listings_count = {final_listings} != target {keep_count} "
            "(upsert tidak menghapus row lama; pakai --truncate untuk reconcile)"
        )

    # Optional: preprocess + save corpus.json untuk indexing
    if args.preprocess:
        logger.info("[preprocess] running pipeline")
        corpus = run_preprocessing(filtered)
        if args.corpus_output:
            args.corpus_output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.corpus_output, "w", encoding="utf-8") as f:
                json.dump(corpus, f, ensure_ascii=False, indent=2)
            logger.info(f"[corpus] saved -> {args.corpus_output}")

    await engine.dispose()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed listings DB dari JSONL")
    parser.add_argument(
        "--input", type=Path, required=True, help="JSONL file dari scraper",
    )
    parser.add_argument(
        "--init-db", action="store_true",
        help="Run init_db() create_all dulu (dev only, prod pakai Alembic)",
    )
    parser.add_argument(
        "--truncate", action="store_true",
        help=(
            "TRUNCATE listings (CASCADE) lalu reseed: bikin DB == source persis. "
            "DESTRUKTIF: ikut mengosongkan ground_truth + consensus (aman di prod "
            "yang tabel eval-nya kosong). Tanpa flag = UPSERT (non-destructive)."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Read-only: lapor rencana + jumlah baris terdampak, tanpa menulis ke DB.",
    )
    parser.add_argument(
        "--skip-if-synced", action="store_true",
        help=(
            "Skip seed kalau DB sudah sinkron dengan source (count + digest id "
            "sama). Untuk cold start container; deploy image baru tetap reseed "
            "kalau id berubah."
        ),
    )
    parser.add_argument(
        "--preprocess", action="store_true",
        help="Run preprocessing pipeline + save corpus.json",
    )
    parser.add_argument(
        "--corpus-output", type=Path, default=None,
        help="Output path untuk processed corpus.json (required kalau --preprocess)",
    )
    args = parser.parse_args()

    if args.preprocess and not args.corpus_output:
        parser.error("--preprocess butuh --corpus-output")

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())

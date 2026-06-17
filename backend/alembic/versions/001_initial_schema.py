"""initial schema: listings + queries + ground_truth + eval_results

Revision ID: 001
Revises:
Create Date: 2026-05-26 00:00:00
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # listings: raw + processed kos data
    # ------------------------------------------------------------------
    op.create_table(
        "listings",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("judul", sa.Text(), nullable=False),
        sa.Column("deskripsi", sa.Text(), nullable=False),
        sa.Column("deskripsi_processed", sa.Text(), nullable=True),
        sa.Column("harga_per_bulan", sa.Integer(), nullable=True),
        sa.Column("tipe", sa.String(length=20), nullable=True),
        sa.Column("fasilitas", sa.ARRAY(sa.String()), nullable=True),
        sa.Column("alamat", sa.Text(), nullable=True),
        sa.Column("kecamatan", sa.String(length=100), nullable=True),
        sa.Column("koordinat_lat", sa.Numeric(precision=8, scale=5), nullable=True),
        sa.Column("koordinat_lng", sa.Numeric(precision=8, scale=5), nullable=True),
        sa.Column("jarak_kampus_km", sa.Numeric(precision=5, scale=2), nullable=True),
        sa.Column("url_source", sa.Text(), nullable=True),
        sa.Column("scrape_date", sa.Date(), nullable=True),
        sa.Column("source", sa.String(length=20), nullable=True),
        sa.Column(
            "inserted_at",
            sa.DateTime(),
            server_default=sa.func.now(),
            nullable=True,
        ),
    )

    # Indexes umum
    op.create_index("ix_listings_tipe", "listings", ["tipe"])
    op.create_index("ix_listings_kecamatan", "listings", ["kecamatan"])
    op.create_index("ix_listings_harga", "listings", ["harga_per_bulan"])

    # ------------------------------------------------------------------
    # queries: query set untuk eval
    # ------------------------------------------------------------------
    op.create_table(
        "queries",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("query_text", sa.Text(), nullable=False),
        sa.Column("context", sa.Text(), nullable=True),
        sa.Column("expected_tipe", sa.String(length=20), nullable=True),
    )

    # ------------------------------------------------------------------
    # ground_truth: raw annotation per annotator (3-annotator setup)
    # ------------------------------------------------------------------
    op.create_table(
        "ground_truth",
        sa.Column("query_id", sa.String(), nullable=False),
        sa.Column("listing_id", sa.String(), nullable=False),
        sa.Column("annotator", sa.String(length=50), nullable=False),
        sa.Column("relevance", sa.SmallInteger(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.CheckConstraint("relevance IN (0, 1, 2)", name="ck_relevance_0_1_2"),
        sa.ForeignKeyConstraint(["query_id"], ["queries.id"]),
        sa.ForeignKeyConstraint(["listing_id"], ["listings.id"]),
        sa.PrimaryKeyConstraint("query_id", "listing_id", "annotator"),
    )

    # ------------------------------------------------------------------
    # ground_truth_consensus: hasil consensus 3 annotator (input ke eval runner)
    # ------------------------------------------------------------------
    op.create_table(
        "ground_truth_consensus",
        sa.Column("query_id", sa.String(), nullable=False),
        sa.Column("listing_id", sa.String(), nullable=False),
        sa.Column("relevance", sa.SmallInteger(), nullable=False),
        sa.CheckConstraint("relevance IN (0, 1, 2)", name="ck_consensus_relevance"),
        sa.ForeignKeyConstraint(["query_id"], ["queries.id"]),
        sa.ForeignKeyConstraint(["listing_id"], ["listings.id"]),
        sa.PrimaryKeyConstraint("query_id", "listing_id"),
    )

    # ------------------------------------------------------------------
    # eval_results: per-model per-query metrics
    # ------------------------------------------------------------------
    op.create_table(
        "eval_results",
        sa.Column("model", sa.String(length=20), nullable=False),
        sa.Column("query_id", sa.String(), nullable=False),
        sa.Column("precision_at_5", sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column("precision_at_10", sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column("average_precision", sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column("ndcg_at_10", sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column("reciprocal_rank", sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column(
            "computed_at",
            sa.DateTime(),
            server_default=sa.func.now(),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(["query_id"], ["queries.id"]),
        sa.PrimaryKeyConstraint("model", "query_id"),
    )


def downgrade() -> None:
    op.drop_table("eval_results")
    op.drop_table("ground_truth_consensus")
    op.drop_table("ground_truth")
    op.drop_table("queries")
    op.drop_index("ix_listings_harga", table_name="listings")
    op.drop_index("ix_listings_kecamatan", table_name="listings")
    op.drop_index("ix_listings_tipe", table_name="listings")
    op.drop_table("listings")

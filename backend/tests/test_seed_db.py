"""Unit test seed_db helpers (pure, tanpa DB): dedupe + truncate plan."""
from scripts.seed_db import build_truncate_plan, dedupe_by_id, rows_digest


def _row(id, harga=500000, tipe="putri", fasilitas=None):
    return {"id": id, "harga_per_bulan": harga, "tipe": tipe,
            "fasilitas": fasilitas or ["wifi"]}


def test_rows_digest_order_independent():
    a = [_row("a"), _row("b")]
    assert rows_digest(a) == rows_digest(list(reversed(a)))


def test_rows_digest_detects_content_change_without_id_change():
    # Kasus nyata: pembersihan fasilitas tanpa ganti id harus memicu reseed
    before = [_row("a", fasilitas=["wifi", "0"])]
    after = [_row("a", fasilitas=["wifi"])]
    assert rows_digest(before) != rows_digest(after)
    assert rows_digest(before) != rows_digest([_row("a", harga=600000, fasilitas=["wifi", "0"])])
    assert rows_digest([]) != rows_digest([_row("a")])


def test_dedupe_keeps_last_per_id():
    rows = [
        {"id": "a", "judul": "x1"},
        {"id": "b", "judul": "y"},
        {"id": "a", "judul": "x2"},
    ]
    out = dedupe_by_id(rows)
    assert len(out) == 2
    by_id = {r["id"]: r for r in out}
    assert by_id["a"]["judul"] == "x2"  # last occurrence wins


def test_dedupe_empty():
    assert dedupe_by_id([]) == []


def test_truncate_plan_prod_case():
    # Production: 2227 listings, tabel eval kosong -> buang 2000, 0 GT loss.
    plan = build_truncate_plan(2227, 227, 0, 0)
    assert plan["current_listings"] == 2227
    assert plan["final_listings"] == 227
    assert plan["listings_removed_net"] == 2000
    assert plan["ground_truth_cascade_deleted"] == 0
    assert plan["consensus_cascade_deleted"] == 0


def test_truncate_plan_local_case_warns_gt_loss():
    # Lokal: listings sudah 227 bersih tapi GT terisi -> truncate nuke 450 GT
    # tanpa membuang listing apa pun (alasan jangan jalanin --truncate di lokal).
    plan = build_truncate_plan(227, 227, 450, 90)
    assert plan["listings_removed_net"] == 0
    assert plan["ground_truth_cascade_deleted"] == 450
    assert plan["consensus_cascade_deleted"] == 90

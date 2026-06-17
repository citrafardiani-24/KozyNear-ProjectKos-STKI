"""Eksplorasi 4: stress-test query understanding (parser + pipeline).

Lempar query nyeleneh, lihat apa yang diekstrak parser (understood) dan
apakah hasil tetap kembali. Tujuan: temukan titik patah parser rule-based.

Kategori: normal (kontrol), typo, code-switch ID-EN, over-specified,
minimalis, nonsense, SQL-ish (cek tidak meledak).

Usage: cd backend && python -m scripts.explore_query_stress
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.indexing.bm25 import BM25Index  # noqa: E402
from app.preprocessing import PreprocessingPipeline  # noqa: E402
from app.search.gazetteer import Gazetteer  # noqa: E402
from app.search.pipeline import smart_rank  # noqa: E402
from scripts.eval_smart import load_listings  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

CASES = [
    ("normal",        "kos putri dekat unila wifi murah"),
    ("typo ringan",   "kos putri dket unila wifii murh"),
    ("typo berat",    "kost cewe deket univ lampng ac"),
    ("code-switch",   "boarding house for girls near unila with wifi"),
    ("over-specified","kos putri ac wifi parkir mobil dapur dekat unila murah dibawah 500rb rajabasa kamar mandi dalam"),
    ("minimalis",     "kos"),
    ("harga juta",    "kos campur maksimal 1,5 juta dekat itera"),
    ("nonsense",      "asdfgh qwerty zzz"),
    ("kosong-ish",    "   "),
    ("injection-ish", "kos'; DROP TABLE listings; --"),
]


def main() -> int:
    bm25 = BM25Index.load(ROOT / "data" / "indexes" / "bm25.pkl")
    pipe = PreprocessingPipeline()
    pre = lambda s: pipe.process(s).processed  # noqa: E731
    gz = Gazetteer.load()
    listings = load_listings()

    for label, q in CASES:
        try:
            ranked, understood, relaxed = smart_rank(
                q, bm25, listings, gz, top_k=5, preprocess=pre)
            u = {k: v for k, v in understood.items() if v}
            top = listings[ranked[0][0]].judul[:40] if ranked else "(kosong)"
            print(f"[{label:<15}] hits={len(ranked)} relaxed={relaxed or '-'}")
            print(f"   understood: {u or '(tak ada yang terdeteksi)'}")
            print(f"   top1: {top}")
        except Exception as e:
            print(f"[{label:<15}] !!! ERROR {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

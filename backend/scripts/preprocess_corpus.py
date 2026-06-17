"""Run preprocessing pipeline pada JSONL listings -> corpus.json untuk indexing.

Output schema (per item, list-format):
    {
        "id": str,
        "text": str,         # processed (lowercased, jargon-substituted, stemmed)
        "raw_text": str,     # original deskripsi
        "metadata": {...}    # judul, harga, tipe, fasilitas, alamat, kecamatan
    }

Usage:
    cd backend
    python -m scripts.preprocess_corpus \\
        --input ../data/raw/kozynear_combined.jsonl \\
        --output ../data/processed/corpus.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure backend/ in path (untuk import app.*)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.preprocessing import PreprocessingPipeline  # noqa: E402
from app.preprocessing.doc_text import (  # noqa: E402
    compose_lexical_text,
    compose_natural_text,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run preprocessing pipeline pada JSONL listings")
    parser.add_argument("--input", type=Path, required=True, help="JSONL listings input")
    parser.add_argument("--output", type=Path, required=True, help="corpus.json output")
    args = parser.parse_args()

    print(f"[load] {args.input}")
    listings: list[dict] = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                listings.append(json.loads(line))
    print(f"[load] {len(listings)} listings")

    print("[init] PreprocessingPipeline (Sastrawi factory)...")
    t0 = time.perf_counter()
    pipeline = PreprocessingPipeline()
    print(f"[init] done in {time.perf_counter() - t0:.2f}s")

    print("[process] running pipeline pada semua dokumen...")
    t0 = time.perf_counter()
    corpus: list[dict] = []
    for i, listing in enumerate(listings):
        # Fielded text: judul x2 + kecamatan + fasilitas + deskripsi
        # (lihat app/preprocessing/doc_text.py). Dulu deskripsi-only.
        result = pipeline.process(compose_lexical_text(listing))
        corpus.append(
            {
                "id": listing["id"],
                "text": result.processed,
                "raw_text": compose_natural_text(listing),
                "metadata": {
                    "judul": listing.get("judul"),
                    "harga_per_bulan": listing.get("harga_per_bulan"),
                    "tipe": listing.get("tipe"),
                    "fasilitas": listing.get("fasilitas", []),
                    "alamat": listing.get("alamat"),
                    "kecamatan": listing.get("kecamatan"),
                    "jarak_kampus_km": listing.get("jarak_kampus_km"),
                    "extracted_prices": result.extracted_prices,
                },
            }
        )
        if (i + 1) % 200 == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            print(f"[progress] {i+1}/{len(listings)} ({rate:.1f} docs/s)")

    elapsed = time.perf_counter() - t0
    print(f"[process] done in {elapsed:.1f}s ({len(corpus) / elapsed:.1f} docs/s)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    # Quick stats
    token_counts = [len(item["text"].split()) for item in corpus]
    avg_tokens = sum(token_counts) / len(token_counts)
    print(f"[stats] tokens after preprocessing: "
          f"min={min(token_counts)}, max={max(token_counts)}, avg={avg_tokens:.1f}")
    print(f"[done] {len(corpus)} docs -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

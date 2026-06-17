"""Rebuild canonical corpus dari data v2 (real scraped Mamikos detail pages).

Real-only: synthetic sudah di-drop sejak 2026-05-29 (commit b7527bb) dan file
kozynear_synthetic.jsonl dihapus. Script ini cuma bekerja dengan data real.

Pipeline:
1. Load mamikos_real_v2.jsonl (deskripsi pemilik real)
2. Normalize kecamatan (strip prefix "Kecamatan ")
3. Drop listings dengan deskripsi kosong
4. Save sebagai kozynear_combined.jsonl (canonical), backup lama ke .bak
5. Cleanup eval files: buang row yang reference id tak ada di corpus baru.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]


def normalize_record(r: dict) -> dict:
    """Cleanup quirks dari v2 extractor sebelum masuk canonical corpus."""
    # Strip "Kecamatan " prefix dari kecamatan field
    kec = r.get("kecamatan") or ""
    if isinstance(kec, str):
        kec = re.sub(r"^Kecamatan\s+", "", kec, flags=re.IGNORECASE).strip()
    r["kecamatan"] = kec or None
    return r


def main() -> int:
    v2_path = ROOT / "data" / "raw" / "mamikos_real_v2.jsonl"
    combined_path = ROOT / "data" / "raw" / "kozynear_combined.jsonl"

    print(f"[load] v2 real: {v2_path}")
    v2 = [json.loads(l) for l in v2_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"[load] {len(v2)} v2 records")

    # Normalize + drop empty deskripsi
    cleaned = []
    n_dropped = 0
    for r in v2:
        r = normalize_record(r)
        if not r.get("deskripsi") or not r.get("deskripsi").strip():
            n_dropped += 1
            continue
        cleaned.append(r)
    print(f"[normalize] kept {len(cleaned)}/{len(v2)} (dropped {n_dropped} empty deskripsi)")

    # Real-only corpus (synthetic di-drop; file kozynear_synthetic.jsonl sudah dihapus)
    combined = cleaned
    print(f"[combine] real-only: {len(combined)} records")

    # Backup old combined if not already
    bak = combined_path.with_suffix(".jsonl.v1.bak")
    if combined_path.exists() and not bak.exists():
        shutil.copy(combined_path, bak)
        print(f"[backup] old combined -> {bak}")

    # Write new combined
    with combined_path.open("w", encoding="utf-8") as f:
        for r in combined:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[write] {combined_path} ({len(combined)} records)")

    # Cleanup eval files: remove rows yang reference doc_ids tidak ada di new combined
    new_ids = {r["id"] for r in combined if r.get("id") is not None}
    import pandas as pd
    eval_dir = ROOT / "eval"
    for fname in ["ground_truth.csv", "annotations_annotator_A.csv",
                  "annotations_annotator_B.csv", "annotations_annotator_C.csv"]:
        p = eval_dir / fname
        df = pd.read_csv(p)
        before = len(df)
        # Backup pre-v2 state
        bak2 = p.with_suffix(".csv.preV2.bak")
        if not bak2.exists():
            shutil.copy(p, bak2)
        df_clean = df[df["doc_id"].isin(new_ids)]
        df_clean.to_csv(p, index=False)
        print(f"[eval] {fname}: {before} -> {len(df_clean)} (kept rows with valid doc_id)")

    # annotation_pool
    pool_path = eval_dir / "annotation_pool.json"
    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    bak3 = pool_path.with_suffix(".json.preV2.bak")
    if not bak3.exists():
        shutil.copy(pool_path, bak3)
    if isinstance(pool, dict):
        cleaned_pool = {}
        for qid, docs in pool.items():
            if isinstance(docs, list):
                kept = [d for d in docs if (isinstance(d, str) and d in new_ids)
                        or (isinstance(d, dict) and d.get("doc_id") in new_ids)]
                cleaned_pool[qid] = kept
            else:
                cleaned_pool[qid] = docs
        pool_path.write_text(json.dumps(cleaned_pool, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[pool] cleaned annotation_pool.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())

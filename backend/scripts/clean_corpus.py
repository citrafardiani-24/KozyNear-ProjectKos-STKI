"""Drop dirty listings dari kozynear_combined.jsonl + eval CSVs.

Dirty criteria:
1. Waitlist placeholders: judul mulai dengan "notification Ikut Daftar Tunggu"
2. Physically impossible prices: harga_per_bulan < Rp 200.000
3. Implausible high prices: harga_per_bulan > Rp 6.000.000 (kos di Bandar
   Lampung jarang > 6 juta; basic-fasilitas dengan harga 7-10 juta strong
   signal scraping error)

Output:
  - data/raw/kozynear_combined.jsonl                  (cleaned, replaces in-place)
  - data/raw/kozynear_combined.jsonl.bak              (backup original)
  - data/raw/dropped_dirty_docs.jsonl                 (audit trail)
  - eval/*.csv                                        (filtered, replaces in-place)
  - eval/*.csv.bak                                    (backup)
"""
from __future__ import annotations

import io
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "kozynear_combined.jsonl"
EVAL = ROOT / "eval"

PRICE_MIN = 200_000
PRICE_MAX = 6_000_000


def is_dirty(rec: dict) -> tuple[bool, str]:
    j = rec.get("judul", "") or ""
    if j.lower().startswith("notification ikut daftar tunggu"):
        return True, "waitlist"
    p = rec.get("harga_per_bulan")
    if isinstance(p, (int, float)):
        if p < PRICE_MIN:
            return True, f"low_price ({p})"
        if p > PRICE_MAX:
            return True, f"high_price ({p})"
    return False, ""


def main() -> int:
    print(f"[clean] reading {RAW}")
    records = [json.loads(line) for line in RAW.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"[clean] {len(records)} records loaded")

    clean, dirty = [], []
    reasons: dict[str, int] = {}
    for r in records:
        flag, reason = is_dirty(r)
        if flag:
            dirty.append({**r, "_drop_reason": reason})
            key = reason.split(" ")[0]
            reasons[key] = reasons.get(key, 0) + 1
        else:
            clean.append(r)

    print(f"[clean] dropped {len(dirty)} records by reason: {reasons}")
    print(f"[clean] keeping {len(clean)} records")

    # Backup: selalu tulis snapshot pre-run ke timestamped file sehingga
    # re-run tidak menimpa backup lama dengan data yang sudah bersih.
    import time as _time
    bak_ts = RAW.with_name(RAW.stem + f".bak.{int(_time.time())}.jsonl")
    shutil.copy(RAW, bak_ts)
    print(f"[backup] {bak_ts}")
    # Juga pertahankan .bak canonical (untuk backward-compat) hanya jika belum ada
    bak = RAW.with_suffix(".jsonl.bak")
    if not bak.exists():
        shutil.copy(RAW, bak)

    # Write cleaned
    with RAW.open("w", encoding="utf-8") as f:
        for r in clean:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[write] {RAW} ({len(clean)} records)")

    # Write dropped audit
    dropped_path = ROOT / "data" / "raw" / "dropped_dirty_docs.jsonl"
    with dropped_path.open("w", encoding="utf-8") as f:
        for r in dirty:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[audit] {dropped_path} ({len(dirty)} records)")

    # Filter eval files
    dirty_ids = {r["id"] for r in dirty}
    eval_files = [
        "ground_truth.csv",
        "annotations_annotator_A.csv",
        "annotations_annotator_B.csv",
        "annotations_annotator_C.csv",
    ]
    for fname in eval_files:
        p = EVAL / fname
        df = pd.read_csv(p)
        before = len(df)
        # Backup
        b = p.with_suffix(".csv.bak")
        if not b.exists():
            shutil.copy(p, b)
        mask_keep = ~df["doc_id"].isin(dirty_ids)
        df_clean = df[mask_keep]
        df_clean.to_csv(p, index=False)
        print(f"[eval] {fname}: {before} -> {len(df_clean)} (dropped {before - len(df_clean)})")

    # Filter annotation_pool.json
    pool_path = EVAL / "annotation_pool.json"
    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    bak_pool = pool_path.with_suffix(".json.bak")
    if not bak_pool.exists():
        shutil.copy(pool_path, bak_pool)
    # Pool structure: {query_id: [doc_id, ...]} or similar
    cleaned_pool: dict | list
    if isinstance(pool, dict):
        cleaned_pool = {}
        n_dropped = 0
        for qid, docs in pool.items():
            if isinstance(docs, list):
                kept = [d for d in docs if (isinstance(d, str) and d not in dirty_ids)
                        or (isinstance(d, dict) and d.get("doc_id") not in dirty_ids)]
                n_dropped += len(docs) - len(kept)
                cleaned_pool[qid] = kept
            else:
                cleaned_pool[qid] = docs
        pool_path.write_text(json.dumps(cleaned_pool, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[pool] dropped {n_dropped} entries from annotation_pool.json")
    elif isinstance(pool, list):
        kept = [item for item in pool if item.get("doc_id") not in dirty_ids]
        pool_path.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[pool] {len(pool)} -> {len(kept)} entries")

    # Delete empty placeholder
    empty = ROOT / "data" / "raw" / "mamikos_test.jsonl"
    if empty.exists() and empty.stat().st_size == 0:
        empty.unlink()
        print(f"[delete] {empty} (was 0 bytes)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

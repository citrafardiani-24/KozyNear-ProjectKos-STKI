"""Gerbang validasi batch listing baru SEBELUM masuk corpus canonical.

Prinsip: "jangan sampai lolos yang salah" — setiap record HARUS lolos semua
gate, dan setiap penolakan dicatat beserta alasannya (audit trail), bukan
hilang diam-diam.

Gates:
1. id unik (tidak duplikat dengan corpus existing maupun sesama batch)
2. judul + deskripsi non-kosong (quality bar yang sama dengan corpus lama)
3. harga_per_bulan ada dan masuk akal (100rb <= harga <= 10jt)
4. kecamatan dikenal (17 kecamatan Bandar Lampung + Jati Agung perbatasan
   ITERA); di luar daftar -> TOLAK (salah kota)
5. koordinat: kalau ada tapi di luar bbox Bandar Lampung raya -> di-NULL
   (kebijakan lama untuk koordinat korup), bukan ditolak
6. PII: owner_name di-strip; fasilitas dibersihkan (digit/1-char dibuang)

Output:
- accepted di-APPEND ke data/raw/mamikos_real_v2.jsonl
- audit lengkap -> eval/_audit_new_batch.json (accepted/rejected + alasan)

Usage:
    cd backend
    python -m scripts.validate_new_listings --input ../data/raw/_new_batch_raw.jsonl
    python -m scripts.validate_new_listings --input ... --dry-run   # review dulu
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parents[2]
REAL_V2 = ROOT / "data" / "raw" / "mamikos_real_v2.jsonl"
AUDIT_OUT = ROOT / "eval" / "_audit_new_batch.json"

KECAMATAN_VALID = {
    # 17 kecamatan kota yang ada di corpus + perbatasan ITERA (disclosed)
    "sukarame", "kedaton", "rajabasa", "tanjung karang timur", "sukabumi",
    "tanjung senang", "kemiling", "kedamaian", "langkapura",
    "tanjung karang pusat", "enggal", "labuhan ratu",
    "teluk betung selatan", "way halim", "teluk betung utara", "panjang",
    "tanjung karang barat", "jati agung",
    # kecamatan kota lain yang sah walau belum ada di corpus lama
    "teluk betung barat", "teluk betung timur", "bumi waras",
}
HARGA_MIN, HARGA_MAX = 100_000, 10_000_000
BBOX_LAT = (-5.55, -5.20)
BBOX_LNG = (105.10, 105.40)


def clean_fasilitas(items: list | None) -> list[str]:
    out: list[str] = []
    for f in items or []:
        s = str(f).strip()
        if not s or len(s) < 2 or s.isdigit():
            continue
        if s not in out:
            out.append(s)
    return out


def normalize_kecamatan(kec: str | None) -> str | None:
    if not kec:
        return None
    return re.sub(r"^Kecamatan\s+", "", str(kec), flags=re.IGNORECASE).strip() or None


def validate(record: dict, existing_ids: set[str], batch_ids: set[str]) -> tuple[dict | None, str | None]:
    """Return (cleaned_record, None) kalau lolos, (None, alasan) kalau tolak."""
    rid = record.get("id")
    if not rid:
        return None, "tanpa id"
    if rid in existing_ids:
        return None, "duplikat (sudah ada di corpus)"
    if rid in batch_ids:
        return None, "duplikat (dobel di batch baru)"

    judul = (record.get("judul") or "").strip()
    deskripsi = (record.get("deskripsi") or "").strip()
    if not judul:
        return None, "judul kosong"
    if not deskripsi:
        return None, "deskripsi kosong"

    harga = record.get("harga_per_bulan")
    if not isinstance(harga, int) or not (HARGA_MIN <= harga <= HARGA_MAX):
        return None, f"harga tidak valid: {harga!r}"

    kec = normalize_kecamatan(record.get("kecamatan"))
    if not kec or kec.lower() not in KECAMATAN_VALID:
        return None, f"kecamatan di luar scope Bandar Lampung: {kec!r}"
    record["kecamatan"] = kec

    koord = record.get("koordinat")
    if koord and isinstance(koord, (list, tuple)) and len(koord) == 2:
        lat, lng = koord
        try:
            lat, lng = float(lat), float(lng)
        except (TypeError, ValueError):
            lat = lng = None
        if (lat is None or not (BBOX_LAT[0] <= lat <= BBOX_LAT[1])
                or not (BBOX_LNG[0] <= lng <= BBOX_LNG[1])):
            record["koordinat"] = None  # korup/di luar kota -> null, jangan tebak
    else:
        record["koordinat"] = None

    record.pop("owner_name", None)  # PII tidak boleh masuk lagi
    record["fasilitas"] = clean_fasilitas(record.get("fasilitas"))
    tipe = record.get("tipe")
    if tipe not in ("putra", "putri", "campur"):
        record["tipe"] = None  # tipe tak dikenal -> null, jangan tebak

    return record, None


def main() -> int:
    parser = argparse.ArgumentParser(description="Validasi batch listing baru")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="Audit saja, JANGAN append ke corpus")
    args = parser.parse_args()

    existing_ids = set()
    for line in open(REAL_V2, encoding="utf-8"):
        if line.strip():
            existing_ids.add(json.loads(line)["id"])
    print(f"[load] corpus existing: {len(existing_ids)} id")

    raw = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    print(f"[load] batch baru: {len(raw)} record")

    accepted: list[dict] = []
    rejected: list[dict] = []
    batch_ids: set[str] = set()
    for r in raw:
        cleaned, reason = validate(r, existing_ids, batch_ids)
        if cleaned is not None:
            batch_ids.add(cleaned["id"])
            accepted.append(cleaned)
        else:
            rejected.append({"id": r.get("id"), "judul": (r.get("judul") or "")[:60],
                             "alasan": reason})

    audit = {
        "input": str(args.input),
        "total_batch": len(raw),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "rejected_detail": rejected,
        "accepted_ids": sorted(r["id"] for r in accepted),
    }
    AUDIT_OUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print(f"[audit] {AUDIT_OUT}")
    print(f"[hasil] accepted={len(accepted)} rejected={len(rejected)}")
    for rj in rejected:
        print(f"  TOLAK {rj['id']}: {rj['alasan']}")

    if args.dry_run:
        print("[dry-run] tidak ada yang ditulis ke corpus")
        return 0
    if accepted:
        with open(REAL_V2, "a", encoding="utf-8") as f:
            for r in accepted:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[append] {len(accepted)} record -> {REAL_V2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

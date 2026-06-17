"""Kos-specific jargon dictionary untuk Indonesian preprocessing.

COURSE REQUIREMENT: ≥100 entries domain-specific (drop rubric Preprocessing 15%
kalau cuma pakai pipeline Sastrawi default tanpa custom dict).

Mentor menyediakan ~105 baseline entries di file ini. Tim Anggota B (Preprocessing
Engineer) WAJIB tambah minimal 20-30 lagi dari hasil EDA, terutama dari:
- Slang kos spesifik Bandar Lampung yang ditemukan saat scrape (cek
  `data/raw/mamikos.jsonl` untuk pattern abbreviation baru)
- Variant kecamatan / area UNILA yang belum tercakup
- Brand merchandise / produk yang sering muncul (merek kasur, mereka brand AC)
- Singkatan unik dari listing pengelola tertentu

Format: variant (apa adanya di listing) -> canonical (bentuk standar).
Matching: word-boundary regex, case-insensitive, longest-first ordering.
"""

from __future__ import annotations


# =============================================================================
# Abbreviation umum kos & fasilitas
# =============================================================================
ABBREVIATIONS: dict[str, str] = {
    # Fasilitas kamar mandi
    "kmd": "kamar mandi dalam",
    "km dlm": "kamar mandi dalam",
    "wc dlm": "kamar mandi dalam",
    "km dalam": "kamar mandi dalam",
    "wc dalam": "kamar mandi dalam",
    "km luar": "kamar mandi luar",
    "wc luar": "kamar mandi luar",
    # Fasilitas elektronik
    "ac": "air conditioner",
    "tv": "televisi",
    "wf": "wifi",
    # Harga & periode
    "jt": "juta",
    "rb": "ribu",
    "thn": "tahun",
    "bln": "bulan",
    "blnan": "bulanan",
    "thnan": "tahunan",
    "min": "minimum",
    "maks": "maksimum",
    # General Indonesian abbreviation
    "dlm": "dalam",
    "sblm": "sebelum",
    "ssdh": "sesudah",
    "yg": "yang",
    "krn": "karena",
    "tdk": "tidak",
    "gak": "tidak",
    "ngga": "tidak",
    "engga": "tidak",
    "dr": "dari",
    "dgn": "dengan",
    "utk": "untuk",
    "skr": "sekarang",
    "blm": "belum",
    "udh": "sudah",
    "udah": "sudah",
    "byk": "banyak",
    "sdh": "sudah",
    "jln": "jalan",
    "jl": "jalan",
    "rmh": "rumah",
    "ortu": "orang tua",
    "bsk": "besok",
    "tgl": "tanggal",
    "tgl2": "tanggal",
}

# =============================================================================
# Location variants (Bandar Lampung area sekitar UNILA)
# =============================================================================
LOCATIONS: dict[str, str] = {
    # Kecamatan & area kampus
    "gdg meneng": "gedong meneng",
    "gd meneng": "gedong meneng",
    "rjbs": "rajabasa",
    "rj basa": "rajabasa",
    "sumbro": "sumantri brojonegoro",
    "kdtn": "kedaton",
    "wh": "way halim",
    "lab ratu": "labuhan ratu",
    "tj senang": "tanjung senang",
    "tlb": "teluk betung",
    "tjk": "tanjungkarang",
    "bdl": "bandar lampung",
    "blampung": "bandar lampung",
    # Kampus & landmark
    "unila": "universitas lampung",
    "unyila": "universitas lampung",
    "unl": "universitas lampung",
    "fmipa": "fakultas mipa universitas lampung",
    "ft unila": "fakultas teknik universitas lampung",
    "fkip": "fakultas keguruan universitas lampung",
    "feb": "fakultas ekonomi bisnis universitas lampung",
    "uin": "uin raden intan lampung",
    "polnep": "politeknik negeri lampung",
    "itera": "institut teknologi sumatera",
    "kampus a": "kampus utama",
}

# =============================================================================
# Tipe kos & slang gender
# =============================================================================
TYPE_SLANG: dict[str, str] = {
    "cowo": "putra",
    "cowok": "putra",
    "pria": "putra",
    "laki": "putra",
    "lk": "putra",
    "cewe": "putri",
    "cewek": "putri",
    "wanita": "putri",
    "perempuan": "putri",
    "pr": "putri",
    "mix": "campur",
}

# =============================================================================
# Aturan kos (rules)
# =============================================================================
RULES: dict[str, str] = {
    "jamal": "jam malam",
    "jam mlm": "jam malam",
    "no smoking": "dilarang merokok",
    "no smoke": "dilarang merokok",
    "free": "bebas",
}

# =============================================================================
# Payment & harga terms
# =============================================================================
PAYMENT: dict[str, str] = {
    "dp": "uang muka",
    "uang dp": "uang muka",
    "down payment": "uang muka",
    "deposit": "uang jaminan",
    "all in": "termasuk semua",
    "all-in": "termasuk semua",
    "include": "termasuk",
    "exclude": "tidak termasuk",
}

# =============================================================================
# Bentuk kos (kostan, kontrakan, dll)
# =============================================================================
KOS_FORM: dict[str, str] = {
    "kostan": "kos kosan",
    "kost-kostan": "kos kosan",
    "kost2an": "kos kosan",
    "kost kostan": "kos kosan",
    "rumah kost": "kos kosan",
    "kontrakan": "kos kosan",
    "kos eksekutif": "kos eksklusif",
    "ekslusive": "eksklusif",
    "exclusive": "eksklusif",
    "exclusif": "eksklusif",
    "exklusif": "eksklusif",
    "exklusive": "eksklusif",
    "ready": "tersedia",
    "available": "tersedia",
    "ready stock": "tersedia",
}

# =============================================================================
# Final merged dict
# =============================================================================
KOS_JARGON_DICT: dict[str, str] = {
    **ABBREVIATIONS,
    **LOCATIONS,
    **TYPE_SLANG,
    **RULES,
    **PAYMENT,
    **KOS_FORM,
}


def jargon_count() -> int:
    """Total unique variants di KOS_JARGON_DICT."""
    return len(KOS_JARGON_DICT)


# Rubric requirement
MIN_REQUIRED = 100


if __name__ == "__main__":
    count = jargon_count()
    print(f"KOS_JARGON_DICT size: {count}")
    print(f"Minimum required for rubric: {MIN_REQUIRED}")
    if count < MIN_REQUIRED:
        print(f"WARNING: tambah {MIN_REQUIRED - count} more entries (tim Anggota B)")
    else:
        print("OK: meets rubric requirement")
    print("\nBreakdown:")
    print(f"  ABBREVIATIONS:  {len(ABBREVIATIONS)}")
    print(f"  LOCATIONS:      {len(LOCATIONS)}")
    print(f"  TYPE_SLANG:     {len(TYPE_SLANG)}")
    print(f"  RULES:          {len(RULES)}")
    print(f"  PAYMENT:        {len(PAYMENT)}")
    print(f"  KOS_FORM:       {len(KOS_FORM)}")

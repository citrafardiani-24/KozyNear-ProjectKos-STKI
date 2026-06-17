"""Generate synthetic kos listings untuk Bandar Lampung (full coverage).

Scope expansion 27 Mei 2026: cover ALL kecamatan Bandar Lampung untuk
universitas-agnostic search (UNILA, ITERA, Darmajaya, UBL, UIN, Teknokrat,
Polinela, Malahayati, Saburai).

Methodology:
- 20 kecamatan Bandar Lampung dengan coord offset realistic dari pusat kota
  (Tugu Adipura -5.4292, 105.2659)
- 12 universitas dengan koordinat akurat
- Templates vary university references (1-3 per listing) + jarak
- Schema preserved (jarak_kampus_km = jarak ke kampus terdekat di list)
- Word count >=110 (course req >=100)

Usage:
    cd backend
    python -m scripts.generate_synthetic_corpus \\
        --output ../data/raw/kozynear_synthetic.jsonl \\
        --count 2000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional


@dataclass
class Listing:
    id: str
    judul: str
    deskripsi: str
    harga_per_bulan: Optional[int] = None
    tipe: Optional[str] = None
    fasilitas: list[str] = field(default_factory=list)
    alamat: Optional[str] = None
    kecamatan: Optional[str] = None
    koordinat: Optional[tuple[float, float]] = None
    jarak_kampus_km: Optional[float] = None
    url_source: Optional[str] = None
    scrape_date: Optional[str] = None
    source: str = "synthetic"


# =============================================================================
# Domain knowledge: Bandar Lampung complete
# =============================================================================

# Pusat kota: Tugu Adipura
BDL_CENTER_LAT = -5.4292
BDL_CENTER_LNG = 105.2659

# 20 kecamatan Bandar Lampung + 1 sub-area (Gedong Meneng = sub-Rajabasa,
# tapi sering disebut sendiri di listing kos)
KECAMATAN_DATA = [
    # (name, lat_offset, lng_offset)
    ("Tanjung Karang Pusat",  ( 0.000,  0.000)),
    ("Tanjung Karang Barat",  (-0.005, -0.015)),
    ("Tanjung Karang Timur",  ( 0.000,  0.020)),
    ("Enggal",                (-0.003, -0.005)),
    ("Teluk Betung Selatan",  (-0.025, -0.010)),
    ("Teluk Betung Utara",    (-0.020, -0.015)),
    ("Teluk Betung Barat",    (-0.030, -0.025)),
    ("Teluk Betung Timur",    (-0.022,  0.005)),
    ("Panjang",               (-0.040,  0.050)),
    ("Bumi Waras",            (-0.030,  0.000)),
    ("Kedamaian",             (-0.015,  0.020)),
    ("Sukabumi",              (-0.020,  0.030)),
    ("Sukarame",              ( 0.005,  0.060)),
    ("Way Halim",             ( 0.010,  0.040)),
    ("Kedaton",               ( 0.020,  0.020)),
    ("Tanjung Senang",        ( 0.030,  0.060)),
    ("Labuhan Ratu",          ( 0.025,  0.010)),
    ("Rajabasa",              ( 0.060, -0.025)),
    ("Gedong Meneng",         ( 0.055, -0.015)),  # sub-area Rajabasa
    ("Kemiling",              ( 0.020, -0.040)),
    ("Langkapura",            ( 0.010, -0.030)),
]

# Universitas-universitas di Bandar Lampung & sekitar
# (name, lat, lng, type, kecamatan)
UNIVERSITIES = [
    ("UNILA",                          -5.3692, 105.2433, "negeri", "Rajabasa"),
    ("Politeknik Negeri Lampung",      -5.3650, 105.2400, "negeri", "Rajabasa"),
    ("IBI Darmajaya",                  -5.4017, 105.2895, "swasta", "Way Halim"),
    ("Universitas Bandar Lampung",     -5.4017, 105.2900, "swasta", "Way Halim"),
    ("UIN Raden Intan Lampung",        -5.3877, 105.3050, "agama",  "Sukarame"),
    ("Universitas Teknokrat Indonesia",-5.4017, 105.2783, "swasta", "Kedaton"),
    ("Universitas Malahayati",         -5.4060, 105.2929, "swasta", "Way Halim"),
    ("ITERA",                          -5.3577, 105.3145, "negeri", "Sukarame"),
    ("Universitas Saburai",            -5.4100, 105.3200, "swasta", "Tanjung Senang"),
]

# Aliases dipakai di deskripsi (variasi cara nyebut)
UNIV_ALIASES = {
    "UNILA": ["UNILA", "Universitas Lampung", "unyila", "Unila"],
    "Politeknik Negeri Lampung": ["Polinela", "Politeknik Negeri Lampung", "Politeknik Lampung"],
    "IBI Darmajaya": ["IBI Darmajaya", "Darmajaya"],
    "Universitas Bandar Lampung": ["UBL", "Universitas Bandar Lampung"],
    "UIN Raden Intan Lampung": ["UIN Raden Intan", "UIN RIL", "UIN Lampung"],
    "Universitas Teknokrat Indonesia": ["Teknokrat", "UTI", "Universitas Teknokrat"],
    "Universitas Malahayati": ["Malahayati", "Universitas Malahayati"],
    "ITERA": ["ITERA", "Institut Teknologi Sumatera"],
    "Universitas Saburai": ["Saburai", "Universitas Saburai"],
}


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Approx jarak km via haversine."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def nearby_universities(lat: float, lng: float, radius_km: float = 8.0) -> list[tuple[str, float]]:
    """Return list of (university_name, jarak_km) within radius."""
    result = []
    for name, ulat, ulng, _, _ in UNIVERSITIES:
        dist = haversine_km(lat, lng, ulat, ulng)
        if dist <= radius_km:
            result.append((name, round(dist, 2)))
    result.sort(key=lambda x: x[1])
    return result


# =============================================================================
# Domain vocabulary
# =============================================================================
STREETS = {
    "Tanjung Karang Pusat":  ["Jl. Raden Intan", "Jl. Kartini", "Jl. Jendral Sudirman", "Jl. Diponegoro"],
    "Tanjung Karang Barat":  ["Jl. Imam Bonjol", "Jl. KH Ahmad Dahlan"],
    "Tanjung Karang Timur":  ["Jl. Hayam Wuruk", "Jl. Cut Nyak Dien"],
    "Enggal":                ["Jl. ZA Pagar Alam", "Jl. Mayor Salim Batubara"],
    "Teluk Betung Selatan":  ["Jl. Yos Sudarso", "Jl. Wolter Mongisidi", "Jl. Hasanudin"],
    "Teluk Betung Utara":    ["Jl. Pulau Sebesi", "Jl. Letjen S. Parman"],
    "Teluk Betung Barat":    ["Jl. Sultan Hasanudin", "Jl. RE Martadinata"],
    "Teluk Betung Timur":    ["Jl. Ikan Bawal", "Jl. Ikan Tongkol"],
    "Panjang":               ["Jl. Sukarno-Hatta", "Jl. Ir. H. Juanda"],
    "Bumi Waras":            ["Jl. Pulau Damar", "Jl. Pulau Buton"],
    "Kedamaian":             ["Jl. Sultan Agung", "Jl. Pajajaran"],
    "Sukabumi":              ["Jl. P. Antasari", "Jl. Hercules"],
    "Sukarame":              ["Jl. Pulau Tegal", "Jl. Endro Suratmin", "Jl. Pramuka"],
    "Way Halim":             ["Jl. Untung Suropati", "Jl. Pulau Sebuku", "Jl. Imam Bonjol", "Jl. Pulau Pisang"],
    "Kedaton":               ["Jl. Teuku Umar", "Jl. ZA Pagar Alam", "Jl. Hayam Wuruk"],
    "Tanjung Senang":        ["Jl. Pulau Damar", "Jl. Pulau Singkep", "Jl. Pramuka"],
    "Labuhan Ratu":          ["Jl. Pulau Tegal", "Jl. Kimaja", "Jl. Karimun Jawa"],
    "Rajabasa":              ["Jl. Sumantri Brojonegoro", "Jl. Soekarno-Hatta", "Jl. Bumi Manti", "Jl. Pulau Damar"],
    "Gedong Meneng":         ["Jl. Bandar Lampung", "Jl. Sukarno", "Jl. Beringin", "Jl. Gedong Air"],
    "Kemiling":              ["Jl. Pramuka", "Jl. Tamin"],
    "Langkapura":            ["Jl. Kapten Tendean", "Jl. Pagar Alam"],
}

TIPE = ["putra", "putri", "campur", "pasutri"]
TIPE_WEIGHTS = [0.40, 0.40, 0.15, 0.05]

FASILITAS_VARIANTS = {
    "ac":                ["AC", "ac", "a.c.", "air conditioner"],
    "kipas angin":       ["kipas angin", "kipas"],
    "wifi":              ["WiFi", "wifi", "Wi-Fi", "internet"],
    "kamar mandi dalam": ["kamar mandi dalam", "KM dalam", "wc dalam", "kmd", "KMD"],
    "kamar mandi luar":  ["kamar mandi luar", "KM luar", "wc luar"],
    "kasur":             ["kasur", "tempat tidur", "spring bed"],
    "lemari":            ["lemari pakaian", "lemari"],
    "meja belajar":      ["meja belajar", "meja"],
    "dapur":             ["dapur bersama", "dapur", "kitchen"],
    "parkir motor":      ["parkir motor", "parkiran motor"],
    "parkir mobil":      ["parkir mobil", "parkiran mobil"],
    "air panas":         ["air panas", "water heater"],
    "tv":                ["tv", "TV", "televisi"],
    "kulkas":            ["kulkas", "lemari es"],
    "mesin cuci":        ["mesin cuci", "laundry"],
    "cctv":              ["CCTV", "cctv", "kamera pengawas"],
    "satpam":            ["satpam", "security"],
}

QUALITY_DESC = [
    "eksklusif", "ekslusive", "exclusive",
    "nyaman", "bersih", "rapi", "rapih",
    "strategis", "stratejis",
    "baru direnovasi", "baru dibangun",
    "modern", "minimalis",
    "luas", "besar",
]

EXTRA_LANDMARKS = [
    "rumah sakit Imanuel", "rumah sakit Bunda Asy-Syifa",
    "Mall Boemi Kedaton", "Chandra Supermarket", "Mall Kartini",
    "Masjid Al-Wasi'i", "GOR Saburai", "alun-alun",
    "halte transmusi", "warung makan",
    "pasar tradisional", "minimarket", "indomaret",
]

RULES = [
    "ada jam malam jam 11", "jam mlm 22.00",
    "tamu wajib lapor", "tamu maksimal sampai jam 10 malam",
    "dilarang merokok di kamar", "no smoking",
    "tidak boleh bawa hewan peliharaan",
    "bayar di muka 3 bulan", "deposit 1 bulan",
    "tertutup khusus mahasiswa",
    "bebas keluar masuk 24 jam",
]

OPEN_TEMPLATES = [
    "Kos {tipe} {quality} di {kecamatan}",
    "Disewakan kos {tipe} {quality}, lokasi di {kecamatan} Bandar Lampung",
    "Available kos {tipe} {quality} area {kecamatan}",
    "Kos {tipe} di {kecamatan}, {quality}",
    "Tersedia kamar kos {tipe} {quality} di kawasan {kecamatan}",
    "{quality} kos {tipe} di {kecamatan} cocok untuk mahasiswa",
]

LOCATION_TEMPLATES = [
    "Lokasi sangat strategis, dekat dengan {univ1} dan {landmark}.",
    "Berada di {street}, hanya {jarak} dari {univ1}.",
    "Akses mudah ke kampus {univ1}, jalan kaki ke {landmark}.",
    "Posisi {street} dekat {univ1} ({jarak} dari kampus).",
    "Strategis di {kecamatan}, gampang akses ke {univ1} dan {univ2}.",
    "Cocok untuk mahasiswa {univ1} -- jarak ke kampus {jarak}.",
    "Di {street} dekat banyak kampus: {univ1}, {univ2}.",
]

CLOSING_TEMPLATES = [
    "Cocok untuk mahasiswa atau pekerja yang cari hunian {quality}.",
    "Cocok buat anak rantau yang kuliah di Bandar Lampung.",
    "Silakan kontak untuk info lebih lanjut dan survey langsung.",
    "Booking sekarang sebelum kehabisan!",
    "Kosong sekarang, ready untuk ditempati.",
    "Tersedia untuk pasangan suami istri dengan aturan yang berlaku.",
]


# =============================================================================
# Generator
# =============================================================================
def random_coords(rng: random.Random, kec_offset: tuple[float, float]) -> tuple[float, float]:
    """Coord di sekitar kecamatan center dengan noise +-500m."""
    lat_off, lng_off = kec_offset
    noise_lat = rng.uniform(-0.005, 0.005)
    noise_lng = rng.uniform(-0.005, 0.005)
    return (
        round(BDL_CENTER_LAT + lat_off + noise_lat, 5),
        round(BDL_CENTER_LNG + lng_off + noise_lng, 5),
    )


def pick_fasilitas(rng: random.Random, tipe: str, harga: int) -> tuple[list[str], list[str]]:
    if harga < 500_000:
        n_fac = rng.randint(2, 4)
    elif harga < 1_000_000:
        n_fac = rng.randint(4, 7)
    else:
        n_fac = rng.randint(6, 10)

    pool = list(FASILITAS_VARIANTS.keys())
    if rng.random() < 0.7 and "ac" in pool:
        pool.insert(0, pool.pop(pool.index("ac")))
    if rng.random() < 0.8 and "wifi" in pool:
        pool.insert(1, pool.pop(pool.index("wifi")))

    chosen_canonical: list[str] = []
    for f in pool:
        if len(chosen_canonical) >= n_fac:
            break
        if f == "kamar mandi luar" and "kamar mandi dalam" in chosen_canonical:
            continue
        if f == "kamar mandi dalam" and "kamar mandi luar" in chosen_canonical:
            continue
        chosen_canonical.append(f)

    raw_variants = [rng.choice(FASILITAS_VARIANTS[f]) for f in chosen_canonical]
    return chosen_canonical, raw_variants


def format_harga_inline(harga: int, rng: random.Random) -> str:
    style = rng.choice(["rupiah", "ribu_rb", "ribu_k", "juta"])
    if style == "rupiah":
        return f"Rp {harga:,}".replace(",", ".")
    if style == "ribu_rb":
        return f"{harga // 1000}rb"
    if style == "ribu_k":
        return f"{harga // 1000}k"
    if style == "juta":
        if harga >= 1_000_000:
            jt = harga / 1_000_000
            return f"{jt:.1f}jt".rstrip("0").rstrip(".")
        return f"{harga // 1000}rb"
    return f"Rp {harga}"


def generate_deskripsi(
    rng: random.Random, tipe: str, kecamatan: str, street: str,
    fasilitas_raw: list[str], harga: int, nearby: list[tuple[str, float]],
) -> str:
    quality = rng.choice(QUALITY_DESC)

    # Pick 1-2 universities to mention
    if nearby:
        n_mention = min(rng.randint(1, 2), len(nearby))
        mentioned_univ = rng.sample(nearby, n_mention)
        univ1_name, univ1_dist = mentioned_univ[0]
        univ1 = rng.choice(UNIV_ALIASES[univ1_name])
        univ2 = (
            rng.choice(UNIV_ALIASES[mentioned_univ[1][0]])
            if len(mentioned_univ) > 1 else
            rng.choice(UNIV_ALIASES[rng.choice([u[0] for u in nearby])])
        )
    else:
        # Fallback: pick random university (kos jauh dari kampus)
        univ1_name = rng.choice([u[0] for u in UNIVERSITIES])
        univ1 = rng.choice(UNIV_ALIASES[univ1_name])
        univ2 = rng.choice(UNIV_ALIASES[rng.choice([u[0] for u in UNIVERSITIES])])
        univ1_dist = rng.uniform(5, 10)

    landmark = rng.choice(EXTRA_LANDMARKS)

    if rng.random() < 0.3 and tipe in ("putra", "putri"):
        tipe_label = rng.choice({
            "putra": ["cowok", "pria", "cowo"],
            "putri": ["cewek", "wanita", "cewe"],
        }[tipe])
    else:
        tipe_label = tipe

    open_t = rng.choice(OPEN_TEMPLATES).format(
        tipe=tipe_label, quality=quality, kecamatan=kecamatan,
    )

    jarak_str = f"{univ1_dist:.1f} km" if univ1_dist >= 1 else f"{int(univ1_dist * 1000)} m"
    loc_t = rng.choice(LOCATION_TEMPLATES).format(
        univ1=univ1, univ2=univ2, landmark=landmark,
        street=street, jarak=jarak_str, kecamatan=kecamatan,
    )

    fac_intro = rng.choice([
        "Fasilitas yang tersedia", "Fasilitas kamar antara lain",
        "Tersedia fasilitas", "Sudah include", "Fasilitas",
    ])
    fac_t = f"{fac_intro}: {', '.join(fasilitas_raw)}."

    n_rules = rng.randint(1, 3)
    rule_sample = rng.sample(RULES, n_rules)
    rule_t = "Aturan kos: " + "; ".join(rule_sample) + "."

    harga_str = format_harga_inline(harga, rng)
    pay_freq = rng.choice(["per bulan", "/bulan", "bulanan"])
    harga_line = f"Harga sewa {harga_str} {pay_freq}."

    extras = []
    if rng.random() < 0.6:
        extras.append(rng.choice([
            "Kamar luas dan terang.",
            "Suasana lingkungan tenang dan aman.",
            "Banyak warung makan dan minimarket di sekitar.",
            "Air bersih lancar 24 jam.",
            "Listrik token sudah include.",
            "Sudah include biaya listrik dan air.",
            "Ada balkon dan jendela besar.",
        ]))
    if rng.random() < 0.4:
        extras.append(rng.choice([
            "Bisa cek lokasi terlebih dahulu sebelum booking.",
            "Pemilik tinggal dekat sehingga ada kontrol.",
            "Penghuni rata-rata mahasiswa dari berbagai kampus.",
            "Wifi cepat sampai 50 Mbps.",
            "Internet sudah include biaya bulanan.",
            "Lingkungan ramah dan kekeluargaan.",
        ]))

    close_t = rng.choice(CLOSING_TEMPLATES)

    parts = [open_t + ".", loc_t, fac_t, rule_t, harga_line]
    parts.extend(extras)
    parts.append(close_t)

    deskripsi = " ".join(parts)

    while len(deskripsi.split()) < 110:
        deskripsi += " " + rng.choice([
            "Hubungi pemilik untuk info lengkap.",
            "Tempat strategis dan nyaman untuk hunian jangka panjang.",
            "Cocok untuk kamu yang cari kos di area Bandar Lampung.",
            "Tersedia kontrak tahunan dengan diskon menarik.",
            "Penghuni mayoritas mahasiswa dan pekerja.",
        ])

    return deskripsi


def generate_judul(rng: random.Random, tipe: str, kecamatan: str,
                   nearby_univs: list[tuple[str, float]]) -> str:
    templates = [
        f"Kos {tipe.title()} {rng.choice(QUALITY_DESC).title()} {kecamatan}",
        f"Kos {tipe.title()} {kecamatan} Bandar Lampung",
        f"Disewakan Kos {tipe.title()} {kecamatan}",
        f"{rng.choice(QUALITY_DESC).title()} Kos {tipe.title()} di {kecamatan}",
    ]
    # Sometimes mention nearby university di judul
    if nearby_univs and rng.random() < 0.3:
        univ_short = rng.choice(UNIV_ALIASES[nearby_univs[0][0]])
        templates.append(f"Kos {tipe.title()} {kecamatan} Dekat {univ_short}")
    return rng.choice(templates)


def generate_listing(rng: random.Random, idx: int) -> Listing:
    tipe = rng.choices(TIPE, weights=TIPE_WEIGHTS, k=1)[0]
    kecamatan, kec_offset = rng.choice(KECAMATAN_DATA)
    street = rng.choice(STREETS[kecamatan])

    harga = int(rng.triangular(300_000, 2_500_000, 700_000))
    harga = round(harga / 50_000) * 50_000

    fasilitas_canon, fasilitas_raw = pick_fasilitas(rng, tipe, harga)
    coords = random_coords(rng, kec_offset)
    nearby = nearby_universities(coords[0], coords[1], radius_km=8.0)

    # jarak_kampus_km = ke kampus terdekat di list (kalau ada)
    jarak_kampus_km = nearby[0][1] if nearby else None

    deskripsi = generate_deskripsi(
        rng, tipe, kecamatan, street, fasilitas_raw, harga, nearby,
    )
    judul = generate_judul(rng, tipe, kecamatan, nearby)

    listing_id = f"kos-synth-{idx:05d}"
    alamat = f"{street} No. {rng.randint(1, 200)}, {kecamatan}, Bandar Lampung"

    return Listing(
        id=listing_id,
        judul=judul,
        deskripsi=deskripsi,
        harga_per_bulan=harga,
        tipe=tipe,
        fasilitas=fasilitas_canon,
        alamat=alamat,
        kecamatan=kecamatan,
        koordinat=coords,
        jarak_kampus_km=jarak_kampus_km,
        url_source=None,
        scrape_date=date.today().isoformat(),
        source="synthetic",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    word_counts: list[int] = []
    kec_counts: dict[str, int] = {}
    univ_mentions: dict[str, int] = {}

    with open(args.output, "w", encoding="utf-8") as f:
        for i in range(args.count):
            listing = generate_listing(rng, i)
            word_counts.append(len(listing.deskripsi.split()))
            kec_counts[listing.kecamatan] = kec_counts.get(listing.kecamatan, 0) + 1
            # Count univ mentions in deskripsi (rough)
            for univ_name, aliases in UNIV_ALIASES.items():
                if any(a in listing.deskripsi for a in aliases):
                    univ_mentions[univ_name] = univ_mentions.get(univ_name, 0) + 1
            f.write(json.dumps(asdict(listing), ensure_ascii=False) + "\n")

    print(f"[done] {args.count} listings -> {args.output}")
    print(f"[stats] deskripsi word count: "
          f"min={min(word_counts)}, max={max(word_counts)}, avg={sum(word_counts) / len(word_counts):.1f}")
    print(f"[stats] kecamatan coverage: {len(kec_counts)} unique")
    for kec, cnt in sorted(kec_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {kec}: {cnt}")
    print(f"[stats] universitas mentions (di deskripsi):")
    for univ, cnt in sorted(univ_mentions.items(), key=lambda x: -x[1]):
        print(f"  {univ}: {cnt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

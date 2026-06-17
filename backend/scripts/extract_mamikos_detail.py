"""Extract structured listing data dari halaman detail Mamikos.

Mamikos detail page meng-inject `var detail = {...}` (valid JSON, ~28KB)
yang berisi 146 field termasuk:
- _id, slug, room_title
- area_subdistrict (kecamatan), area_city
- latitude, longitude (REAL koordinat!)
- description, html_description (REAL cerita pemilik)
- price (monthly/yearly/etc), available_room
- gender (0=campur, 1=putra, 2=putri, 3=pasutri — heuristic)
- top_facilities, fac_room, fac_share, fac_bath, fac_park (real fasilitas)
- booking_rules, owner_name, verification_status

Output canonical schema (compatible dengan corpus pipeline existing):
  {
    "id": "mamikos-{_id}",          # REAL Mamikos ID
    "judul": room_title,
    "deskripsi": description,        # REAL owner story (bukan template!)
    "harga_per_bulan": int,
    "tipe": "campur"|"putra"|"putri"|"pasutri",
    "fasilitas": list[str],          # all facility names combined
    "alamat": str|null,
    "kecamatan": str,
    "koordinat": [lat, lng],         # REAL
    "jarak_kampus_km": float|null,   # computed via haversine
    "url_source": full URL,          # canonical
    "scrape_date": "YYYY-MM-DD",
    "source": "mamikos-real-v2",     # marker schema v2
    # New fields (extra signal untuk IR):
    "owner_name": str|null,
    "available_room": int|null,
    "room_size": str|null,           # e.g. "3 x 4 meter"
    "rules": list[str],              # booking_rules + per-tipe rules
    "verified": bool,
  }

Usage:
  cd backend
  python -m scripts.extract_mamikos_detail \\
      --urls ../data/raw/_discovered_slugs.txt \\
      --output ../data/raw/mamikos_real_v2.jsonl \\
      --delay 4
"""
from __future__ import annotations

import argparse
import io
import json
import math
import random
import re
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, Optional

import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


UA_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0 Safari/537.36",
]

# Mamikos gender code mapping (reverse-engineered from sample data).
# Kode 3 ("pasutri") adalah kategori marketing (campur yang boleh berdua),
# bukan tipe gender terpisah — diperlakukan sebagai "campur".
GENDER_MAP = {
    0: "campur",
    1: "putra",
    2: "putri",
    3: "campur",  # pasutri → campur (marketing term, bukan tipe gender DB)
}

# Universities di Bandar Lampung (lat, lng) — sama dengan generate_synthetic_corpus
UNIVERSITIES: list[tuple[str, float, float]] = [
    ("UNILA", -5.3692, 105.2433),
    ("Politeknik Negeri Lampung", -5.3650, 105.2400),
    ("IBI Darmajaya", -5.4017, 105.2895),
    ("Universitas Bandar Lampung", -5.4017, 105.2900),
    ("UIN Raden Intan Lampung", -5.3877, 105.3050),
    ("Universitas Teknokrat Indonesia", -5.4017, 105.2783),
    ("Universitas Malahayati", -5.4060, 105.2929),
    ("ITERA", -5.3577, 105.3145),
    ("Universitas Saburai", -5.4100, 105.3200),
]


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def nearest_university_km(lat: float, lng: float) -> tuple[Optional[str], Optional[float]]:
    best_name, best_d = None, None
    for name, ulat, ulng in UNIVERSITIES:
        d = haversine_km(lat, lng, ulat, ulng)
        if best_d is None or d < best_d:
            best_name, best_d = name, d
    return best_name, (round(best_d, 2) if best_d is not None else None)


def find_balanced(s: str, open_ch: str = "{", close_ch: str = "}") -> int:
    depth = 0
    in_str: Optional[str] = None
    esc = False
    for i, c in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == in_str:
                in_str = None
        else:
            if c in ('"', "'"):
                in_str = c
            elif c == open_ch:
                depth += 1
            elif c == close_ch:
                depth -= 1
                if depth == 0:
                    return i + 1
    return -1


def extract_detail_var(html: str) -> Optional[dict]:
    """Find `var detail = {...}` and parse as JSON."""
    m = re.search(r"var\s+detail\s*=\s*", html)
    if not m:
        return None
    start = m.end()
    while start < len(html) and html[start] in " \t":
        start += 1
    if start >= len(html) or html[start] != "{":
        return None
    end = find_balanced(html[start:])
    if end < 0:
        return None
    obj_text = html[start : start + end]
    try:
        return json.loads(obj_text)
    except json.JSONDecodeError:
        return None


def collect_facilities(detail: dict) -> list[str]:
    """Combine semua facility lists ke flat unique list (lowercased canonical)."""
    items: list[str] = []
    # fac_room/share/bath/park = list of strings; *_icon = list of dicts
    for key in ("fac_room", "fac_share", "fac_bath", "fac_park"):
        vals = detail.get(key) or []
        for v in vals:
            if isinstance(v, str):
                items.append(v.strip().lower())
            elif isinstance(v, dict) and v.get("name"):
                items.append(str(v["name"]).strip().lower())
    # top_facilities = list of dicts
    for v in detail.get("top_facilities") or []:
        if isinstance(v, dict) and v.get("name"):
            items.append(str(v["name"]).strip().lower())
    # Dedup preserving order
    seen = set()
    out = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def collect_rules(detail: dict) -> list[str]:
    """Extract booking_rules + general rules."""
    rules: list[str] = []
    br = detail.get("booking_rules") or {}
    if isinstance(br, dict):
        if br.get("is_bring_child") is False:
            rules.append("tidak boleh bawa anak")
        if br.get("is_married") is False:
            rules.append("tidak untuk pasutri")
    # Some Mamikos pages also include `tag_rule` or `rules` list
    for key in ("rules", "tag_rule", "rule"):
        v = detail.get(key)
        if isinstance(v, list):
            for item in v:
                if isinstance(item, str):
                    rules.append(item.strip().lower())
                elif isinstance(item, dict) and item.get("name"):
                    rules.append(str(item["name"]).strip().lower())
    # Dedup
    return list(dict.fromkeys(rules))


def parse_price_monthly(detail: dict) -> Optional[int]:
    """Extract integer monthly price."""
    # price_title_formats > price_monthly
    ptf = detail.get("price_title_formats") or {}
    pm = ptf.get("price_monthly") if isinstance(ptf, dict) else None
    if isinstance(pm, dict):
        p = pm.get("price")
        if isinstance(p, (int, float)):
            return int(p)
        if isinstance(p, str):
            cleaned = re.sub(r"[^\d]", "", p)
            if cleaned:
                return int(cleaned)
    # Fallback: discount_price or top-level price
    for key in ("price_monthly", "price_int", "harga"):
        v = detail.get(key)
        if isinstance(v, (int, float)):
            return int(v)
    return None


def detail_to_canonical(detail: dict, source_url: str) -> dict:
    """Map Mamikos detail var ke canonical schema corpus pipeline."""
    real_id = detail.get("_id")
    lat = detail.get("latitude")
    lng = detail.get("longitude")
    coords = [lat, lng] if (isinstance(lat, (int, float)) and isinstance(lng, (int, float))) else None
    jarak = None
    nearest = None
    if coords:
        nearest, jarak = nearest_university_km(coords[0], coords[1])

    fasilitas = collect_facilities(detail)
    rules = collect_rules(detail)
    gender_code = detail.get("gender")
    tipe = GENDER_MAP.get(gender_code) if isinstance(gender_code, int) else None

    # Room size from room_type if available
    room_size = None
    rt = detail.get("room_type") or detail.get("room_size")
    if isinstance(rt, dict):
        w = rt.get("width") or rt.get("room_width")
        l = rt.get("length") or rt.get("room_length")
        if w and l:
            room_size = f"{w} x {l} meter"
    elif isinstance(rt, str):
        room_size = rt

    # Verified
    vs = detail.get("verification_status") or {}
    verified = bool(
        vs.get("is_verified_kost")
        or vs.get("is_verified_by_mamikos")
        or vs.get("is_verified_address")
    ) if isinstance(vs, dict) else False

    return {
        "id": f"mamikos-{real_id}" if real_id else None,
        "judul": detail.get("room_title") or detail.get("name_slug"),
        "deskripsi": detail.get("description") or "",
        "harga_per_bulan": parse_price_monthly(detail),
        "tipe": tipe,
        "fasilitas": fasilitas,
        "alamat": (detail.get("address") or "").strip() or None,
        "kecamatan": detail.get("area_subdistrict") or detail.get("area_city"),
        "koordinat": coords,
        "jarak_kampus_km": jarak,
        "kampus_terdekat": nearest,
        "url_source": source_url,
        "scrape_date": date.today().isoformat(),
        "source": "mamikos-real-v2",
        # Extra signal
        "owner_name": detail.get("owner_name"),
        "available_room": detail.get("available_room"),
        "room_size": room_size,
        "rules": rules,
        "verified": verified,
        "view_count": detail.get("view_count"),
    }


def fetch_detail(url: str, rng: random.Random, timeout: int = 20) -> Optional[dict]:
    """HTTP GET + parse var detail. Returns canonical dict or None."""
    headers = {
        "User-Agent": rng.choice(UA_POOL),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "id-ID,id;q=0.9,en;q=0.8",
    }
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        if r.status_code != 200:
            print(f"  [HTTP {r.status_code}] {url}")
            return None
        detail = extract_detail_var(r.text)
        if detail is None:
            print(f"  [no var detail] {url}")
            return None
        return detail_to_canonical(detail, url)
    except Exception as e:
        print(f"  [error {type(e).__name__}] {url}: {e}")
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls", type=Path, required=True, help="Text file dengan satu URL per baris")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--delay", type=float, default=4.0, help="Sleep antar request (detik)")
    parser.add_argument("--max", type=int, default=0, help="Max URLs to process (0=all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    urls = [line.strip() for line in args.urls.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    if args.max > 0:
        urls = urls[: args.max]
    print(f"[start] {len(urls)} URLs, delay={args.delay}s")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load already-scraped URLs dari output yang ada (resume support).
    # Append mode: interrupt tidak menghapus progress.
    already_done: set[str] = set()
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("url_source"):
                    already_done.add(rec["url_source"])
            except json.JSONDecodeError:
                pass
        if already_done:
            print(f"[resume] {len(already_done)} URLs sudah di-scrape, skip")

    urls_todo = [u for u in urls if u not in already_done]
    print(f"[todo] {len(urls_todo)} URLs remaining")

    n_ok = 0
    n_fail = 0
    with args.output.open("a", encoding="utf-8") as f:
        for i, url in enumerate(urls_todo, start=1):
            if i > 1:
                time.sleep(args.delay + rng.uniform(0, 1.5))
            rec = fetch_detail(url, rng)
            if rec and rec.get("id"):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                n_ok += 1
                if i % 10 == 0 or i == len(urls_todo):
                    print(f"[{i}/{len(urls_todo)}] ok={n_ok} fail={n_fail} | {rec['judul'][:50]}")
            else:
                n_fail += 1
                print(f"[{i}/{len(urls_todo)}] FAIL")

    print(f"\n[done] {n_ok} extracted, {n_fail} failed -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

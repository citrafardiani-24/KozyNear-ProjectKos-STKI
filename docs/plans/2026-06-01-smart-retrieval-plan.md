# Smart Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bangun pipeline pencarian kos yang memahami query (gender/harga/fasilitas/lokasi) lalu merangking dengan gabungan BM25 + kedekatan geografis + kecocokan atribut, tanpa model neural di runtime.

**Architecture:** Query Parser (rule-based, reuse jargon dict + price extractor) menghasilkan ParsedQuery. BM25 cari kandidat dari teks sisa. Ranker menggabung skor teks + geo (haversine ke anchor dari gazetteer statis) + atribut, dengan hard filter gender/harga + fallback. Endpoint baru `model=smart`.

**Tech Stack:** Python 3.11, FastAPI, SQLAlchemy async, rank-bm25, pytest + pytest-asyncio. Tanpa fastembed/faiss di runtime.

---

## File Structure

Buat (baru):
- `backend/app/search/__init__.py` — package marker
- `backend/app/search/gazetteer.py` — koordinat kampus+landmark, haversine, lookup anchor
- `backend/app/search/query_parser.py` — ParsedQuery + parse()
- `backend/app/search/ranker.py` — Candidate, fuse(), hard filter + fallback
- `backend/app/search/pipeline.py` — orkestrasi smart_search (BM25 -> DB -> rank)
- `backend/data/gazetteer.json` — data anchor statis (di-commit)
- `backend/scripts/build_gazetteer.py` — bangun gazetteer.json sekali via Nominatim
- `backend/tests/test_gazetteer.py`, `test_query_parser.py`, `test_ranker.py`, `test_smart_pipeline.py`

Modifikasi:
- `backend/app/models/search.py` — tambah `smart` ke Model + field `understood`, `relaxed`
- `backend/app/api/search.py` — branch `model == "smart"` panggil pipeline
- `backend/app/main.py` — load gazetteer ke app.state; gate load neural di balik env
- `backend/app/core/config.py` — setting `enable_neural` (default False)
- `backend/requirements-runtime.txt` — buang fastembed + faiss-cpu
- eval: `backend/app/evaluation/metrics.py` + runner (tambah smart + constraint satisfaction)

---

## Task 1: Gazetteer (haversine + lookup anchor)

**Files:**
- Create: `backend/app/search/__init__.py` (kosong)
- Create: `backend/app/search/gazetteer.py`
- Create: `backend/data/gazetteer.json`
- Test: `backend/tests/test_gazetteer.py`

- [ ] **Step 1: Seed `backend/data/gazetteer.json`** (kampus dari enrich_geo + alias dari jargon LOCATIONS; landmark coords nanti diisi build script)

```json
[
  {"name": "universitas lampung", "lat": -5.3692, "lng": 105.2433, "aliases": ["unila", "unyila", "unl", "fmipa", "ft unila", "fkip", "feb unila"]},
  {"name": "itera", "lat": -5.3577, "lng": 105.3145, "aliases": ["institut teknologi sumatera"]},
  {"name": "uin raden intan", "lat": -5.3877, "lng": 105.3050, "aliases": ["uin"]},
  {"name": "politeknik negeri lampung", "lat": -5.3650, "lng": 105.2400, "aliases": ["polinela", "polnep"]},
  {"name": "universitas teknokrat indonesia", "lat": -5.4017, "lng": 105.2783, "aliases": ["teknokrat"]},
  {"name": "ibi darmajaya", "lat": -5.4017, "lng": 105.2895, "aliases": ["darmajaya"]},
  {"name": "universitas bandar lampung", "lat": -5.4017, "lng": 105.2900, "aliases": ["ubl"]},
  {"name": "universitas malahayati", "lat": -5.4060, "lng": 105.2929, "aliases": ["malahayati"]},
  {"name": "mall boemi kedaton", "lat": -5.3766, "lng": 105.2496, "aliases": ["mbk", "mall bumi kedaton"]},
  {"name": "transmart lampung", "lat": -5.3930, "lng": 105.2620, "aliases": ["transmart"]}
]
```

(Catatan: koordinat kampus diangkat dari `backend/scripts/enrich_geo.py:33-43`. Landmark MBK/Transmart = perkiraan awal; diperbaiki oleh build script Task 1b. Tidak apa untuk unit test karena test pakai entri kampus yang sudah pasti.)

- [ ] **Step 2: Write failing test** `backend/tests/test_gazetteer.py`

```python
from app.search.gazetteer import Gazetteer, haversine_km


def test_haversine_known_distance():
    # UNILA ke ITERA kira-kira 9-11 km
    d = haversine_km(-5.3692, 105.2433, -5.3577, 105.3145)
    assert 8.0 < d < 12.0


def test_lookup_alias_unila():
    gz = Gazetteer.load()
    anchor = gz.lookup("kos murah deket unila")
    assert anchor is not None
    assert anchor.name == "universitas lampung"
    assert anchor.lat == -5.3692


def test_lookup_landmark_mbk():
    gz = Gazetteer.load()
    anchor = gz.lookup("kos deket mbk")
    assert anchor is not None
    assert "boemi kedaton" in anchor.name


def test_lookup_none_when_no_anchor():
    gz = Gazetteer.load()
    assert gz.lookup("kos murah ac") is None
```

- [ ] **Step 3: Run test, expect fail**

Run: `cd backend && python -m pytest tests/test_gazetteer.py -v`
Expected: FAIL (ImportError: no module named app.search.gazetteer)

- [ ] **Step 4: Implement** `backend/app/search/gazetteer.py`

```python
"""Gazetteer anchor (kampus + landmark) untuk geo ranking. Statis, no API runtime."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

_DATA = Path(__file__).resolve().parents[2] / "data" / "gazetteer.json"


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


@dataclass(frozen=True)
class Anchor:
    name: str
    lat: float
    lng: float


class Gazetteer:
    def __init__(self, entries: list[dict]):
        # (alias_or_name, Anchor), diurut terpanjang dulu biar match spesifik menang
        pairs: list[tuple[str, Anchor]] = []
        for e in entries:
            anchor = Anchor(e["name"], float(e["lat"]), float(e["lng"]))
            for key in [e["name"], *e.get("aliases", [])]:
                pairs.append((key.lower(), anchor))
        self._pairs = sorted(pairs, key=lambda p: len(p[0]), reverse=True)

    @classmethod
    def load(cls, path: Path = _DATA) -> "Gazetteer":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def lookup(self, text: str) -> Anchor | None:
        low = f" {text.lower()} "
        for key, anchor in self._pairs:
            if f" {key} " in low or low.strip().endswith(f" {key}") or low.strip().startswith(f"{key} "):
                return anchor
        return None
```

- [ ] **Step 5: Run test, expect pass**

Run: `cd backend && python -m pytest tests/test_gazetteer.py -v`
Expected: PASS (4 passed)

- [ ] **Step 6: Commit**

```bash
git add backend/app/search/__init__.py backend/app/search/gazetteer.py backend/data/gazetteer.json backend/tests/test_gazetteer.py
git commit -m "feat(search): add static gazetteer with haversine + anchor lookup"
```

### Task 1b (opsional, build-time): `backend/scripts/build_gazetteer.py`
Reuse `nominatim_geocode` dari `enrich_geo.py` untuk geocode daftar nama landmark, tulis `data/gazetteer.json`. Jalankan sekali manual, hasilnya di-commit. Tidak dipakai runtime, tidak ada test.

---

## Task 2: Query Parser

**Files:**
- Create: `backend/app/search/query_parser.py`
- Test: `backend/tests/test_query_parser.py`

- [ ] **Step 1: Write failing test**

```python
from app.search.gazetteer import Gazetteer
from app.search.query_parser import parse

GZ = Gazetteer.load()


def test_parse_full_query():
    p = parse("kos cewe ac deket unila murah", GZ)
    assert p.gender == "putri"
    assert "air conditioner" in p.fasilitas
    assert p.anchor is not None and p.anchor.name == "universitas lampung"
    assert p.harga_max == 1_000_000   # heuristik "murah"


def test_parse_explicit_price():
    p = parse("kos putra dekat itera maksimal 800rb", GZ)
    assert p.gender == "putra"
    assert p.harga_max == 800_000


def test_parse_gender_conflict_drops_gender():
    p = parse("kos putra putri campur", GZ)
    assert p.gender is None


def test_parse_plain_query_degrades():
    p = parse("kos bagus nyaman", GZ)
    assert p.gender is None and p.anchor is None and p.harga_max is None
    assert "kos" in p.residual_text
```

- [ ] **Step 2: Run test, expect fail**

Run: `cd backend && python -m pytest tests/test_query_parser.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement** `backend/app/search/query_parser.py`

```python
"""Rule-based query understanding untuk pencarian kos."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.preprocessing.jargon import ABBREVIATIONS, TYPE_SLANG
from app.preprocessing.normalizer import extract_prices_inline
from app.search.gazetteer import Anchor, Gazetteer

MURAH_DEFAULT_MAX = 1_000_000  # heuristik "murah" tanpa angka (tunable)

_GENDER_WORDS = {"putra": "putra", "putri": "putri", "campur": "campur",
                 **TYPE_SLANG}
# Subset fasilitas dari ABBREVIATIONS (ac, wifi, tv, kamar mandi dalam, ...)
_FACILITY_MAP = {k: v for k, v in ABBREVIATIONS.items()
                 if v in {"air conditioner", "wifi", "televisi",
                          "kamar mandi dalam", "kamar mandi luar"}}


@dataclass
class ParsedQuery:
    gender: str | None = None
    harga_min: int | None = None
    harga_max: int | None = None
    fasilitas: list[str] = field(default_factory=list)
    anchor: Anchor | None = None
    residual_text: str = ""
    understood: dict = field(default_factory=dict)


def parse(q: str, gazetteer: Gazetteer) -> ParsedQuery:
    low = q.lower()
    p = ParsedQuery()

    # gender (bentrok -> drop)
    genders = {canon for word, canon in _GENDER_WORDS.items()
               if re.search(rf"\b{re.escape(word)}\b", low)}
    p.gender = next(iter(genders)) if len(genders) == 1 else None

    # fasilitas
    for word, canon in _FACILITY_MAP.items():
        if re.search(rf"\b{re.escape(word)}\b", low) and canon not in p.fasilitas:
            p.fasilitas.append(canon)

    # harga: angka eksplisit, else heuristik "murah"
    prices = extract_prices_inline(low)
    if prices:
        p.harga_max = max(prices)
    elif re.search(r"\bmurah\b", low):
        p.harga_max = MURAH_DEFAULT_MAX

    # anchor lokasi
    p.anchor = gazetteer.lookup(low)

    # residual: buang token gender/fasilitas/harga/anchor alias yang dikenal
    residual = low
    for word in list(_GENDER_WORDS) + list(_FACILITY_MAP):
        residual = re.sub(rf"\b{re.escape(word)}\b", " ", residual)
    residual = re.sub(r"\b(murah|maksimal|max|rp)\b", " ", residual)
    residual = re.sub(r"[\d.,]+\s*(jt|juta|rb|ribu|k)?\b", " ", residual)
    p.residual_text = re.sub(r"\s+", " ", residual).strip()

    p.understood = {
        "gender": p.gender, "harga_max": p.harga_max,
        "fasilitas": p.fasilitas,
        "anchor": p.anchor.name if p.anchor else None,
    }
    return p
```

- [ ] **Step 4: Run test, expect pass**

Run: `cd backend && python -m pytest tests/test_query_parser.py -v`
Expected: PASS (4 passed). Jika `test_parse_full_query` gagal di fasilitas, cek `ABBREVIATIONS["ac"] == "air conditioner"` ([jargon.py:34](../../backend/app/preprocessing/jargon.py)).

- [ ] **Step 5: Commit**

```bash
git add backend/app/search/query_parser.py backend/tests/test_query_parser.py
git commit -m "feat(search): add rule-based query parser (gender/price/facility/anchor)"
```

---

## Task 3: Ranker (fusion + hard filter + fallback)

**Files:**
- Create: `backend/app/search/ranker.py`
- Test: `backend/tests/test_ranker.py`

- [ ] **Step 1: Write failing test**

```python
from app.search.gazetteer import Anchor
from app.search.query_parser import ParsedQuery
from app.search.ranker import Candidate, apply_hard_filter, fuse

UNILA = Anchor("universitas lampung", -5.3692, 105.2433)


def _cand(doc_id, text, tipe, harga, fasilitas, lat, lng):
    return Candidate(doc_id, text, tipe, harga, fasilitas, lat, lng)


def test_geo_boost_orders_near_above_far():
    p = ParsedQuery(anchor=UNILA, residual_text="kos")
    near = _cand("a", 1.0, "putri", 800000, [], -5.3700, 105.2440)  # ~0.1km
    far = _cand("b", 1.0, "putri", 800000, [], -5.4100, 105.3200)   # jauh
    ranked = fuse([far, near], p, weights=(0.4, 0.4, 0.2))
    assert ranked[0][0] == "a"  # near menang walau text_score sama


def test_hard_filter_removes_wrong_gender():
    p = ParsedQuery(gender="putri")
    cands = [_cand("a", 1.0, "putra", 800000, [], None, None),
             _cand("b", 1.0, "putri", 800000, [], None, None)]
    kept, relaxed = apply_hard_filter(cands, p)
    assert [c.doc_id for c in kept] == ["b"]
    assert relaxed == []


def test_fallback_relaxes_when_empty():
    p = ParsedQuery(gender="putri", harga_max=500000)
    cands = [_cand("a", 1.0, "putri", 900000, [], None, None)]  # harga lewat
    kept, relaxed = apply_hard_filter(cands, p)
    assert len(kept) == 1  # diloggarin
    assert "harga" in relaxed
```

- [ ] **Step 2: Run test, expect fail**

Run: `cd backend && python -m pytest tests/test_ranker.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement** `backend/app/search/ranker.py`

```python
"""Fusion ranking: gabung skor teks (BM25) + geo + atribut, + hard filter."""
from __future__ import annotations

from dataclasses import dataclass

from app.search.gazetteer import haversine_km
from app.search.query_parser import ParsedQuery


@dataclass
class Candidate:
    doc_id: str
    text_score: float
    tipe: str | None
    harga: int | None
    fasilitas: list[str] | None
    lat: float | None
    lng: float | None


def _minmax(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.0] * len(values)  # semua sama -> netral
    return [(v - lo) / (hi - lo) for v in values]


def _geo_raw(c: Candidate, p: ParsedQuery) -> float:
    if p.anchor is None or c.lat is None or c.lng is None:
        return 0.0
    return 1.0 / (1.0 + haversine_km(float(c.lat), float(c.lng),
                                     p.anchor.lat, p.anchor.lng))


def _attr_raw(c: Candidate, p: ParsedQuery) -> float:
    if not p.fasilitas:
        return 0.0
    have = set(c.fasilitas or [])
    return sum(1 for f in p.fasilitas if f in have) / len(p.fasilitas)


def fuse(cands: list[Candidate], p: ParsedQuery,
         weights: tuple[float, float, float]) -> list[tuple[str, float]]:
    if not cands:
        return []
    w_text, w_geo, w_attr = weights
    text = _minmax([c.text_score for c in cands])
    geo = _minmax([_geo_raw(c, p) for c in cands])
    attr = _minmax([_attr_raw(c, p) for c in cands])
    scored = [(c.doc_id, w_text * text[i] + w_geo * geo[i] + w_attr * attr[i])
              for i, c in enumerate(cands)]
    return sorted(scored, key=lambda x: x[1], reverse=True)


def apply_hard_filter(cands: list[Candidate], p: ParsedQuery
                      ) -> tuple[list[Candidate], list[str]]:
    """Filter gender + harga. Kalau kosong, longgarin (harga dulu, lalu gender)."""
    def keep(c: Candidate, use_gender: bool, use_harga: bool) -> bool:
        if use_gender and p.gender and c.tipe and c.tipe != p.gender:
            return False
        if use_harga and p.harga_max and c.harga and c.harga > p.harga_max:
            return False
        return True

    relaxed: list[str] = []
    for ug, uh, tag in [(True, True, None), (True, False, "harga"),
                        (False, False, "gender")]:
        kept = [c for c in cands if keep(c, ug, uh)]
        if kept:
            if tag == "harga":
                relaxed = ["harga"]
            elif tag == "gender":
                relaxed = ["harga", "gender"]
            return kept, relaxed
    return [], ["harga", "gender"]
```

- [ ] **Step 4: Run test, expect pass**

Run: `cd backend && python -m pytest tests/test_ranker.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/app/search/ranker.py backend/tests/test_ranker.py
git commit -m "feat(search): add fusion ranker with geo/attr scoring + hard filter fallback"
```

---

## Task 4: Pipeline orchestration (BM25 -> DB -> rank)

**Files:**
- Create: `backend/app/search/pipeline.py`
- Test: `backend/tests/test_smart_pipeline.py`

- [ ] **Step 1: Implement** `backend/app/search/pipeline.py`

```python
"""Orkestrasi smart search: parse -> BM25 kandidat -> fetch DB -> rank."""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.indexing.bm25 import BM25Index
from app.models.listing import Listing, ListingRead
from app.search.gazetteer import Gazetteer
from app.search.query_parser import parse
from app.search.ranker import Candidate, apply_hard_filter, fuse

OVERSHOOT = 5
DEFAULT_WEIGHTS = (0.4, 0.4, 0.2)


async def smart_search(q: str, bm25: BM25Index, session: AsyncSession,
                       gazetteer: Gazetteer, top_k: int = 10,
                       weights=DEFAULT_WEIGHTS):
    parsed = parse(q, gazetteer)
    text_q = parsed.residual_text or q
    hits = bm25.query(text_q, top_k=top_k * OVERSHOOT)
    if not hits:
        return [], parsed.understood, []

    score_map = {h.doc_id: h.score for h in hits}
    rows = (await session.execute(
        select(Listing).where(Listing.id.in_(list(score_map))))).scalars().all()
    by_id = {r.id: r for r in rows}

    cands = [Candidate(
        doc_id=h.doc_id, text_score=score_map[h.doc_id],
        tipe=by_id[h.doc_id].tipe, harga=by_id[h.doc_id].harga_per_bulan,
        fasilitas=by_id[h.doc_id].fasilitas,
        lat=(float(by_id[h.doc_id].koordinat_lat)
             if by_id[h.doc_id].koordinat_lat is not None else None),
        lng=(float(by_id[h.doc_id].koordinat_lng)
             if by_id[h.doc_id].koordinat_lng is not None else None),
    ) for h in hits if h.doc_id in by_id]

    kept, relaxed = apply_hard_filter(cands, parsed)
    ranked = fuse(kept, parsed, weights)[:top_k]

    results = []
    for doc_id, score in ranked:
        r = by_id[doc_id]
        results.append(ListingRead(
            id=r.id, judul=r.judul, deskripsi=r.deskripsi,
            harga_per_bulan=r.harga_per_bulan, tipe=r.tipe,
            fasilitas=r.fasilitas, alamat=r.alamat, kecamatan=r.kecamatan,
            score=score))
    return results, parsed.understood, relaxed
```

- [ ] **Step 2: Write integration test** `backend/tests/test_smart_pipeline.py` (pakai BM25 in-memory + monkeypatch session). Lihat pola test async existing di `backend/tests/`. Minimal: build BM25 dari 2-3 Document, fake session yang balikin Listing objek, assert urutan + relaxed.

```python
import pytest
from app.indexing.base import Document
from app.indexing.bm25 import BM25Index
from app.search.gazetteer import Gazetteer
from app.search.pipeline import smart_search


class _FakeResult:
    def __init__(self, rows): self._rows = rows
    def scalars(self): return self
    def all(self): return self._rows


class _FakeSession:
    def __init__(self, rows): self._rows = rows
    async def execute(self, stmt): return _FakeResult(self._rows)


@pytest.mark.asyncio
async def test_smart_search_ranks_near_first(make_listing):
    bm25 = BM25Index(); bm25.build([
        Document(id="a", text="kos nyaman dekat kampus"),
        Document(id="b", text="kos nyaman dekat kampus")])
    rows = [make_listing("a", tipe="putri", lat=-5.3700, lng=105.2440),
            make_listing("b", tipe="putri", lat=-5.4100, lng=105.3200)]
    results, understood, relaxed = await smart_search(
        "kos putri deket unila", bm25, _FakeSession(rows), Gazetteer.load(), top_k=2)
    assert results[0].id == "a"
    assert understood["anchor"] == "universitas lampung"
```

(Tambah fixture `make_listing` di `backend/tests/conftest.py` yang bikin objek `Listing` dengan field minimal.)

- [ ] **Step 3: Run, expect pass** `cd backend && python -m pytest tests/test_smart_pipeline.py -v`

- [ ] **Step 4: Commit**

```bash
git add backend/app/search/pipeline.py backend/tests/test_smart_pipeline.py backend/tests/conftest.py
git commit -m "feat(search): add smart_search pipeline orchestration"
```

---

## Task 5: Schema + endpoint wiring

**Files:**
- Modify: `backend/app/models/search.py`
- Modify: `backend/app/api/search.py`
- Modify: `backend/app/main.py`, `backend/app/core/config.py`

- [ ] **Step 1: Update `search.py` schema**

```python
Model = Literal["tfidf", "bm25", "indobert", "hybrid", "smart"]

class SearchResponse(BaseModel):
    query: str
    model: Model
    top_k: int
    took_ms: int = Field(..., description="Latency total dalam ms")
    results: list[ListingRead]
    understood: dict = Field(default_factory=dict, description="Atribut yang terdeteksi")
    relaxed: list[str] = Field(default_factory=list, description="Filter yang dilonggarkan")
```

- [ ] **Step 2: Branch `smart` di `app/api/search.py`** (di awal handler, sebelum logika index lama):

```python
    if model == "smart":
        gz = request.app.state.gazetteer
        bm25 = request.app.state.bm25
        if bm25 is None or gz is None:
            raise HTTPException(503, detail={"error": "smart index belum siap"})
        t0 = time.perf_counter()
        results, understood, relaxed = await smart_search(
            q, bm25, session, gz, top_k=top_k)
        return SearchResponse(query=q, model=model, top_k=top_k,
                              took_ms=int((time.perf_counter() - t0) * 1000),
                              results=results, understood=understood, relaxed=relaxed)
```

(import: `from app.search.pipeline import smart_search`)

- [ ] **Step 3: Load gazetteer + setting di startup.** `config.py`: tambah `enable_neural: bool = False`. `main.py` lifespan: `app.state.gazetteer = Gazetteer.load()`; bungkus load indobert/hybrid + prewarm di `if settings.enable_neural:`. Default `model` di endpoint ganti ke `"smart"`.

- [ ] **Step 4: Run full test + manual smoke**

Run: `cd backend && python -m pytest tests/ -q`
Expected: semua hijau. Lalu jalanin lokal: `uvicorn app.main:app` lalu `GET /search?q=kos putri deket unila murah&model=smart` -> cek `understood` + hasil dekat unila di atas.

- [ ] **Step 5: Commit**

```bash
git add backend/app/models/search.py backend/app/api/search.py backend/app/main.py backend/app/core/config.py
git commit -m "feat(search): wire smart model into endpoint + load gazetteer at startup"
```

---

## Task 6: Evaluasi (smart system + Constraint Satisfaction @K)

**Files:**
- Modify: `backend/app/evaluation/metrics.py` (tambah fungsi)
- Modify: eval runner untuk masukkan sistem `smart`
- Create: `eval/queries_constraints.json` (query berkendala + konstrainnya)

- [ ] **Step 1: Tambah metrik** `constraint_satisfaction_at_k` di `metrics.py`:

```python
def constraint_satisfaction_at_k(results, constraints, k=5, max_km=3.0):
    """results: list listing dict (urut). constraints: {gender, harga_max, fasilitas, anchor:(lat,lng)}.
    Return rasio top-k yang memenuhi SEMUA konstrain yang ada."""
    from app.search.gazetteer import haversine_km
    topk = results[:k]
    if not topk:
        return 0.0
    ok = 0
    for r in topk:
        good = True
        if constraints.get("gender") and r.get("tipe") != constraints["gender"]:
            good = False
        if constraints.get("harga_max") and (r.get("harga_per_bulan") or 0) > constraints["harga_max"]:
            good = False
        for f in constraints.get("fasilitas", []):
            if f not in (r.get("fasilitas") or []):
                good = False
        anc = constraints.get("anchor")
        if anc and r.get("lat") is not None:
            if haversine_km(r["lat"], r["lng"], anc[0], anc[1]) > max_km:
                good = False
        ok += int(good)
    return ok / len(topk)
```

- [ ] **Step 2: Test metrik**

```python
def test_constraint_satisfaction_all_pass():
    res = [{"tipe": "putri", "harga_per_bulan": 700000, "fasilitas": ["air conditioner"], "lat": -5.37, "lng": 105.244}]
    c = {"gender": "putri", "harga_max": 1000000, "fasilitas": ["air conditioner"], "anchor": (-5.3692, 105.2433)}
    assert constraint_satisfaction_at_k(res, c, k=5) == 1.0
```

Run: `cd backend && python -m pytest tests/test_evaluation.py -k constraint -v` -> PASS

- [ ] **Step 3: Di notebook eval**, jalankan sistem `smart` + `bm25` di 15 query, lapor DUA tabel: (1) P@5/MAP/NDCG di qrels lama, (2) Constraint Satisfaction @5 di `queries_constraints.json`. Tuning bobot via grid kecil, lapor sebagai indikatif (n=15).

- [ ] **Step 4: Commit**

```bash
git add backend/app/evaluation/metrics.py backend/tests/test_evaluation.py eval/queries_constraints.json
git commit -m "feat(eval): add constraint-satisfaction@k metric + smart system in harness"
```

---

## Task 7: Deploy / RAM (drop neural dari runtime)

**Files:**
- Modify: `backend/requirements-runtime.txt`
- Modify: `backend/app/main.py` (gate neural), `Dockerfile` (komentar)

- [ ] **Step 1:** Hapus `fastembed==0.4.2` dan `faiss-cpu==1.9.0` dari `requirements-runtime.txt` (tetap ada di `requirements.txt` full untuk build index/notebook).
- [ ] **Step 2:** Pastikan `main.py` load indobert/hybrid HANYA jika `settings.enable_neural` (default False) — jadi runtime tidak import faiss/fastembed. Verifikasi `app/indexing/indobert.py` import faiss/fastembed di dalam method (bukan top-level) supaya import modul aman tanpa paket itu.
- [ ] **Step 3:** Update komentar `Dockerfile` (hapus klaim "IndoBERT lazy load"; sekarang neural off di prod).
- [ ] **Step 4: Verifikasi** `cd backend && python -m pytest tests/ -q` hijau, lalu cek `python -c "import app.main"` sukses TANPA fastembed/faiss terinstall (simulasikan runtime).
- [ ] **Step 5: Commit**

```bash
git add backend/requirements-runtime.txt backend/app/main.py Dockerfile
git commit -m "chore(deploy): drop neural from runtime (smart pipeline default), gate behind enable_neural"
```

---

## Self-Review (sudah dijalankan saat menulis)
- Spec coverage: parser/gazetteer/geo/ranker/fallback/endpoint/eval-dua-lensa/RAM semua punya task. OK.
- Placeholder: tidak ada TODO/TBD; kode lengkap di tiap step inti. Task 1b & notebook step bersifat build/manual (ditandai jelas).
- Type consistency: `Anchor`, `ParsedQuery`, `Candidate`, `Gazetteer.lookup`, `fuse(weights tuple)`, `apply_hard_filter -> (kept, relaxed)`, `smart_search(...)` konsisten lintas task.
- Catatan: koordinat landmark di gazetteer.json awal = perkiraan; Task 1b memperbaikinya via Nominatim sebelum dipakai di laporan.

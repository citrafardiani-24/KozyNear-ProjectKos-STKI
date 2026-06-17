"""Comprehensive data quality audit for STKI Kos UNILA project.

Audits:
  - data/raw/*.jsonl       (corpus mentah Mamikos + synthetic)
  - data/processed/corpus.json
  - eval/*.csv             (ground_truth, annotator A/B/C, results)
  - eval/queries.json

Outputs JSON+text report with DQS per dataset and cross-referential checks.
"""
from __future__ import annotations

import io
import json
import math
import statistics
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(r"D:\Project TKI Kos")

# ---------- Domain knowledge ----------
VALID_TIPE = {"putra", "putri", "campur"}
# Bandar Lampung bounding box (rough)
LAT_RANGE = (-5.55, -5.30)
LNG_RANGE = (105.10, 105.40)
# Price plausibility (per month, IDR)
PRICE_MIN, PRICE_MAX = 200_000, 6_000_000
VALID_REL = {0, 1, 2}


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists() or p.stat().st_size == 0:
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


# ---------- Corpus audit ----------
def audit_corpus_records(name: str, records: list[dict]) -> dict:
    if not records:
        return {"name": name, "n": 0, "dqs": 0, "issues": ["EMPTY FILE"]}

    n = len(records)
    fields = [
        "id", "judul", "deskripsi", "harga_per_bulan", "tipe", "fasilitas",
        "alamat", "kecamatan", "koordinat", "jarak_kampus_km",
        "url_source", "scrape_date", "source",
    ]
    nulls = {f: 0 for f in fields}
    type_violations = defaultdict(int)
    issues: list[str] = []

    # Uniqueness
    ids = [r.get("id") for r in records]
    id_counts = Counter(ids)
    dup_ids = [k for k, v in id_counts.items() if v > 1]
    n_missing_id = sum(1 for i in ids if i in (None, ""))

    # Domain validity
    invalid_tipe = []
    invalid_coords = []
    out_of_range_price = []
    invalid_jarak = []
    invalid_date = []
    empty_fasilitas = 0

    # Text statistics
    desc_lens, judul_lens = [], []
    fasilitas_counts = []

    sources = Counter()
    kecamatan_counts = Counter()
    tipe_counts = Counter()
    scrape_dates = Counter()

    for r in records:
        for f in fields:
            v = r.get(f)
            if v is None or v == "" or v == []:
                nulls[f] += 1

        # Types
        if "harga_per_bulan" in r and r["harga_per_bulan"] is not None:
            if not isinstance(r["harga_per_bulan"], (int, float)):
                type_violations["harga_per_bulan"] += 1
            else:
                if not (PRICE_MIN <= r["harga_per_bulan"] <= PRICE_MAX):
                    out_of_range_price.append((r.get("id"), r["harga_per_bulan"]))

        if r.get("tipe") and r["tipe"] not in VALID_TIPE:
            invalid_tipe.append((r.get("id"), r["tipe"]))

        if r.get("koordinat") is not None:
            c = r["koordinat"]
            if (not isinstance(c, list) or len(c) != 2
                    or not (LAT_RANGE[0] <= c[0] <= LAT_RANGE[1])
                    or not (LNG_RANGE[0] <= c[1] <= LNG_RANGE[1])):
                invalid_coords.append((r.get("id"), c))

        if r.get("jarak_kampus_km") is not None:
            j = r["jarak_kampus_km"]
            if not isinstance(j, (int, float)) or j < 0 or j > 50:
                invalid_jarak.append((r.get("id"), j))

        if r.get("scrape_date"):
            d = r["scrape_date"]
            if not (isinstance(d, str) and len(d) == 10 and d[4] == "-" and d[7] == "-"):
                invalid_date.append((r.get("id"), d))
            else:
                scrape_dates[d] += 1

        if r.get("fasilitas") in (None, []):
            empty_fasilitas += 1
        else:
            fasilitas_counts.append(len(r["fasilitas"]))

        if r.get("deskripsi"):
            desc_lens.append(len(r["deskripsi"]))
        if r.get("judul"):
            judul_lens.append(len(r["judul"]))

        sources[r.get("source", "<missing>")] += 1
        if r.get("kecamatan"):
            kecamatan_counts[r["kecamatan"]] += 1
        if r.get("tipe"):
            tipe_counts[r["tipe"]] += 1

    null_pct = {f: safe_div(nulls[f], n) for f in fields}

    # ---------- Scoring ----------
    # Completeness: weighted average across CRITICAL fields (id, judul, deskripsi, harga, tipe, source)
    critical = ["id", "judul", "deskripsi", "harga_per_bulan", "tipe", "source"]
    completeness = 100 * (1 - statistics.mean(null_pct[f] for f in critical))

    # Validity: penalize domain violations
    n_invalid = (len(invalid_tipe) + len(invalid_coords) + len(out_of_range_price)
                 + len(invalid_jarak) + len(invalid_date))
    validity = max(0, 100 * (1 - safe_div(n_invalid, n * 5)))

    # Uniqueness
    uniqueness = max(0, 100 * (1 - safe_div(len(dup_ids) + n_missing_id, n)))

    # Consistency: schema fields all present? type violations?
    consistency = max(0, 100 * (1 - safe_div(sum(type_violations.values()), n * len(fields))))

    # Timeliness: scrape_date present for ≥80% of records
    timeliness = 100 * (1 - null_pct["scrape_date"])

    dqs = round(
        0.30 * completeness + 0.20 * validity + 0.25 * consistency
        + 0.15 * uniqueness + 0.10 * timeliness, 1
    )

    issue_log = []
    for f in fields:
        if null_pct[f] > 0:
            sev = "🔴" if null_pct[f] > 0.3 else "🟡" if null_pct[f] > 0.05 else "🟢"
            issue_log.append(f"{sev} null {f}: {nulls[f]}/{n} ({pct(null_pct[f])})")
    if dup_ids:
        issue_log.append(f"🔴 duplicate IDs: {len(dup_ids)}")
    if invalid_tipe:
        issue_log.append(f"🔴 invalid tipe: {len(invalid_tipe)} (sample: {invalid_tipe[:3]})")
    if invalid_coords:
        issue_log.append(f"🟡 invalid coordinat: {len(invalid_coords)} (outside Bandar Lampung bbox)")
    if out_of_range_price:
        issue_log.append(f"🟡 price out of range [{PRICE_MIN:,}-{PRICE_MAX:,}]: {len(out_of_range_price)}")
    if invalid_jarak:
        issue_log.append(f"🟡 invalid jarak_kampus_km: {len(invalid_jarak)}")
    if invalid_date:
        issue_log.append(f"🟡 invalid scrape_date format: {len(invalid_date)}")
    if empty_fasilitas:
        issue_log.append(f"🟡 fasilitas kosong: {empty_fasilitas} ({pct(safe_div(empty_fasilitas, n))})")

    return {
        "name": name,
        "n": n,
        "dqs": dqs,
        "dim": {
            "completeness": round(completeness, 1),
            "validity": round(validity, 1),
            "consistency": round(consistency, 1),
            "uniqueness": round(uniqueness, 1),
            "timeliness": round(timeliness, 1),
        },
        "null_pct": {f: round(null_pct[f] * 100, 1) for f in fields if null_pct[f] > 0},
        "sources": dict(sources),
        "tipe_dist": dict(tipe_counts),
        "n_kecamatan_unique": len(kecamatan_counts),
        "top_kecamatan": kecamatan_counts.most_common(5),
        "scrape_dates": dict(scrape_dates),
        "duplicate_ids": dup_ids[:5],
        "n_duplicate_ids": len(dup_ids),
        "stats": {
            "judul_len": (min(judul_lens), int(statistics.mean(judul_lens)), max(judul_lens)) if judul_lens else None,
            "desc_len": (min(desc_lens), int(statistics.mean(desc_lens)), max(desc_lens)) if desc_lens else None,
            "fasilitas_count": (min(fasilitas_counts), round(statistics.mean(fasilitas_counts), 1), max(fasilitas_counts)) if fasilitas_counts else None,
        },
        "issues": issue_log,
        "invalid_samples": {
            "tipe": invalid_tipe[:3],
            "coords": invalid_coords[:3],
            "price": out_of_range_price[:3],
            "jarak": invalid_jarak[:3],
            "date": invalid_date[:3],
        },
    }


def cross_check_real_vs_synth(real: list[dict], synth: list[dict]) -> dict:
    """Compare distributions real vs synthetic."""
    def mean_std(xs):
        if len(xs) < 2:
            return None
        return round(statistics.mean(xs), 2), round(statistics.stdev(xs), 2)

    r_prices = [r["harga_per_bulan"] for r in real if isinstance(r.get("harga_per_bulan"), (int, float))]
    s_prices = [r["harga_per_bulan"] for r in synth if isinstance(r.get("harga_per_bulan"), (int, float))]
    r_desc = [len(r["deskripsi"]) for r in real if r.get("deskripsi")]
    s_desc = [len(r["deskripsi"]) for r in synth if r.get("deskripsi")]

    return {
        "n_real": len(real),
        "n_synth": len(synth),
        "ratio_synth_to_real": round(safe_div(len(synth), len(real)), 1),
        "price_real": mean_std(r_prices),
        "price_synth": mean_std(s_prices),
        "desc_len_real": mean_std(r_desc),
        "desc_len_synth": mean_std(s_desc),
    }


# ---------- Eval audit ----------
def audit_annotations(name: str, csv_path: Path, corpus_ids: set[str], query_ids: set[str]) -> dict:
    df = pd.read_csv(csv_path)
    n = len(df)
    issues: list[str] = []

    null_qid = df["query_id"].isna().sum()
    null_did = df["doc_id"].isna().sum()
    null_rel = df["relevance"].isna().sum()

    # Relevance domain
    invalid_rel = ~df["relevance"].isin(list(VALID_REL))
    n_invalid_rel = int(invalid_rel.sum())

    # Referential integrity
    unknown_doc = set(df["doc_id"]) - corpus_ids
    unknown_query = set(df["query_id"]) - query_ids
    n_unknown_doc_rows = int(df["doc_id"].isin(unknown_doc).sum())

    # Duplicates (query, doc) pairs
    dup_pairs = df.duplicated(subset=["query_id", "doc_id"]).sum()

    rel_dist = df["relevance"].value_counts().to_dict()

    # Per-query annotation count
    per_query = df["query_id"].value_counts().to_dict()
    short_queries = {q: c for q, c in per_query.items() if c < 20}

    completeness = 100 * (1 - safe_div(null_qid + null_did + null_rel, n * 3))
    validity = 100 * (1 - safe_div(n_invalid_rel, n))
    integrity = 100 * (1 - safe_div(n_unknown_doc_rows, n))
    uniqueness = 100 * (1 - safe_div(int(dup_pairs), n))

    dqs = round(0.30 * completeness + 0.30 * validity + 0.25 * integrity + 0.15 * uniqueness, 1)

    if null_qid or null_did or null_rel:
        issues.append(f"🔴 null cells: query={null_qid}, doc={null_did}, relevance={null_rel}")
    if n_invalid_rel:
        issues.append(f"🔴 invalid relevance values (not in {{0,1,2}}): {n_invalid_rel}")
    if unknown_doc:
        issues.append(f"🔴 doc_id tidak ada di corpus: {len(unknown_doc)} unique IDs ({n_unknown_doc_rows} rows)")
    if unknown_query:
        issues.append(f"🔴 query_id tidak ada di queries.json: {len(unknown_query)} unique IDs")
    if dup_pairs:
        issues.append(f"🟡 duplikasi pasangan (query_id, doc_id): {int(dup_pairs)}")
    if short_queries:
        issues.append(f"🟡 query dengan anotasi <20: {len(short_queries)} query")

    return {
        "name": name,
        "n_rows": n,
        "dqs": dqs,
        "dim": {
            "completeness": round(completeness, 1),
            "validity": round(validity, 1),
            "integrity": round(integrity, 1),
            "uniqueness": round(uniqueness, 1),
        },
        "rel_dist": {str(k): int(v) for k, v in rel_dist.items()},
        "n_unique_queries": int(df["query_id"].nunique()),
        "n_unique_docs": int(df["doc_id"].nunique()),
        "annotations_per_query": {
            "min": min(per_query.values()),
            "median": int(statistics.median(per_query.values())),
            "max": max(per_query.values()),
        },
        "issues": issues,
    }


def compute_annotator_agreement(a_path: Path, b_path: Path, c_path: Path) -> dict:
    a = pd.read_csv(a_path)[["query_id", "doc_id", "relevance"]].rename(columns={"relevance": "rel_A"})
    b = pd.read_csv(b_path)[["query_id", "doc_id", "relevance"]].rename(columns={"relevance": "rel_B"})
    c = pd.read_csv(c_path)[["query_id", "doc_id", "relevance"]].rename(columns={"relevance": "rel_C"})

    m = a.merge(b, on=["query_id", "doc_id"], how="inner").merge(c, on=["query_id", "doc_id"], how="inner")

    n = len(m)
    # Exact 3-way agreement
    three_way = ((m["rel_A"] == m["rel_B"]) & (m["rel_B"] == m["rel_C"])).sum()
    ab = (m["rel_A"] == m["rel_B"]).sum()
    ac = (m["rel_A"] == m["rel_C"]).sum()
    bc = (m["rel_B"] == m["rel_C"]).sum()

    def cohen_k(r1, r2) -> float:
        from collections import Counter
        labels = sorted({*r1.unique(), *r2.unique()})
        cm = pd.crosstab(r1, r2).reindex(index=labels, columns=labels, fill_value=0)
        total = cm.values.sum()
        if total == 0:
            return 0.0
        p_obs = sum(cm.values[i][i] for i in range(len(labels))) / total
        p_exp = sum((cm.sum(axis=1)[lab] / total) * (cm.sum(axis=0)[lab] / total) for lab in labels)
        if abs(1 - p_exp) < 1e-12:
            return 1.0
        return round((p_obs - p_exp) / (1 - p_exp), 3)

    return {
        "n_common": int(n),
        "three_way_agreement": round(safe_div(int(three_way), n), 3),
        "pairwise_agreement": {
            "AB": round(safe_div(int(ab), n), 3),
            "AC": round(safe_div(int(ac), n), 3),
            "BC": round(safe_div(int(bc), n), 3),
        },
        "cohens_kappa": {
            "AB": cohen_k(m["rel_A"], m["rel_B"]),
            "AC": cohen_k(m["rel_A"], m["rel_C"]),
            "BC": cohen_k(m["rel_B"], m["rel_C"]),
        },
    }


def audit_results(p: Path) -> dict:
    df = pd.read_csv(p)
    issues: list[str] = []
    n = len(df)
    metric_cols = ["p_at_5", "p_at_10", "ap", "ndcg_at_10", "rr"]

    nulls = df[metric_cols].isna().sum().to_dict()
    out_of_range = {c: int(((df[c] < 0) | (df[c] > 1)).sum()) for c in metric_cols}

    per_model = df.groupby("model")[metric_cols].mean().round(3).to_dict()

    if any(nulls.values()):
        issues.append(f"🟡 null pada metric: {nulls}")
    if any(out_of_range.values()):
        issues.append(f"🔴 metric di luar range [0,1]: {out_of_range}")

    return {
        "n_rows": n,
        "models": df["model"].unique().tolist(),
        "queries_per_model": df.groupby("model").size().to_dict(),
        "mean_metrics_per_model": per_model,
        "nulls": nulls,
        "out_of_range": out_of_range,
        "issues": issues,
    }


# ---------- Main ----------
def main() -> None:
    print("=" * 80)
    print("STKI KOS UNILA — DATA QUALITY AUDIT")
    print("=" * 80)

    # === 1. RAW CORPUS (real-only, post drop-synthetic 2026-05-29) ===
    raw = ROOT / "data" / "raw"
    files = {
        "mamikos_real_v2.jsonl": load_jsonl(raw / "mamikos_real_v2.jsonl"),
        "kozynear_combined.jsonl": load_jsonl(raw / "kozynear_combined.jsonl"),
    }

    print("\n--- RAW JSONL CORPUS ---")
    corpus_reports = {}
    for fname, recs in files.items():
        rep = audit_corpus_records(fname, recs)
        corpus_reports[fname] = rep
        print(f"\n[{fname}] n={rep['n']}, DQS={rep['dqs']}")
        if rep["n"] == 0:
            print("  ⚠ FILE KOSONG")
            continue
        print(f"  Dim: {rep['dim']}")
        print(f"  Sources: {rep['sources']}")
        print(f"  Tipe: {rep['tipe_dist']}")
        print(f"  N kecamatan unique: {rep['n_kecamatan_unique']}")
        print(f"  Top kecamatan: {rep['top_kecamatan']}")
        if rep["scrape_dates"]:
            print(f"  Scrape dates: {rep['scrape_dates']}")
        if rep["issues"]:
            print("  Issues:")
            for i in rep["issues"]:
                print(f"    {i}")

    # === 2. PRODUCTION/MAPS READINESS (real-data specific) ===
    print("\n--- PRODUCTION & GOOGLE MAPS READINESS ---")
    real_recs = files["mamikos_real_v2.jsonl"]
    n_real = len(real_recs) or 1
    maps_fields = ["koordinat", "url_source", "owner_name", "kampus_terdekat", "jarak_kampus_km"]
    cross = {"n_real": len(real_recs)}
    for f in maps_fields:
        present = sum(1 for r in real_recs if r.get(f) not in (None, "", []))
        cross[f"{f}_present_pct"] = round(present / n_real * 100, 1)
        sev = "🟢" if present / n_real >= 0.95 else "🟡" if present / n_real >= 0.7 else "🔴"
        print(f"  {sev} {f}: {present}/{len(real_recs)} ({cross[f'{f}_present_pct']}%)")
    # Template-leak check (deskripsi harus REAL owner story, bukan generated)
    tmpl = sum(1 for r in real_recs if "platform Mamikos.com" in (r.get("deskripsi") or ""))
    cross["template_leak"] = tmpl
    print(f"  {'🟢' if tmpl == 0 else '🔴'} template-leak deskripsi (harus 0): {tmpl}/{len(real_recs)}")
    # Verified ratio
    verif = sum(1 for r in real_recs if r.get("verified"))
    cross["verified_pct"] = round(verif / n_real * 100, 1)
    print(f"  Mamikos-verified: {verif}/{len(real_recs)} ({cross['verified_pct']}%)")

    # === 3. PROCESSED CORPUS ===
    print("\n--- PROCESSED CORPUS ---")
    corpus_path = ROOT / "data" / "processed" / "corpus.json"
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    def _flatten_corpus_item(item: dict) -> dict:
        """corpus.json shape = {id, text, raw_text, metadata:{judul,...}}.
        Flatten metadata ke top-level supaya audit_corpus_records akurat."""
        md = item.get("metadata") or {}
        return {
            "id": item.get("id"),
            "judul": md.get("judul"),
            "deskripsi": item.get("raw_text") or item.get("text"),
            "harga_per_bulan": md.get("harga_per_bulan"),
            "tipe": md.get("tipe"),
            "fasilitas": md.get("fasilitas"),
            "alamat": md.get("alamat"),
            "kecamatan": md.get("kecamatan"),
            "koordinat": md.get("koordinat"),
            "jarak_kampus_km": md.get("jarak_kampus_km"),
            "source": md.get("source", "processed"),
            "scrape_date": md.get("scrape_date"),
        }

    if isinstance(corpus, list):
        proc_rep = audit_corpus_records("corpus.json", [_flatten_corpus_item(r) for r in corpus])
    elif isinstance(corpus, dict) and "documents" in corpus:
        proc_rep = audit_corpus_records("corpus.json", corpus["documents"])
    else:
        proc_rep = {"name": "corpus.json", "n": 0, "issues": [f"unknown shape: {type(corpus)}"]}
        if isinstance(corpus, dict):
            proc_rep["keys"] = list(corpus.keys())[:20]
    print(f"  Shape: {type(corpus).__name__}, n={proc_rep.get('n')}")
    if "dqs" in proc_rep:
        print(f"  DQS={proc_rep['dqs']}, dim={proc_rep['dim']}")
        if proc_rep["issues"]:
            for i in proc_rep["issues"]:
                print(f"    {i}")
    else:
        print(f"  Keys: {proc_rep.get('keys')}")

    # === 4. INDEX META ===
    meta_path = ROOT / "data" / "indexes" / "indobert" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    print("\n--- INDOBERT INDEX META ---")
    if isinstance(meta, dict):
        print(f"  Keys: {list(meta.keys())[:15]}")
        if "doc_ids" in meta:
            print(f"  n_docs: {len(meta['doc_ids'])}")
            print(f"  sample doc_ids: {meta['doc_ids'][:3]}")
    elif isinstance(meta, list):
        print(f"  Length: {len(meta)}")
        if meta:
            print(f"  Sample[0] keys: {list(meta[0].keys()) if isinstance(meta[0], dict) else type(meta[0])}")

    # === 5. QUERIES ===
    print("\n--- QUERIES.JSON ---")
    q_data = json.loads((ROOT / "eval" / "queries.json").read_text(encoding="utf-8"))
    queries = q_data["queries"]
    query_ids = {q["id"] for q in queries}
    print(f"  N queries: {len(queries)}")
    print(f"  Has metadata: {'metadata' in q_data}")
    null_ctx = sum(1 for q in queries if not q.get("context"))
    null_exp = sum(1 for q in queries if not q.get("expected_tipe"))
    print(f"  null context: {null_ctx}, null expected_tipe: {null_exp}")
    print(f"  ID range: {sorted(query_ids)}")

    # Build corpus ID set from combined (the canonical index)
    canonical_ids = {r["id"] for r in files["kozynear_combined.jsonl"]}

    # === 6. EVAL FILES ===
    print("\n--- EVAL ANNOTATIONS ---")
    eval_reports = {}
    for name in ["ground_truth.csv", "annotations_annotator_A.csv", "annotations_annotator_B.csv", "annotations_annotator_C.csv"]:
        rep = audit_annotations(name, ROOT / "eval" / name, canonical_ids, query_ids)
        eval_reports[name] = rep
        print(f"\n[{name}] n_rows={rep['n_rows']}, DQS={rep['dqs']}")
        print(f"  Dim: {rep['dim']}")
        print(f"  Rel dist: {rep['rel_dist']}")
        print(f"  n_unique_queries={rep['n_unique_queries']}, n_unique_docs={rep['n_unique_docs']}")
        print(f"  Annot per query: {rep['annotations_per_query']}")
        if rep["issues"]:
            for i in rep["issues"]:
                print(f"    {i}")

    # === 7. INTER-ANNOTATOR AGREEMENT ===
    print("\n--- INTER-ANNOTATOR AGREEMENT ---")
    iaa = compute_annotator_agreement(
        ROOT / "eval" / "annotations_annotator_A.csv",
        ROOT / "eval" / "annotations_annotator_B.csv",
        ROOT / "eval" / "annotations_annotator_C.csv",
    )
    for k, v in iaa.items():
        print(f"  {k}: {v}")

    # === 8. RESULTS ===
    print("\n--- RESULTS ---")
    res = audit_results(ROOT / "eval" / "results.csv")
    print(f"  n_rows: {res['n_rows']}")
    print(f"  models: {res['models']}")
    print(f"  queries per model: {res['queries_per_model']}")
    print(f"  mean metrics per model: {res['mean_metrics_per_model']}")
    if res["issues"]:
        for i in res["issues"]:
            print(f"    {i}")

    # === 9. PIPELINE ALIGNMENT (corpus ↔ index ↔ DB-seed) ===
    print("\n--- PIPELINE ALIGNMENT (corpus ↔ index ↔ seed) ---")
    corpus_ids = {r["id"] for r in corpus} if isinstance(corpus, list) else set()
    meta_ids = set(meta["doc_ids"]) if isinstance(meta, dict) and "doc_ids" in meta else set()
    # DB seed eligibility: real listing dengan deskripsi non-kosong + judul (MIN_DESKRIPSI_WORDS=1)
    seed_eligible = {
        r["id"] for r in real_recs
        if (r.get("deskripsi") or "").split() and r.get("judul")
    }
    align = {
        "corpus_n": len(corpus_ids),
        "index_n": len(meta_ids),
        "seed_eligible_n": len(seed_eligible),
        "corpus_eq_index": corpus_ids == meta_ids,
        "corpus_subset_of_seed": corpus_ids <= seed_eligible,
        "gt_docs_in_corpus": None,
    }
    gt_ids = {r for r in eval_reports["ground_truth.csv"].get("rel_dist", {})} and None
    import pandas as _pd
    gt_doc_ids = set(_pd.read_csv(ROOT / "eval" / "ground_truth.csv")["doc_id"])
    align["gt_docs_in_corpus"] = len(gt_doc_ids & corpus_ids)
    align["gt_docs_total"] = len(gt_doc_ids)
    align["gt_orphans"] = len(gt_doc_ids - corpus_ids)
    print(f"  corpus={align['corpus_n']}, index={align['index_n']}, seed_eligible={align['seed_eligible_n']}")
    print(f"  {'🟢' if align['corpus_eq_index'] else '🔴'} corpus == index order: {align['corpus_eq_index']}")
    print(f"  {'🟢' if align['corpus_subset_of_seed'] else '🔴'} corpus ⊆ seed-eligible (search hydration): {align['corpus_subset_of_seed']}")
    print(f"  {'🟢' if align['gt_orphans'] == 0 else '🔴'} GT doc orphans (not in corpus): {align['gt_orphans']}/{align['gt_docs_total']}")

    # Save JSON
    full = {
        "corpus": corpus_reports,
        "maps_readiness": cross,
        "processed": proc_rep,
        "queries": {"n": len(queries), "ids": sorted(query_ids)},
        "annotations": eval_reports,
        "iaa": iaa,
        "results": res,
        "pipeline_alignment": align,
    }
    out = ROOT / "eval" / "_audit_report.json"
    out.write_text(json.dumps(full, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    print(f"\n✓ Saved full JSON report to: {out}")


if __name__ == "__main__":
    main()

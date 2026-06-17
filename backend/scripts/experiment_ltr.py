"""Learning-to-Rank mini: pelajari bobot fusion dari ground truth.

Pertanyaan: kalau bobot fusion DIPELAJARI (logistic regression pointwise)
alih-alih grid search manual, apakah mengalahkan smart fusion handcrafted?

Setup:
- Sampel = 900 pasangan (query, doc) ber-label dari ground_truth.csv
  (label biner: relevance >= 1).
- Fitur per pasangan (semua sinyal yang juga tersedia di sistem):
  bm25_score, tfidf_cosine, neural_cosine, geo_score (1/(1+km) ke anchor
  query), price_fit, facility_overlap, gender_match, log_view_count,
  log_desc_len.
- Validasi: GroupKFold 5-fold BY QUERY (tidak ada query bocor antar fold).
  Ranking dalam pool judged per query test -> MAP pool-restricted,
  dibandingkan dengan smart fusion + BM25 pada protokol yang sama.

Disclosure: label berasal dari annotator SIMULASI yang heuristiknya
berkorelasi dengan beberapa fitur (gender/harga/fasilitas) -> hasil
dibaca sebagai studi konsep, bukan klaim produksi.

Output: eval/ltr_results.csv + ringkasan console.

Usage:
    cd backend
    python -m scripts.experiment_ltr
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from app.evaluation.metrics import average_precision  # noqa: E402
from app.evaluation.statistical import wilcoxon_signed_rank  # noqa: E402
from app.indexing.loader import load_all_indexes  # noqa: E402
from app.preprocessing import PreprocessingPipeline  # noqa: E402
from app.search.gazetteer import Gazetteer, haversine_km  # noqa: E402
from app.search.pipeline import smart_rank  # noqa: E402
from app.search.query_parser import parse  # noqa: E402
from scripts.eval_smart import load_listings  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "eval" / "ltr_results.csv"

FEATURES = [
    "bm25", "tfidf", "neural", "geo", "price_fit",
    "facility_overlap", "gender_match", "log_views", "log_desc_len",
]


def main() -> int:
    print("[load] indexes + pipeline + listings + GT...")
    idx = load_all_indexes(ROOT / "data" / "indexes", include_neural=True)
    bm25, tfidf, neural = idx["bm25"], idx["tfidf"], idx["indobert"]
    pipeline = PreprocessingPipeline()
    pre = lambda s: pipeline.process(s).processed  # noqa: E731
    gz = Gazetteer.load()
    listings = load_listings()

    # view_count tidak ada di adapter eval -> ambil langsung dari jsonl
    views: dict[str, int] = {}
    for line in open(ROOT / "data" / "raw" / "mamikos_real_v2.jsonl", encoding="utf-8"):
        d = json.loads(line)
        views[d["id"]] = d.get("view_count") or 0

    queries = json.loads((ROOT / "eval" / "queries.json").read_text(encoding="utf-8"))["queries"]
    gt: dict[str, dict[str, int]] = {}
    with open(ROOT / "eval" / "ground_truth.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gt.setdefault(row["query_id"], {})[row["doc_id"]] = int(row["relevance"])

    bm25_pos = {d: i for i, d in enumerate(bm25.doc_ids)}
    tfidf_pos = {d: i for i, d in enumerate(tfidf.doc_ids)}

    from sklearn.metrics.pairwise import cosine_similarity

    # ------------------------------------------------------------------
    # Bangun matriks fitur untuk semua pasangan judged
    # ------------------------------------------------------------------
    X, y, groups, pair_ids = [], [], [], []
    for qi, q in enumerate(queries):
        qid, q_text = q["id"], q["query"]
        rel = gt.get(qid, {})
        if not rel:
            continue
        processed = pre(q_text)
        bm25_scores = bm25.bm25.get_scores(processed.split())
        q_vec = tfidf.vectorizer.transform([processed])
        tfidf_scores = cosine_similarity(q_vec, tfidf.doc_matrix).flatten()
        q_emb = neural.encode_query(q_text)
        neural_map = dict(neural.score_docs(q_emb, list(rel.keys())))
        parsed = parse(q_text, gz)

        for did, r in rel.items():
            row_doc = listings.get(did)
            if row_doc is None:
                continue
            geo = 0.0
            if parsed.anchor and row_doc.koordinat_lat is not None:
                km = haversine_km(float(row_doc.koordinat_lat), float(row_doc.koordinat_lng),
                                  parsed.anchor.lat, parsed.anchor.lng)
                geo = 1.0 / (1.0 + km)
            price_fit = 0.5
            if parsed.harga_max and row_doc.harga_per_bulan:
                price_fit = 1.0 if row_doc.harga_per_bulan <= parsed.harga_max else 0.0
            fac = 0.0
            if parsed.fasilitas:
                have = [f.lower() for f in (row_doc.fasilitas or [])]
                fac = sum(1 for kw in parsed.fasilitas
                          if any(kw in f for f in have)) / len(parsed.fasilitas)
            gender = 0.5
            if parsed.gender and row_doc.tipe:
                gender = 1.0 if row_doc.tipe == parsed.gender else 0.0

            X.append([
                float(bm25_scores[bm25_pos[did]]) if did in bm25_pos else 0.0,
                float(tfidf_scores[tfidf_pos[did]]) if did in tfidf_pos else 0.0,
                float(neural_map.get(did, 0.0)),
                geo, price_fit, fac, gender,
                math.log1p(views.get(did, 0)),
                math.log1p(len((row_doc.deskripsi or "").split())),
            ])
            y.append(1 if r >= 1 else 0)
            groups.append(qi)
            pair_ids.append((qid, did))

    X = np.asarray(X); y = np.asarray(y); groups = np.asarray(groups)
    print(f"[data] {len(y)} pasangan, positif {int(y.sum())} ({y.mean():.1%})")

    # ------------------------------------------------------------------
    # GroupKFold by query: train LR -> rank pool query test -> AP
    # ------------------------------------------------------------------
    qid_of_group = {qi: q["id"] for qi, q in enumerate(queries)}
    ap_ltr: dict[str, float] = {}
    coefs = []
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, groups)):
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        # Standarisasi ringan per-fold (fit di train saja)
        mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0) + 1e-9
        clf.fit((X[tr] - mu) / sd, y[tr])
        coefs.append(clf.coef_[0] / sd)  # de-standarisasi untuk interpretasi
        scores = clf.predict_proba((X[te] - mu) / sd)[:, 1]
        per_q: dict[str, list[tuple[str, float, int]]] = {}
        for i, s in zip(te, scores):
            qid, did = pair_ids[i]
            per_q.setdefault(qid, []).append((did, float(s), y[i]))
        for qid, items in per_q.items():
            ranked = [d for d, _, _ in sorted(items, key=lambda t: -t[1])]
            rel_set = {d for d, _, r in items if r == 1}
            ap_ltr[qid] = average_precision(ranked, rel_set)

    # Baseline pada protokol pool yang sama: BM25 + smart fusion
    ap_bm25, ap_smart = {}, {}
    for q in queries:
        qid = q["id"]
        rel = gt.get(qid, {})
        pool = [d for d in rel if d in listings]
        if not pool:
            continue
        processed = pre(q["query"])
        scores = bm25.bm25.get_scores(processed.split())
        ranked = sorted(pool, key=lambda d: -scores[bm25_pos.get(d, 0)])
        rel_set = {d for d, r in rel.items() if r >= 1}
        ap_bm25[qid] = average_precision(ranked, rel_set)

        pool_listings = {d: listings[d] for d in pool}
        ranked_s, _, _ = smart_rank(q["query"], bm25, pool_listings, gz,
                                    top_k=len(pool), preprocess=pre)
        pred = [d for d, _ in ranked_s]
        pred += [d for d in pool if d not in set(pred)]
        ap_smart[qid] = average_precision(pred, rel_set)

    qids = sorted(ap_ltr)
    map_ltr = sum(ap_ltr[q] for q in qids) / len(qids)
    map_bm = sum(ap_bm25[q] for q in qids) / len(qids)
    map_sm = sum(ap_smart[q] for q in qids) / len(qids)
    print(f"\n[MAP pool-restricted, 5-fold by query, n={len(qids)}]")
    print(f"  LTR (logreg 9 fitur) : {map_ltr:.4f}")
    print(f"  Smart fusion (manual): {map_sm:.4f}")
    print(f"  BM25                 : {map_bm:.4f}")
    t1 = wilcoxon_signed_rank([ap_ltr[q] for q in qids], [ap_smart[q] for q in qids])
    t2 = wilcoxon_signed_rank([ap_ltr[q] for q in qids], [ap_bm25[q] for q in qids])
    print(f"  LTR vs smart: {t1}")
    print(f"  LTR vs bm25 : {t2}")

    mean_coef = np.mean(coefs, axis=0)
    order = np.argsort(-np.abs(mean_coef))
    print("\n[bobot ter-pelajari, rata-rata 5 fold, urut |koef|]")
    for i in order:
        print(f"  {FEATURES[i]:<17} {mean_coef[i]:+.3f}")

    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "ap_ltr", "ap_smart", "ap_bm25"])
        for qid in qids:
            w.writerow([qid, f"{ap_ltr[qid]:.4f}", f"{ap_smart[qid]:.4f}",
                        f"{ap_bm25[qid]:.4f}"])
    print(f"\n[saved] {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

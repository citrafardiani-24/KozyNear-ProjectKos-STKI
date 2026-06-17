"""Grid search bobot fusion smart (w_text, w_geo, w_attr).

Metric utama: Constraint-Satisfaction@5 (apa yang dioptimalkan smart,
bebas qrels). Guard: MAP standard (jangan sampai relevansi teks anjlok).
Grid: simplex step 0.1 (w_text + w_geo + w_attr = 1.0), w_text >= 0.1
supaya sinyal teks tidak hilang total.

Output: eval/smart_weights_grid.csv + ringkasan top kombinasi di console.

Usage:
    cd backend
    python -m scripts.experiment_smart_weights
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.evaluation.metrics import average_precision, constraint_satisfaction_at_k  # noqa: E402
from app.indexing.bm25 import BM25Index  # noqa: E402
from app.preprocessing import PreprocessingPipeline  # noqa: E402
from app.search.gazetteer import Gazetteer  # noqa: E402
from app.search.pipeline import smart_rank  # noqa: E402
from scripts.eval_smart import load_ground_truth, load_listings  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT_CSV = ROOT / "eval" / "smart_weights_grid.csv"


def simplex_grid(step: float = 0.1, min_text: float = 0.1) -> list[tuple[float, float, float]]:
    combos: list[tuple[float, float, float]] = []
    n = round(1.0 / step)
    for i in range(round(min_text / step), n + 1):
        for j in range(0, n - i + 1):
            k = n - i - j
            combos.append((round(i * step, 1), round(j * step, 1), round(k * step, 1)))
    return combos


def main() -> int:
    print("[load] bm25 + pipeline + gazetteer + listings + GT + constraints...")
    bm25 = BM25Index.load(ROOT / "data" / "indexes" / "bm25.pkl")
    pipeline = PreprocessingPipeline()
    preprocess = lambda s: pipeline.process(s).processed  # noqa: E731
    gz = Gazetteer.load()
    listings = load_listings()
    gt = load_ground_truth()
    queries = json.loads((ROOT / "eval" / "queries.json").read_text(encoding="utf-8"))["queries"]
    cqueries = json.loads((ROOT / "eval" / "queries_constraints.json").read_text(encoding="utf-8"))

    def to_dict(doc_id: str) -> dict:
        r = listings[doc_id]
        return {"tipe": r.tipe, "harga_per_bulan": r.harga_per_bulan,
                "fasilitas": r.fasilitas, "lat": r.koordinat_lat, "lng": r.koordinat_lng}

    rows = []
    for weights in simplex_grid():
        # CS@5 atas constraint queries
        cs_vals: list[float] = []
        for cq in cqueries:
            constraints = dict(cq["constraints"])
            if constraints.get("anchor"):
                constraints["anchor"] = tuple(constraints["anchor"])
            ranked, _, _ = smart_rank(cq["query"], bm25, listings, gz,
                                      top_k=5, weights=weights, preprocess=preprocess)
            docs = [to_dict(d) for d, _ in ranked if d in listings]
            cs_vals.append(constraint_satisfaction_at_k(docs, constraints, k=5))
        mean_cs = sum(cs_vals) / len(cs_vals)

        # Guard: MAP standard
        aps: list[float] = []
        for q in queries:
            ranked, _, _ = smart_rank(q["query"], bm25, listings, gz,
                                      top_k=10, weights=weights, preprocess=preprocess)
            rel_set = {d for d, r in gt.get(q["id"], {}).items() if r >= 1}
            aps.append(average_precision([d for d, _ in ranked], rel_set))
        mean_map = sum(aps) / len(aps)

        rows.append({"w_text": weights[0], "w_geo": weights[1], "w_attr": weights[2],
                     "cs_at_5": round(mean_cs, 4), "map_standard": round(mean_map, 4)})

    rows.sort(key=lambda r: (-r["cs_at_5"], -r["map_standard"]))
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["w_text", "w_geo", "w_attr", "cs_at_5", "map_standard"])
        w.writeheader()
        w.writerows(rows)

    current = next(r for r in rows if (r["w_text"], r["w_geo"], r["w_attr"]) == (0.4, 0.4, 0.2))
    print(f"\n[saved] {OUT_CSV} ({len(rows)} kombinasi)")
    print("\nTop 8 by CS@5 (guard MAP standard):")
    print(f"{'w_text':>7} {'w_geo':>6} {'w_attr':>7} {'CS@5':>7} {'MAP':>7}")
    for r in rows[:8]:
        print(f"{r['w_text']:>7} {r['w_geo']:>6} {r['w_attr']:>7} "
              f"{r['cs_at_5']:>7} {r['map_standard']:>7}")
    print(f"\nDefault sekarang (0.4/0.4/0.2): CS@5={current['cs_at_5']} MAP={current['map_standard']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

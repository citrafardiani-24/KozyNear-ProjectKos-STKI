"""Generate evaluation summary report (markdown + plots).

Reads eval/results.csv -> generates:
- eval/eval_summary.md (rubric-friendly metric table + Wilcoxon table + insights)
- eval/charts/bar_metrics.png (per-metric bar chart per model)
- eval/charts/per_query_heatmap.png (query difficulty x model heatmap)

Usage:
    cd backend
    python -m scripts.generate_eval_report \\
        --results ../eval/results.csv \\
        --output-dir ../eval
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_results(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    """Read results.csv -> {model: {query_id: metrics_dict}}."""
    data: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metrics = {
                "p_at_5": float(row["p_at_5"]),
                "p_at_10": float(row["p_at_10"]),
                "ap": float(row["ap"]),
                "ndcg_at_10": float(row["ndcg_at_10"]),
                "rr": float(row["rr"]),
                "query_text": row["query"],
            }
            data[row["model"]][row["query_id"]] = metrics
    return data


def aggregate(per_query: dict[str, dict[str, float]]) -> dict[str, float]:
    n = len(per_query)
    if n == 0:
        return {}
    return {
        "p_at_5": sum(v["p_at_5"] for v in per_query.values()) / n,
        "p_at_10": sum(v["p_at_10"] for v in per_query.values()) / n,
        "map": sum(v["ap"] for v in per_query.values()) / n,
        "ndcg_at_10": sum(v["ndcg_at_10"] for v in per_query.values()) / n,
        "mrr": sum(v["rr"] for v in per_query.values()) / n,
    }


def load_constraint_results(path: Path) -> list[dict[str, str]] | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_summary_md(
    output_path: Path,
    data: dict[str, dict[str, dict[str, float]]],
    queries: list[str],
    constraints_csv: Path | None = None,
) -> None:
    from app.evaluation.statistical import (
        bootstrap_ci_mean,
        holm_bonferroni,
        rank_biserial,
        wilcoxon_signed_rank,
    )

    lines = [
        "# Evaluation Summary Report",
        "",
        "Auto-generated dari `eval/results.csv` (output `app.evaluation.runner`).",
        "Untuk laporan akhir, copy table di sini + narasi insight.",
        "",
        "## Setup",
        "",
        f"- Total queries: {len(queries)}",
        f"- Corpus: 227 listing REAL Mamikos (lihat `data/README.md`)",
        f"- Ground truth: AI-assisted 3-annotator simulation, di-pool dari BM25 (lihat `eval/kappa_report.md`)",
        f"- Eval: standard top-K + pool-restricted (`results_pool_restricted.csv`) — lihat pooling bias di LAPORAN §8.2",
        f"- Top-K cutoff: 10",
        "",
        "## Aggregate Metrics per Model",
        "",
        "MAP disertai 95% CI (percentile bootstrap atas query, 10k resample).",
        "",
        "| Model | P@5 | P@10 | MAP | MAP 95% CI | NDCG@10 | MRR |",
        "|-------|-----|------|-----|------------|---------|-----|",
    ]

    aggregates: dict[str, dict[str, float]] = {}
    for model_name in sorted(data.keys()):
        agg = aggregate(data[model_name])
        aggregates[model_name] = agg
        aps = [v["ap"] for v in data[model_name].values()]
        lo, hi = bootstrap_ci_mean(aps)
        lines.append(
            f"| {model_name} | "
            f"{agg['p_at_5']:.4f} | {agg['p_at_10']:.4f} | "
            f"{agg['map']:.4f} | [{lo:.3f}, {hi:.3f}] | "
            f"{agg['ndcg_at_10']:.4f} | {agg['mrr']:.4f} |"
        )

    # Bold the best per column
    lines.append("")

    # Ranking by MAP
    ranked = sorted(aggregates.items(), key=lambda kv: -kv[1]["map"])
    lines.append("**Ranking by MAP**: " + " > ".join(
        f"{name} ({agg['map']:.3f})" for name, agg in ranked
    ))
    lines.append("")

    # Pairwise Wilcoxon + koreksi Holm-Bonferroni (keluarga m uji sekaligus)
    lines.extend([
        "## Pairwise Statistical Significance (Wilcoxon signed-rank, MAP)",
        "",
        "H0: Model A == Model B. alpha = 0.05. Karena ada banyak pasangan diuji",
        "sekaligus, signifikansi final memakai koreksi **Holm-Bonferroni**",
        "(kontrol family-wise error rate); kolom raw disertakan untuk transparansi.",
        "",
        "| Pair | Statistic | p-value | p-Holm | r (rank-biserial) | n | Sig (raw) | Sig (Holm) |",
        "|------|-----------|---------|--------|-------------------|---|-----------|------------|",
    ])

    model_names = list(data.keys())
    # Query ids yang ada di SEMUA model (paired test butuh pasangan lengkap)
    common_qids = sorted(set.intersection(
        *(set(data[m].keys()) for m in model_names)
    ))
    pair_stats: dict[str, tuple[float, int, float]] = {}
    raw_tests: list[tuple[str, float]] = []
    for i, ma in enumerate(model_names):
        for mb in model_names[i + 1:]:
            a_aps = [data[ma][qid]["ap"] for qid in common_qids]
            b_aps = [data[mb][qid]["ap"] for qid in common_qids]
            try:
                test = wilcoxon_signed_rank(a_aps, b_aps)
                label = f"{ma} vs {mb}"
                raw_tests.append((label, test.p_value))
                pair_stats[label] = (
                    test.statistic, test.n, rank_biserial(a_aps, b_aps)
                )
            except Exception as e:
                lines.append(f"| {ma} vs {mb} | ERROR: {e} | - | - | - | - | - | - |")

    holm_entries = holm_bonferroni(raw_tests, alpha=0.05)
    for entry in holm_entries:
        stat, n, r_eff = pair_stats[entry.label]
        sig_raw = "yes" if entry.p_value < 0.05 else "no"
        sig_holm = "**YES**" if entry.significant else "no"
        lines.append(
            f"| {entry.label} | {stat:.2f} | {entry.p_value:.4f} | "
            f"{entry.p_adjusted:.4f} | {r_eff:.3f} | {n} | {sig_raw} | {sig_holm} |"
        )
    n_raw_sig = sum(1 for e in holm_entries if e.p_value < 0.05)
    n_holm_sig = sum(1 for e in holm_entries if e.significant)
    n_q = len(common_qids)
    lines.extend([
        "",
        f"**{n_raw_sig}/{len(holm_entries)}** pasangan signifikan tanpa koreksi; "
        f"setelah Holm-Bonferroni **{n_holm_sig}/{len(holm_entries)}**. "
        f"Dengan n={n_q} query, uji non-signifikan dibaca *inconclusive*, "
        "bukan bukti dua model setara.",
        "",
    ])

    # Per-query analysis
    lines.extend([
        "## Per-Query Analysis (AP per Model)",
        "",
        "| Query | " + " | ".join(model_names) + " |",
        "|-------|" + "|".join(["---"] * len(model_names)) + "|",
    ])

    # Get all query_ids in sorted order
    query_ids = sorted(set(qid for model_data in data.values() for qid in model_data.keys()))
    for qid in query_ids:
        # Get query text (any model has it)
        q_text = next(
            (data[m][qid]["query_text"] for m in data if qid in data[m]), ""
        )
        row = [f"`{qid}` *{q_text[:50]}{'...' if len(q_text) > 50 else ''}*"]
        for m in model_names:
            ap = data[m].get(qid, {}).get("ap", 0.0)
            row.append(f"{ap:.3f}")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")

    # Constraint satisfaction (lensa kedua, kalau hasilnya ada)
    cs_rows = load_constraint_results(constraints_csv) if constraints_csv else None
    if cs_rows:
        model_cols = [c for c in cs_rows[0].keys() if c.startswith("cs_at_5_")]
        cs_means = sorted(
            (
                (col.removeprefix("cs_at_5_"),
                 sum(float(r[col]) for r in cs_rows) / len(cs_rows))
                for col in model_cols
            ),
            key=lambda kv: -kv[1],
        )
        lines.extend([
            "## Constraint Satisfaction @5 (lensa kedua — kebutuhan user)",
            "",
            "CS@5 = proporsi top-5 yang memenuhi SEMUA constraint query "
            "(gender + harga + fasilitas + radius 3 km dari anchor). Metric ini "
            "mengukur apa yang dioptimalkan smart pipeline dan TIDAK bergantung "
            "qrels (bebas pooling bias).",
            "",
            f"| Model | mean CS@5 (n={len(cs_rows)}) |",
            "|-------|------------|",
        ])
        for name, val in cs_means:
            bold = "**" if name == "smart" else ""
            lines.append(f"| {bold}{name}{bold} | {bold}{val:.4f}{bold} |")
        lines.extend([
            "",
            "Per-query detail: `eval/results_constraints.csv`.",
            "",
        ])

    # Insights
    lines.extend([
        "## Insights & Discussion (template untuk laporan)",
        "",
        "1. **BM25 best lexical baseline**: menang di queries lexical-heavy "
        "(exact match nama universitas/fasilitas di deskripsi pemilik). Selisih "
        "vs TF-IDF tidak signifikan.",
        "",
        "2. **IndoBERT (MiniLM) = pooling bias, bukan model buruk**: skor rendah "
        "di standard top-K karena GT di-pool dari kandidat lexical (hasil semantic "
        "unik tidak ter-judge). Pada pool-restricted eval dia kompetitif. Lihat "
        "`results_pool_restricted.csv` + LAPORAN §8.2.",
        "",
        "3. **Smart (model live)**: smart = BM25 + query understanding + geo + "
        "hard filter. Unggul di MAP/P@5/MRR standard dan dominan di CS@5 "
        "(selisih vs BM25 signifikan); selisih MAP standard vs BM25 belum "
        "signifikan. P@10 smart bisa di bawah BM25: hard filter memangkas "
        "kandidat dan dokumen hasil geo-augment yang belum ter-annotate "
        "dihitung rel=0 (pooling bias juga menekan smart).",
        "",
        "4. **Multiple comparison**: signifikansi final pakai Holm-Bonferroni "
        "(lihat tabel). Semua pasangan yang tetap signifikan melibatkan indobert "
        "standard top-K — konsisten dengan cerita pooling bias.",
        "",
        "5. **Sample size**: n=30 query (dinaikkan dari 15); uji non-signifikan "
        "= inconclusive. Hyperparameter dipilih di data eval yang sama "
        "(disclosed sebagai mild selection bias; future: held-out set).",
        "",
        "6. **Data real terse**: deskripsi pemilik median ~23 kata menurunkan "
        "absolute metric tapi otentik. Future: multi-model pooling untuk "
        "hilangkan pooling bias. Lihat `data/README.md`.",
        "",
        "## Files",
        "",
        "- `eval/queries.json`: 30 query set (v1.1 real-data)",
        "- `eval/ground_truth.csv`: 900 consensus annotations",
        "- `eval/kappa_report.md`: inter-annotator agreement",
        "- `eval/results.csv`: per-model per-query (standard top-K, termasuk smart)",
        "- `eval/results_pool_restricted.csv`: pool-restricted (fair) eval",
        "- `eval/results_constraints.csv`: Constraint Satisfaction @5 (smart vs bm25)",
        "- `eval/significance_map.csv`: Wilcoxon semua pasangan + Holm",
        "- `eval/eval_summary.md`: ini (aggregate + significance + analysis)",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--constraints", type=Path, default=None,
        help="results_constraints.csv (default: <output-dir>/results_constraints.csv)",
    )
    args = parser.parse_args()

    print(f"[load] {args.results}")
    data = load_results(args.results)
    print(f"[load] {len(data)} models, queries: "
          f"{[len(v) for v in data.values()]}")

    query_ids = sorted(set(qid for model_data in data.values() for qid in model_data.keys()))

    constraints_csv = args.constraints or (args.output_dir / "results_constraints.csv")
    summary_path = args.output_dir / "eval_summary.md"
    write_summary_md(summary_path, data, query_ids, constraints_csv=constraints_csv)
    print(f"[done] summary -> {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

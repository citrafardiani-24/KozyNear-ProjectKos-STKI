"""Gabungkan lembar anotasi manusia -> ground_truth_human.csv + Kappa asli.

Input: 1-3 file CSV hasil isi annotation_sheet (kolom relevance terisi 0/1/2).
Output:
  - eval/ground_truth_human.csv  (konsensus majority vote; 1 annotator =
    labelnya langsung dipakai)
  - eval/kappa_report_human.md   (Cohen's Kappa antar-manusia, kalau >= 2)

Lalu jalankan ulang evaluasi terhadap GT manusia:
  python -m app.evaluation.runner --queries ../eval/queries.json \
      --ground-truth ../eval/ground_truth_human.csv \
      --indexes-dir ../data/indexes --output ../eval/results_human.csv
  python -m scripts.eval_smart --ground-truth ../eval/ground_truth_human.csv

Usage:
  python -m scripts.ingest_human_annotations --sheets ../eval/annotation_A.csv ../eval/annotation_B.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.evaluation import cohen_kappa, weighted_kappa  # noqa: E402
from app.evaluation.kappa import interpret_kappa  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def load_sheet(path: Path) -> dict[tuple[str, str], int]:
    labels: dict[tuple[str, str], int] = {}
    skipped = 0
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            raw = (row.get("relevance") or "").strip()
            if raw not in {"0", "1", "2"}:
                skipped += 1
                continue
            labels[(row["query_id"], row["doc_id"])] = int(raw)
    if skipped:
        print(f"[warn] {path.name}: {skipped} baris tanpa label valid di-skip")
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest anotasi manusia")
    parser.add_argument("--sheets", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "eval" / "ground_truth_human.csv")
    args = parser.parse_args()

    annotators = [load_sheet(p) for p in args.sheets]
    for p, a in zip(args.sheets, annotators):
        print(f"[load] {p.name}: {len(a)} label")
    if not any(annotators):
        print("[error] tidak ada label valid"); return 1

    all_keys = sorted(set().union(*[set(a) for a in annotators]))
    consensus: dict[tuple[str, str], int] = {}
    for key in all_keys:
        votes = [a[key] for a in annotators if key in a]
        top, top_count = Counter(votes).most_common(1)[0]
        # 3 annotator beda semua -> ambil median-ish (nilai tengah sorted)
        consensus[key] = sorted(votes)[len(votes) // 2] if top_count == 1 else top

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "doc_id", "relevance"])
        for (qid, did), rel in consensus.items():
            w.writerow([qid, did, rel])
    dist = Counter(consensus.values())
    print(f"[consensus] {len(consensus)} label -> {args.output}")
    print(f"[distribusi] rel0={dist.get(0,0)} rel1={dist.get(1,0)} rel2={dist.get(2,0)}")

    if len(annotators) >= 2:
        shared = sorted(set.intersection(*[set(a) for a in annotators]))
        lines = [
            "# Inter-Annotator Agreement (HUMAN)", "",
            f"Annotator: {len(annotators)} manusia, shared items: {len(shared)}", "",
            "| Pair | Cohen's Kappa | Interpretasi | Weighted Kappa |",
            "|------|---------------|--------------|----------------|",
        ]
        for i in range(len(annotators)):
            for j in range(i + 1, len(annotators)):
                a = [annotators[i][k] for k in shared]
                b = [annotators[j][k] for k in shared]
                k = cohen_kappa(a, b, labels=[0, 1, 2])
                kw = weighted_kappa(a, b, labels=[0, 1, 2], weight_type="linear")
                lines.append(
                    f"| {args.sheets[i].stem} vs {args.sheets[j].stem} "
                    f"| {k:.3f} | {interpret_kappa(k)} | {kw:.3f} |")
                print(f"[kappa] {args.sheets[i].stem} vs {args.sheets[j].stem}: "
                      f"{k:.3f} ({interpret_kappa(k)})")
        report = ROOT / "eval" / "kappa_report_human.md"
        report.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[saved] {report}")
    else:
        print("[info] 1 annotator: Kappa butuh >= 2; label dipakai langsung "
              "(disclose sebagai single-annotator di laporan)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

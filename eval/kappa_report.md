# Inter-Annotator Agreement Report

**Total shared annotations** (across 3 annotators): 900

Pairwise Cohen's Kappa (nominal) + Weighted Kappa (ordinal, linear):

| Pair | Cohen's Kappa | Interpretation | Weighted Kappa |
|------|---------------|----------------|----------------|
| Annotator A vs B | 0.731 | substantial | 0.786 |
| Annotator A vs C | 0.866 | almost perfect | 0.887 |
| Annotator B vs C | 0.604 | substantial | 0.687 |

**Status**: ada pair dengan Kappa < 0.7. Untuk full production anotasi, lakuin disagreement discussion + re-annotate problematic queries (lihat docs/ground_truth_protocol.md).

## Methodology Disclosure

Ini AI-assisted rule-based annotation (bootstrap GT), BUKAN human annotation. 3 'annotators' adalah heuristic variants:
- A (strict): rule-based scoring
- B (lenient): A + 0.5 score boost
- C (noisy): A + uniform(-0.5, 0.5) jitter

Trade-off vs human annotation: faster (instant vs 4 jam), reproducible (deterministic), tapi authenticity rubric Evaluation 10% lebih rendah. 

Untuk laporan akhir, **wajib document** ini sebagai limitation dan kalau ada budget waktu replace dengan 3-human annotator pass mengikuti [docs/ground_truth_protocol.md](../docs/ground_truth_protocol.md).
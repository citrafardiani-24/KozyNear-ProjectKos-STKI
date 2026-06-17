import { useEffect, useState } from 'react'
import {
  getEvalSummary,
  type EvalSummary,
  type ModelMetrics,
} from '../api/client'

const MODEL_LABELS: Record<string, string> = {
  smart: 'Smart (live)',
  bm25: 'BM25',
  tfidf: 'TF-IDF',
  indobert: 'Neural MiniLM',
  hybrid: 'Hybrid',
}

function fmt(x: number): string {
  return x.toFixed(4)
}

function MetricsTable({ rows, caption }: { rows: ModelMetrics[]; caption: string }) {
  if (rows.length === 0) return null
  const bestMap = Math.max(...rows.map((r) => r.map))
  return (
    <div className="eval-block">
      <h3>{caption}</h3>
      <table className="eval-table">
        <thead>
          <tr>
            <th>Model</th>
            <th>P@5</th>
            <th>P@10</th>
            <th>MAP</th>
            <th>NDCG@10</th>
            <th>MRR</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.model} className={r.map === bestMap ? 'best-row' : ''}>
              <td>{MODEL_LABELS[r.model] ?? r.model}</td>
              <td>{fmt(r.p_at_5)}</td>
              <td>{fmt(r.p_at_10)}</td>
              <td>{fmt(r.map)}</td>
              <td>{fmt(r.ndcg_at_10)}</td>
              <td>{fmt(r.mrr)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

const CACHE_KEY = 'kozynear-eval-summary'
const CACHE_TTL_MS = 5 * 60 * 1000

export default function EvalTab() {
  const [data, setData] = useState<EvalSummary | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    // Hasil eval statis per deploy: cache 5 menit di sessionStorage supaya
    // pindah-pindah tab tidak refetch terus.
    try {
      const cached = sessionStorage.getItem(CACHE_KEY)
      if (cached) {
        const { ts, payload } = JSON.parse(cached) as {
          ts: number
          payload: EvalSummary
        }
        if (Date.now() - ts < CACHE_TTL_MS) {
          setData(payload)
          return
        }
      }
    } catch {
      // cache korup -> abaikan, fetch ulang
    }
    getEvalSummary()
      .then((payload) => {
        setData(payload)
        try {
          sessionStorage.setItem(
            CACHE_KEY,
            JSON.stringify({ ts: Date.now(), payload }),
          )
        } catch {
          // storage penuh/blocked -> tidak fatal
        }
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))
  }, [])

  if (error) return <p className="error">Gagal memuat evaluasi: {error}</p>
  if (!data) return <p className="empty">Memuat hasil evaluasi...</p>

  return (
    <div>
      <div className="tab-intro">
        Perbandingan lima model lewat <strong>tiga lensa</strong> supaya jujur:{' '}
        <strong>Standard</strong> (ranking atas seluruh corpus),{' '}
        <strong>Pool-Restricted</strong> (adil antar model, menghapus bias
        karena ground-truth di-pool dari satu model), dan{' '}
        <strong>Constraint Satisfaction@5</strong> (apakah hasil benar-benar
        memenuhi kebutuhan user; bebas ground-truth). Signifikansi diuji
        Wilcoxon + koreksi Holm-Bonferroni dengan effect size.
      </div>
      <p className="meta">
        {data.total_queries} query &middot; corpus 227 listing real Mamikos &middot;
        Wilcoxon signed-rank + koreksi Holm-Bonferroni
      </p>

      <MetricsTable rows={data.standard} caption="Standard Top-K (seluruh corpus)" />
      <MetricsTable
        rows={data.pool_restricted}
        caption="Pool-Restricted (fair, ranking di dalam pool annotated)"
      />

      {data.constraints && (
        <div className="eval-block">
          <h3>Constraint Satisfaction @5 (kebutuhan user, bebas qrels)</h3>
          <table className="eval-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>mean CS@5 (n={data.constraints.n_queries})</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(data.constraints.mean_cs_at_5)
                .sort((a, b) => b[1] - a[1])
                .map(([model, value]) => (
                  <tr key={model}>
                    <td>{MODEL_LABELS[model] ?? model}</td>
                    <td>{fmt(value)}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      )}

      {data.significance.length > 0 && (
        <div className="eval-block">
          <h3>Signifikansi Pairwise (MAP)</h3>
          <table className="eval-table">
            <thead>
              <tr>
                <th>Pasangan</th>
                <th>p-value</th>
                <th>p-Holm</th>
                <th>Effect size (r)</th>
                <th>Signifikan (Holm)</th>
              </tr>
            </thead>
            <tbody>
              {data.significance.map((s) => (
                <tr key={s.pair}>
                  <td>{s.pair}</td>
                  <td>{s.p_value.toFixed(4)}</td>
                  <td>{s.p_holm.toFixed(4)}</td>
                  <td>{s.r_rank_biserial != null ? s.r_rank_biserial.toFixed(3) : '-'}</td>
                  <td>{s.significant_holm ? 'YA' : 'tidak'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <p className="eval-note">{data.note}</p>
    </div>
  )
}

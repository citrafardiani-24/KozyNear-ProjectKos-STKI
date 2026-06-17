import { useState, type FormEvent } from 'react'
import DebugPanel, { type DebugInfo } from './components/DebugPanel'
import EvalTab from './components/EvalTab'
import FilterPanel from './components/FilterPanel'
import NotebookTab from './components/NotebookTab'
import PreprocessTab from './components/PreprocessTab'
import ResultCard from './components/ResultCard'
import StatsTab from './components/StatsTab'
import {
  ApiError,
  searchKos,
  type Model,
  type SearchFilters,
  type SearchResult,
} from './api/client'

type Tab = 'search' | 'eval' | 'prepro' | 'stats' | 'notebook'

const TABS: { id: Tab; label: string }[] = [
  { id: 'search', label: 'Pencarian' },
  { id: 'eval', label: 'Evaluasi Model' },
  { id: 'prepro', label: 'Preprocessing' },
  { id: 'stats', label: 'Statistik' },
  { id: 'notebook', label: 'Notebook' },
]

// Contoh query untuk demo: tiap satu menonjolkan satu kemampuan sistem.
const EXAMPLE_QUERIES: { q: string; note: string }[] = [
  { q: 'kos putri dekat unila wifi murah', note: 'multi-constraint: gender + kampus + fasilitas + budget' },
  { q: 'boarding house for girls near itera', note: 'tahan code-switch Inggris + typo nama kampus' },
  { q: 'kos campur maksimal 1,5 juta', note: 'paham harga "juta"' },
  { q: 'kos dekat transmart way halim', note: 'geo ke landmark, bukan cuma kampus' },
  { q: 'kos putra kamar mandi dalam rajabasa', note: 'fasilitas + kecamatan' },
]

export default function App() {
  const [tab, setTab] = useState<Tab>('search')
  const [query, setQuery] = useState('')
  const [model, setModel] = useState<Model>('smart')
  const [filters, setFilters] = useState<SearchFilters>({})
  const [results, setResults] = useState<SearchResult[]>([])
  const [loading, setLoading] = useState(false)
  const [debug, setDebug] = useState<DebugInfo | null>(null)
  const [tookMs, setTookMs] = useState<number | null>(null)
  const [hasSearched, setHasSearched] = useState(false)
  const [understood, setUnderstood] = useState<Record<string, unknown> | null>(null)
  const [relaxed, setRelaxed] = useState<string[]>([])

  async function runSearch(currentFilters: SearchFilters, q: string = query) {
    setLoading(true)
    setDebug(null)
    try {
      const data = await searchKos(q, model, 10, currentFilters)
      setResults(data.results ?? [])
      setTookMs(data.took_ms ?? null)
      setUnderstood(data.understood ?? null)
      setRelaxed(data.relaxed ?? [])
    } catch (err) {
      setResults([])
      setUnderstood(null)
      setRelaxed([])
      setDebug(buildDebugInfo(err, model, query, currentFilters))
    } finally {
      setLoading(false)
    }
  }

  async function handleSearch(e: FormEvent) {
    e.preventDefault()
    if (!query.trim()) return
    setHasSearched(true)
    await runSearch(filters)
  }

  function handleFiltersChange(next: SearchFilters) {
    setFilters(next)
    if (hasSearched && query.trim()) {
      void runSearch(next)
    }
  }

  async function runExample(ex: string) {
    setQuery(ex)
    setModel('smart')
    setHasSearched(true)
    await runSearch(filters, ex)
  }

  return (
    <div className="container">
      <header className="app-header">
        <h1>KozyNear</h1>
        <p className="subtitle">
          Cari kos seluruh Bandar Lampung &mdash; UNILA, ITERA, Darmajaya, UBL, UIN, Teknokrat, dll.
        </p>
      </header>

      <nav className="tabs" aria-label="Navigasi tab">
        {TABS.map((t) => (
          <button
            key={t.id}
            className={`tab-btn ${tab === t.id ? 'active' : ''}`}
            onClick={() => setTab(t.id)}
            type="button"
          >
            {t.label}
          </button>
        ))}
      </nav>

      {tab === 'eval' && <EvalTab />}
      {tab === 'prepro' && <PreprocessTab />}
      {tab === 'stats' && <StatsTab />}
      {tab === 'notebook' && <NotebookTab />}

      {tab === 'search' && (
        <>
      <div className="tab-intro">
        Ketik kebutuhan kos pakai <strong>bahasa natural</strong>. Model{' '}
        <strong>Smart</strong> (default) memecah query jadi constraint
        terstruktur &mdash; gender, budget, fasilitas, dan kampus tujuan
        &mdash; lalu memeringkat dengan gabungan kecocokan teks, jarak
        geografis, dan atribut. Coba salah satu contoh:
      </div>

      <div className="examples">
        {EXAMPLE_QUERIES.map((ex) => (
          <button
            key={ex.q}
            type="button"
            className="example-chip"
            title={ex.note}
            onClick={() => void runExample(ex.q)}
          >
            {ex.q}
          </button>
        ))}
      </div>

      <form onSubmit={handleSearch} className="search-form">
        <input
          type="text"
          placeholder="Contoh: kos putra dekat unila ada wifi dan ac"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          aria-label="Query pencarian"
        />
        <select
          value={model}
          onChange={(e) => setModel(e.target.value as Model)}
          aria-label="Pilih model IR"
          title="Model IR yang dipakai untuk ranking"
        >
          <option value="smart">Smart (rekomendasi)</option>
          <option value="bm25">BM25</option>
          <option value="tfidf">TF-IDF</option>
          <option value="indobert">Neural MiniLM (butuh server 16GB)</option>
          <option value="hybrid">Hybrid BM25+MiniLM (butuh server 16GB)</option>
        </select>
        <button type="submit" disabled={loading}>
          {loading ? 'Mencari...' : 'Cari'}
        </button>
      </form>

      <div className="layout">
        <FilterPanel filters={filters} onChange={handleFiltersChange} />

        <main className="results-main">
          {debug && <DebugPanel info={debug} />}

          {tookMs !== null && hasSearched && !debug && (
            <div className="meta">
              {results.length} hasil &middot; {tookMs}ms &middot; model{' '}
              <code>{model}</code>
            </div>
          )}

          {model === 'smart' && understood && hasSearched && !debug && (
            <div className="understood">
              <span className="understood-label">Yang kami pahami:</span>
              {renderUnderstoodChips(understood)}
              {relaxed.length > 0 && (
                <span className="relaxed-note">
                  filter dilonggarkan: {relaxed.join(', ')}
                </span>
              )}
            </div>
          )}

          <div className="results">
            {results.length === 0 && !loading && !debug && (
              <p className="empty">
                {hasSearched
                  ? 'Tidak ada hasil. Coba ubah query atau filter.'
                  : 'Ketik query lalu klik Cari untuk mulai.'}
              </p>
            )}
            {results.map((r) => (
              <ResultCard key={r.id} result={r} />
            ))}
          </div>
        </main>
      </div>
        </>
      )}

      <footer className="footer">
        <small>
          STKI Final Project &middot; Universitas Lampung &middot; 2026
          {' · '}
          <a
            href="/api/status"
            target="_blank"
            rel="noreferrer"
            className="footer-link"
          >
            /api/status
          </a>
        </small>
      </footer>
    </div>
  )
}

function buildDebugInfo(
  err: unknown,
  model: Model,
  query: string,
  filters: SearchFilters,
): DebugInfo {
  const timestamp = new Date().toISOString()
  if (err instanceof ApiError) {
    return {
      timestamp,
      status: err.status,
      url: err.url,
      method: err.method,
      request: err.request,
      responseBody: err.responseBody,
      responseHeaders: err.responseHeaders,
      errorMessage: err.message,
      errorStack: err.stack,
    }
  }
  if (err instanceof Error) {
    return {
      timestamp,
      errorMessage: err.message,
      errorStack: err.stack,
      request: { model, query, ...filters },
    }
  }
  return {
    timestamp,
    errorMessage: String(err),
    request: { model, query, ...filters },
  }
}

function renderUnderstoodChips(u: Record<string, unknown>) {
  const chips: string[] = []
  if (u.gender) chips.push(String(u.gender))
  if (Array.isArray(u.fasilitas)) {
    for (const f of u.fasilitas) chips.push(String(f))
  }
  if (u.anchor) chips.push(`dekat ${String(u.anchor)}`)
  if (u.kecamatan) chips.push(`kec. ${String(u.kecamatan)}`)
  if (u.harga_min) {
    chips.push(`min Rp ${Number(u.harga_min).toLocaleString('id-ID')}`)
  }
  if (u.harga_max) {
    chips.push(`maks Rp ${Number(u.harga_max).toLocaleString('id-ID')}`)
  }
  return chips.map((c) => (
    <span key={c} className="chip">
      {c}
    </span>
  ))
}

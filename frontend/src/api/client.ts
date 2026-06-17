// API client untuk KozyNear backend.
//
// VITE_API_URL behavior:
// - Local dev: VITE_API_URL=http://localhost:8000 (di .env)
// - Production Docker: VITE_API_URL="" -> pakai relative path (same origin)
//   karena frontend + API di-serve dari kozynear.onrender.com bersama.

const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

export type Model = 'tfidf' | 'bm25' | 'indobert' | 'hybrid' | 'smart'
export type Tipe = 'putra' | 'putri' | 'campur'

export interface SearchFilters {
  harga_min?: number
  harga_max?: number
  tipe?: Tipe
  kecamatan?: string
}

export interface SearchResult {
  id: string
  judul: string
  deskripsi: string
  harga_per_bulan: number
  tipe: string
  fasilitas?: string[]
  alamat: string
  kecamatan: string
  score: number
  koordinat?: [number, number]
}

export interface SearchResponse {
  query: string
  model: Model
  top_k: number
  took_ms: number
  results: SearchResult[]
  understood?: Record<string, unknown>
  relaxed?: string[]
}

export interface ListingDetail extends SearchResult {
  koordinat?: [number, number]
  jarak_kampus_km?: number
  url_source?: string
  scrape_date?: string
}

export interface ModelMetrics {
  model: string
  p_at_5: number
  p_at_10: number
  map: number
  ndcg_at_10: number
  mrr: number
  n_queries: number
}

export interface SignificanceRow {
  pair: string
  p_value: number
  p_holm: number
  r_rank_biserial?: number | null
  significant_raw: boolean
  significant_holm: boolean
}

export interface ConstraintSummary {
  n_queries: number
  mean_cs_at_5: Record<string, number>
}

export interface EvalSummary {
  standard: ModelMetrics[]
  pool_restricted: ModelMetrics[]
  constraints: ConstraintSummary | null
  significance: SignificanceRow[]
  total_queries: number
  note: string
}

export interface PreprocessStage {
  stage: string
  output: string | string[] | number[]
}

export interface PreprocessTrace {
  raw: string
  processed: string
  tokens: string[]
  extracted_prices: number[]
  stages: PreprocessStage[]
}

export interface CorpusStats {
  total_listings: number
  kecamatan: Record<string, number>
  tipe: Record<string, number>
  harga_min: number | null
  harga_max: number | null
  harga_avg: number | null
  vocab_size: number | null
  models_loaded: string[]
  source: string
}

/** Rich error class — carries HTTP context untuk debug UI. */
export class ApiError extends Error {
  status: number
  url: string
  method: string
  request: Record<string, unknown>
  responseBody?: unknown
  responseHeaders?: Record<string, string>

  constructor(args: {
    message: string
    status: number
    url: string
    method?: string
    request?: Record<string, unknown>
    responseBody?: unknown
    responseHeaders?: Record<string, string>
  }) {
    super(args.message)
    this.name = 'ApiError'
    this.status = args.status
    this.url = args.url
    this.method = args.method ?? 'GET'
    this.request = args.request ?? {}
    this.responseBody = args.responseBody
    this.responseHeaders = args.responseHeaders
  }
}

function apiUrl(path: string, params?: URLSearchParams): string {
  const fullPath = params ? `${path}?${params}` : path
  return `${API_BASE}${fullPath}`
}

async function parseErrorBody(res: Response): Promise<unknown> {
  const text = await res.text().catch(() => '')
  if (!text) return null
  try {
    return JSON.parse(text)
  } catch {
    return text
  }
}

function extractRelevantHeaders(res: Response): Record<string, string> {
  const headers: Record<string, string> = {}
  const keys = ['content-type', 'x-render-origin-server', 'x-frame-options', 'server', 'date']
  for (const k of keys) {
    const v = res.headers.get(k)
    if (v) headers[k] = v
  }
  return headers
}

function friendlyMessage(status: number, body: unknown): string {
  if (status === 502 || status === 504) {
    return 'Backend sedang loading (cold start ~30-60 detik). Coba lagi dalam 1 menit.'
  }
  if (status === 503) {
    if (body && typeof body === 'object' && 'detail' in body) {
      const detail = (body as { detail: unknown }).detail
      if (typeof detail === 'object' && detail && 'error' in detail) {
        return String((detail as { error: unknown }).error)
      }
      if (typeof detail === 'string') return detail
    }
    return 'Service unavailable: index belum di-load atau model gak siap.'
  }
  if (status === 404) return 'Endpoint atau resource tidak ditemukan.'
  if (status >= 500) return `Server error HTTP ${status}.`
  return `Request gagal HTTP ${status}.`
}

export async function searchKos(
  query: string,
  model: Model = 'smart',
  topK = 10,
  filters: SearchFilters = {},
): Promise<SearchResponse> {
  const params = new URLSearchParams({
    q: query,
    model,
    top_k: String(topK),
  })
  if (filters.harga_min != null) params.set('harga_min', String(filters.harga_min))
  if (filters.harga_max != null) params.set('harga_max', String(filters.harga_max))
  if (filters.tipe) params.set('tipe', filters.tipe)
  if (filters.kecamatan) params.set('kecamatan', filters.kecamatan)

  const url = apiUrl('/api/search', params)
  const requestParams = { q: query, model, top_k: topK, ...filters }

  const res = await fetch(url, { headers: { Accept: 'application/json' } })
  if (!res.ok) {
    const body = await parseErrorBody(res)
    throw new ApiError({
      message: friendlyMessage(res.status, body),
      status: res.status,
      url,
      method: 'GET',
      request: requestParams,
      responseBody: body,
      responseHeaders: extractRelevantHeaders(res),
    })
  }
  return res.json()
}

export async function getListing(id: string): Promise<ListingDetail> {
  const url = apiUrl(`/api/listings/${encodeURIComponent(id)}`)
  const res = await fetch(url)
  if (!res.ok) {
    const body = await parseErrorBody(res)
    throw new ApiError({
      message: friendlyMessage(res.status, body),
      status: res.status,
      url,
      request: { id },
      responseBody: body,
      responseHeaders: extractRelevantHeaders(res),
    })
  }
  return res.json()
}

export async function getHealth(): Promise<{ status: string; service: string }> {
  const res = await fetch(apiUrl('/health'))
  return res.json()
}

export async function getStatus(): Promise<unknown> {
  const res = await fetch(apiUrl('/api/status'))
  return res.json()
}

export async function getEvalSummary(): Promise<EvalSummary> {
  const url = apiUrl('/api/eval/summary')
  const res = await fetch(url)
  if (!res.ok) {
    const body = await parseErrorBody(res)
    throw new ApiError({
      message: friendlyMessage(res.status, body),
      status: res.status,
      url,
      responseBody: body,
      responseHeaders: extractRelevantHeaders(res),
    })
  }
  return res.json()
}

export async function getPreprocessTrace(text: string): Promise<PreprocessTrace> {
  const params = new URLSearchParams({ text })
  const url = apiUrl('/api/preprocess', params)
  const res = await fetch(url)
  if (!res.ok) {
    const body = await parseErrorBody(res)
    throw new ApiError({
      message: friendlyMessage(res.status, body),
      status: res.status,
      url,
      request: { text },
      responseBody: body,
      responseHeaders: extractRelevantHeaders(res),
    })
  }
  return res.json()
}

export async function getCorpusStats(): Promise<CorpusStats> {
  const url = apiUrl('/api/stats')
  const res = await fetch(url)
  if (!res.ok) {
    const body = await parseErrorBody(res)
    throw new ApiError({
      message: friendlyMessage(res.status, body),
      status: res.status,
      url,
      responseBody: body,
      responseHeaders: extractRelevantHeaders(res),
    })
  }
  return res.json()
}

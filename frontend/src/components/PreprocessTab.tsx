import { useState, type FormEvent } from 'react'
import { getPreprocessTrace, type PreprocessTrace } from '../api/client'

const STAGE_LABELS: Record<string, string> = {
  strip_html: '1. Strip HTML',
  normalize_whitespace: '2. Normalisasi whitespace',
  extract_prices: '3. Ekstraksi harga',
  lowercase: '4. Case folding',
  apply_jargon_dict: '5. Kamus jargon (106 entri)',
  correct_spelling: '6. Koreksi ejaan',
  tokenize: '7. Tokenisasi',
  remove_stopwords: '8. Stopword removal (Sastrawi + domain)',
  stem: '9. Stemming (Sastrawi)',
}

function StageOutput({ output }: { output: string | string[] | number[] }) {
  if (Array.isArray(output)) {
    if (output.length === 0) return <em className="stage-empty">(kosong)</em>
    return (
      <span className="token-list">
        {output.map((tok, i) => (
          <span key={`${tok}-${i}`} className="chip">
            {String(tok)}
          </span>
        ))}
      </span>
    )
  }
  return <span>{output || <em className="stage-empty">(kosong)</em>}</span>
}

export default function PreprocessTab() {
  const [text, setText] = useState('Kos putri KM dlm, AC, wifi kenceng dkt UNILA 800rb/bln')
  const [trace, setTrace] = useState<PreprocessTrace | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function run(e: FormEvent) {
    e.preventDefault()
    if (!text.trim()) return
    setLoading(true)
    setError(null)
    try {
      setTrace(await getPreprocessTrace(text))
    } catch (err: unknown) {
      setTrace(null)
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <div className="tab-intro">
        Teks mentah listing kos diolah lewat <strong>pipeline 9-stage</strong>{' '}
        sebelum diindeks: bersihkan HTML, ekstrak harga, normalisasi,{' '}
        <strong>kamus jargon domain</strong> (mis. "KM Dlm" &rarr; "kamar mandi
        dalam"), koreksi ejaan, tokenisasi, buang stopword, lalu{' '}
        <strong>stemming Sastrawi</strong>. Lihat tiap langkah secara langsung di
        bawah (coba ketik singkatan + harga).
      </div>
      <form onSubmit={run} className="search-form">
        <input
          type="text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          aria-label="Teks untuk preprocessing"
        />
        <button type="submit" disabled={loading}>
          {loading ? 'Memproses...' : 'Proses'}
        </button>
      </form>

      {error && <p className="error">{error}</p>}

      {trace && (
        <div className="stage-list">
          <div className="stage-card stage-raw">
            <div className="stage-name">Input mentah</div>
            <div className="stage-output">{trace.raw}</div>
          </div>
          {trace.stages.map((s) => (
            <div key={s.stage} className="stage-card">
              <div className="stage-name">{STAGE_LABELS[s.stage] ?? s.stage}</div>
              <div className="stage-output">
                <StageOutput output={s.output} />
              </div>
            </div>
          ))}
          <div className="stage-card stage-final">
            <div className="stage-name">Hasil akhir (yang masuk index)</div>
            <div className="stage-output">
              <StageOutput output={trace.tokens} />
            </div>
            {trace.extracted_prices.length > 0 && (
              <div className="stage-prices">
                Harga terdeteksi:{' '}
                {trace.extracted_prices
                  .map((p) => `Rp ${p.toLocaleString('id-ID')}`)
                  .join(', ')}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

import { useEffect, useState } from 'react'
import { getCorpusStats, type CorpusStats } from '../api/client'

function rupiah(x: number | null): string {
  return x === null ? '-' : `Rp ${x.toLocaleString('id-ID')}`
}

export default function StatsTab() {
  const [stats, setStats] = useState<CorpusStats | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    getCorpusStats()
      .then(setStats)
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))
  }, [])

  if (error) return <p className="error">Gagal memuat statistik: {error}</p>
  if (!stats) return <p className="empty">Memuat statistik corpus...</p>

  const kecEntries = Object.entries(stats.kecamatan).sort((a, b) => b[1] - a[1])
  const maxKec = kecEntries.length > 0 ? kecEntries[0][1] : 1

  return (
    <div>
      <div className="tab-intro">
        Statistik corpus secara langsung dari database. Semua{' '}
        <strong>227 listing nyata</strong> hasil scrape Mamikos (deskripsi
        pemilik + koordinat asli), setara <strong>88,7% populasi</strong> kos
        bulanan Bandar Lampung versi hitungan Mamikos (256) &mdash; mendekati
        sensus, bukan sampel kecil.
      </div>
      <div className="stat-cards">
        <div className="stat-card">
          <div className="stat-value">{stats.total_listings}</div>
          <div className="stat-label">listing real Mamikos</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{kecEntries.length}</div>
          <div className="stat-label">kecamatan ter-cover</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{stats.vocab_size ?? '-'}</div>
          <div className="stat-label">term unik di index BM25</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{rupiah(stats.harga_avg)}</div>
          <div className="stat-label">harga rata-rata / bulan</div>
        </div>
      </div>

      <div className="eval-block">
        <h3>Distribusi Tipe</h3>
        <div className="token-list">
          {Object.entries(stats.tipe).map(([t, c]) => (
            <span key={t} className={`badge badge-${t}`}>
              {t}: {c}
            </span>
          ))}
        </div>
      </div>

      <div className="eval-block">
        <h3>Distribusi Kecamatan</h3>
        <div className="bar-list">
          {kecEntries.map(([kec, count]) => (
            <div key={kec} className="bar-row">
              <span className="bar-label">{kec}</span>
              <div className="bar-track">
                <div
                  className="bar-fill"
                  style={{ width: `${(count / maxKec) * 100}%` }}
                />
              </div>
              <span className="bar-count">{count}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="eval-block">
        <h3>Rentang Harga</h3>
        <p className="meta">
          {rupiah(stats.harga_min)} sampai {rupiah(stats.harga_max)} per bulan
        </p>
      </div>

      <p className="eval-note">
        Sumber data: {stats.source} &middot; model siap:{' '}
        {stats.models_loaded.join(', ') || '-'}
      </p>
    </div>
  )
}

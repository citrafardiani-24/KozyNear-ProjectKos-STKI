import type { SearchResult } from '../api/client'

interface ResultCardProps {
  result: SearchResult
}

export default function ResultCard({ result }: ResultCardProps) {
  const facilities = result.fasilitas ?? []
  const description =
    result.deskripsi.length > 220
      ? result.deskripsi.slice(0, 220) + '...'
      : result.deskripsi

  return (
    <article className="result-card">
      <header className="result-card-header">
        <h3 className="result-title">{result.judul}</h3>
        {result.tipe && (
          <span className={`badge badge-${result.tipe}`}>{result.tipe}</span>
        )}
      </header>

      {result.harga_per_bulan != null && (
        <p className="price">
          Rp {result.harga_per_bulan.toLocaleString('id-ID')}{' '}
          <span className="price-unit">/ bulan</span>
        </p>
      )}

      {(result.alamat || result.kecamatan) && (
        <p className="address">
          {result.alamat}
          {result.alamat && result.kecamatan ? ' · ' : ''}
          {result.kecamatan}
        </p>
      )}

      {facilities.length > 0 && (
        <ul className="facilities">
          {facilities.slice(0, 6).map((f) => (
            <li key={f}>{f}</li>
          ))}
          {facilities.length > 6 && (
            <li className="facility-more">+{facilities.length - 6} lainnya</li>
          )}
        </ul>
      )}

      <p className="description">{description}</p>

      <footer className="result-card-footer">
        <span className="score" title="Relevance score dari IR model">
          score: {result.score.toFixed(4)}
        </span>
        {result.koordinat && (
          <a
            className="maps-link"
            href={`https://www.google.com/maps?q=${result.koordinat[0]},${result.koordinat[1]}`}
            target="_blank"
            rel="noreferrer"
          >
            Lihat di Maps
          </a>
        )}
      </footer>
    </article>
  )
}

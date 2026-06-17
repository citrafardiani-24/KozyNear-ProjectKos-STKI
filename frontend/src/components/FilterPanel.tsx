import type { SearchFilters, Tipe } from '../api/client'

// Kecamatan Bandar Lampung — 17 kecamatan yang punya listing real di corpus
// (urut by jumlah listing). Sumber: data/raw/mamikos_real_v2.jsonl.
const KECAMATAN_OPTIONS = [
  'Sukarame',
  'Kedaton',
  'Rajabasa',
  'Tanjung Karang Timur',
  'Sukabumi',
  'Tanjung Senang',
  'Kedamaian',
  'Kemiling',
  'Tanjung Karang Pusat',
  'Enggal',
  'Langkapura',
  'Labuhan Ratu',
  'Way Halim',
  'Teluk Betung Selatan',
  'Teluk Betung Utara',
  'Tanjung Karang Barat',
  'Panjang',
]

interface FilterPanelProps {
  filters: SearchFilters
  onChange: (next: SearchFilters) => void
}

export default function FilterPanel({ filters, onChange }: FilterPanelProps) {
  const hasActiveFilter =
    filters.tipe ||
    filters.harga_min != null ||
    filters.harga_max != null ||
    filters.kecamatan

  return (
    <aside className="filter-panel">
      <div className="filter-header">
        <h3>Filter</h3>
        {hasActiveFilter && (
          <button
            type="button"
            className="filter-reset"
            onClick={() => onChange({})}
          >
            Reset
          </button>
        )}
      </div>

      {/* Tipe kos */}
      <div className="filter-group">
        <label htmlFor="filter-tipe">Tipe Kos</label>
        <select
          id="filter-tipe"
          value={filters.tipe ?? ''}
          onChange={(e) =>
            onChange({
              ...filters,
              tipe: (e.target.value as Tipe) || undefined,
            })
          }
        >
          <option value="">Semua</option>
          <option value="putra">Putra</option>
          <option value="putri">Putri</option>
          <option value="campur">Campur</option>
        </select>
      </div>

      {/* Range harga */}
      <div className="filter-group">
        <label>Harga / bulan (Rp)</label>
        <div className="filter-range">
          <input
            type="number"
            min="0"
            step="50000"
            placeholder="Min"
            value={filters.harga_min ?? ''}
            onChange={(e) =>
              onChange({
                ...filters,
                harga_min: e.target.value
                  ? parseInt(e.target.value)
                  : undefined,
              })
            }
          />
          <span aria-hidden="true">&mdash;</span>
          <input
            type="number"
            min="0"
            step="50000"
            placeholder="Max"
            value={filters.harga_max ?? ''}
            onChange={(e) =>
              onChange({
                ...filters,
                harga_max: e.target.value
                  ? parseInt(e.target.value)
                  : undefined,
              })
            }
          />
        </div>
      </div>

      {/* Kecamatan */}
      <div className="filter-group">
        <label htmlFor="filter-kecamatan">Kecamatan</label>
        <select
          id="filter-kecamatan"
          value={filters.kecamatan ?? ''}
          onChange={(e) =>
            onChange({ ...filters, kecamatan: e.target.value || undefined })
          }
        >
          <option value="">Semua</option>
          {KECAMATAN_OPTIONS.map((kec) => (
            <option key={kec} value={kec}>
              {kec}
            </option>
          ))}
        </select>
      </div>
    </aside>
  )
}

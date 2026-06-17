/**
 * Tab Notebook: menampilkan notebook model yang sudah dieksekusi (render
 * Jupyter asli: kode + output + teks + chart) lewat iframe ke /notebook.html.
 * File dihasilkan offline oleh backend/scripts/build_showcase_notebook.py dan
 * disalin Vite dari frontend/public/ ke root build (same-origin dengan API).
 */
export default function NotebookTab() {
  return (
    <div>
      <p className="meta">
        Notebook reproducible: lima model IR di corpus 227 listing real, tiap
        sel dieksekusi sungguhan (kode, output, chart). Sumber:{' '}
        <a href="/notebook.html" target="_blank" rel="noreferrer" className="footer-link">
          buka di tab penuh
        </a>
        .
      </p>
      <iframe
        src="/notebook.html"
        title="Notebook showcase model"
        className="notebook-frame"
      />
    </div>
  )
}

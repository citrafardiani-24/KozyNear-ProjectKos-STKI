/// <reference types="vite/client" />

// Vite expose `import.meta.env` dengan type info untuk env vars.
// Tambah type khusus untuk env var yang dipakai di app:
interface ImportMetaEnv {
  readonly VITE_API_URL: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}

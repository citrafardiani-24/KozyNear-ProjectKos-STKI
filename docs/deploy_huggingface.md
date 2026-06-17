# Deploy KozyNear ke Hugging Face Spaces

HF Spaces = free hosting untuk ML demos. CPU basic free tier:
- **16GB RAM** (32x Render's 512MB)
- **2 vCPU**
- Docker support (sama Dockerfile bisa pakai)
- Sleep setelah 48 jam idle (vs Render 15 menit) — better untuk demo

## Prerequisite

- Akun [HuggingFace](https://huggingface.co/join) (free, GitHub/Google login)
- Repo `KozyNear` di GitHub sudah punya Dockerfile + frontmatter di README

## Step 1 — Create Space (~3 menit, di mobile browser)

1. Login ke https://huggingface.co
2. Profile menu kanan atas → **New Space**
3. Form fields:
   - **Owner**: pilih kamu (DYmazeh)
   - **Space name**: `kozynear`
   - **License**: MIT
   - **Select the Space SDK**: tap **Docker** card
   - **Docker template**: tap **Blank** (kita pakai Dockerfile sendiri)
   - **Space hardware**: CPU basic (free) — **DEFAULT, jangan pilih GPU**
   - **Public/Private**: Public (untuk demo dosen + course rubric require public)
4. Tap **Create Space**

URL hasil: `https://huggingface.co/spaces/DYmazeh/kozynear`
Live app URL: `https://dymazeh-kozynear.hf.space`

## Step 2 — Connect ke GitHub Repo (~2 menit)

HF Spaces bisa import dari GitHub:

1. Di Space yang baru dibuat, tap tab **Files**
2. Tap **+ Contribute** atau opsi import
3. **Option A — Manual upload**: tap **Upload files** → upload semua file dari `D:/Project TKI Kos/`. Tedious untuk banyak file.
4. **Option B — Git push** (RECOMMENDED):
   ```bash
   # Di laptop kamu, dari D:/Project TKI Kos/
   git remote add huggingface https://huggingface.co/spaces/DYmazeh/kozynear
   git push huggingface main
   ```
   HF Space punya git repo built-in. Push trigger build.

   ⚠️ HF butuh login token: ketik HF username + token (generate di
   https://huggingface.co/settings/tokens → "Write" permission).

## Step 3 — Set Environment Variables (~3 menit)

HF Spaces UI → tab **Settings** → scroll down ke **Variables and secrets**:

**Secrets (sensitive, encrypted)**:
- `DATABASE_URL`: pakai **Supabase** project `kozynear` (sudah dibuat,
  ref `giowoltyoawpoefpdmwc`, region Singapore, free tanpa expire 90 hari;
  bonus: Table Editor buat lihat isi data + SQL editor buat query manual).
  Cara ambil connection string (~2 menit):
  1. https://supabase.com/dashboard/project/giowoltyoawpoefpdmwc
  2. Settings → Database → **Reset database password** → simpan passwordnya
  3. Tombol **Connect** (atas) → tab **Session pooler** → copy URI
     (format `postgresql://postgres.giowoltyoawpoefpdmwc:[PASSWORD]@aws-x-ap-southeast-1.pooler.supabase.com:5432/postgres`)
  4. Ganti `[PASSWORD]`, paste sebagai secret `DATABASE_URL` di HF
  - WAJIB **Session pooler (port 5432)**, BUKAN Transaction pooler (6543):
    asyncpg pakai prepared statements yang tidak kompatibel transaction mode.
    Jangan pakai Direct connection juga (host `db.*.supabase.co` IPv6-only).
  - DB sengaja dibiarkan kosong: boot pertama container otomatis jalankan
    `alembic upgrade head` + seed 227 listing. Setelah itu tabel `listings`
    kelihatan di Table Editor.
  - Catatan free tier: project pause setelah ~7 hari tanpa aktivitas;
    keepalive.yml sudah ping `/api/stats` (menyentuh DB). Kalau sempat
    pause, restore 1 klik di dashboard.
  - Alternatif fallback: Render PG lama (Dashboard → kozynear-db →
    Connections → **External** Database URL), tapi ingat expire 90 hari
    sejak dibuat.

**Variables (public, visible)**:
- `ENABLE_NEURAL`: `true` ← **kunci utama HF**: 16GB RAM cukup untuk MiniLM + Hybrid live
- `CORS_ORIGINS`: `["https://dymazeh-kozynear.hf.space"]`
- `ENVIRONMENT`: `production`
- `INDEXES_DIR`: `/app/data/indexes`
- `LOG_LEVEL`: `INFO`

Add satu per satu via UI: **+ New secret** / **+ New variable**.

Catatan: model ONNX MiniLM sudah di-pre-download saat docker build ke
`FASTEMBED_CACHE_PATH` (lihat Dockerfile), jadi cold start tidak download
ulang ~120MB.

## Step 4 — Wait for Build (~15-20 menit first time)

HF Spaces build Docker image dari Dockerfile:
- Pull layers dari Docker Hub (Python, Node base images)
- pip install dependencies (~5 min)
- Pre-download fastembed ONNX model (~2 min)
- npm install + vite build (~3 min)
- Final image push (~1 min)

Monitor via **Logs** tab. Spaces UI mirip Render, ada logs streaming.

⚠️ HF Spaces 1-jam build timeout. Kalau lewat → Build Fail. Solusi: keep
Docker layers small (already optimized).

## Step 5 — Verify Live

Setelah build done, status `Running`:

1. Tap **App** tab → embeds the live UI
2. Atau buka langsung: `https://dymazeh-kozynear.hf.space`
3. Test:
   - `/api/status` → JSON dengan `indexes.indobert_model_ready: true`
   - Search "kos putra dekat unila" + BM25 → results
   - Search same + IndoBERT → fast (~50ms)

## Step 6 — Update Repo Links

Update README.md + LAPORAN.md untuk point ke HF Space URL:
- Replace `kozynear.onrender.com` → `dymazeh-kozynear.hf.space`

```bash
git commit -m "docs: update deploy URLs to HuggingFace Spaces"
git push origin main
```

## Common Issues

### Build hangs di "Building Docker image"

HF free tier kadang build slow. Tunggu, jangan cancel. Kalau >30 min, retry
via **Factory rebuild** di Settings.

### "Insufficient disk space"

HF Space disk free tier 50GB — should be enough untuk our Docker image
(~2GB). Kalau hit limit, drop large files dari image (e.g., synthetic
JSONL kalau gak perlu).

### Database connection refused

DATABASE_URL salah format. Render PG kasih `postgres://...` — works di
SQLAlchemy 2.0 setelah `config.py` normalize ke `postgresql+asyncpg://`.
Verify connection string dari Render Dashboard → PG service → Connection.

### CORS error di browser console

CORS_ORIGINS env var salah. Update value ke exact match HF Space URL:
```json
["https://dymazeh-kozynear.hf.space"]
```
Include `https://`, no trailing slash.

### Out of memory tetap kena

Unlikely di HF 16GB, tapi kalau iya:
- Reduce fastembed batch_size (di indobert.py)
- Disable IndoBERT temporarily via env `IR_DISABLE_INDOBERT=true`
- Upgrade HF Spaces ke CPU+memory paid tier (~$0.05/hour)

## Migration Checklist

- [ ] HF Space `kozynear` created (Docker SDK, Blank template, CPU basic)
- [ ] GitHub repo pushed via `git push huggingface main`
- [ ] Env vars set: `DATABASE_URL` (secret) + `ENABLE_NEURAL=true` + CORS dll
- [ ] Build done, status Running
- [ ] `/api/status` shows indexes loaded (tfidf/bm25/indobert/hybrid) + `indobert_model_ready: true` + memory.rss_mb wajar (<2GB)
- [ ] Search test: smart (default), BM25, Neural MiniLM, Hybrid
- [ ] Tab Evaluasi / Preprocessing / Statistik tampil datanya
- [ ] README + LAPORAN updated dengan HF URL (sudah, verifikasi saja)
- [ ] Keep Render service sebagai backup URL (keepalive.yml ping dua-duanya)

## Cost Comparison

| Provider | Tier | RAM | Always-on | Cost |
|----------|------|-----|-----------|------|
| Render free | Web Service | 512MB | Sleep 15min | $0 |
| Render Starter | Web Service | 2GB | Yes | $7/month |
| HF Spaces CPU basic | Docker | **16GB** | Sleep 48h | **$0** |
| HF Spaces CPU upgrade | Docker | 16-64GB | Yes | $0.05-0.30/hr |
| Fly.io free | shared-cpu-1x | 256MB-1GB | Yes | $0 |

**Recommended**: HF Spaces CPU basic free (16GB RAM, 48h sleep).

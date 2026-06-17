# Deploy Guide — Render.com

Step-by-step deploy TKI-KOS ke Render.com (free tier).

## Prerequisites

- [x] Repo public di GitHub: https://github.com/DYmazeh/TKI-KOS
- [x] Render account (sign up free di https://render.com)
- [x] Connect GitHub account ke Render (Account Settings -> Connect GitHub)
- [x] Scraping done -> data ready: `data/processed/corpus.json` exists
- [x] Indexes done: `data/indexes/{tfidf.pkl, bm25.pkl, indobert/}`

## Critical Question — How to Get Indexes ke Render?

Indexes (TF-IDF pickle, BM25 pickle, IndoBERT embeddings + FAISS) BUTUH ada
di Render file system supaya backend bisa load saat startup. Tiga opsi:

| Opsi | Pros | Cons |
|------|------|------|
| **A: Commit indexes ke git** | Simplest, auto-deploy via render.yaml | Repo bloat (~30-50MB), GitHub warning di 50MB |
| **B: Build saat deploy** | Repo clean | Build time lama (~5 min IndoBERT encode), bisa exceed Render free tier 15-min build timeout |
| **C: External storage (HF Datasets / R2 / GDrive)** | Repo clean, fleksibel | Setup complex, butuh download step di lifespan |

**Rekomendasi untuk student project: Opsi A**, asal total `data/indexes/` <
50MB. Untuk corpus 3000 listing dengan MiniLM, total biasanya 20-30MB.

### Step untuk Opsi A:

1. Temporary remove `data/indexes/*.pkl` etc dari `.gitignore`:
   ```diff
   -data/indexes/*.pkl
   -data/indexes/*.index
   -data/indexes/*.bin
   -data/indexes/*.faiss
   -data/indexes/*.npy
   ```

2. Commit indexes:
   ```bash
   git add data/indexes data/processed/corpus.json
   git commit -m "data: add prebuilt indexes for deploy"
   git push
   ```

3. NOTE: kalau kamu rebuild indexes (e.g., tune hyperparameter), harus commit
   ulang. Auto-deploy via render.yaml pickup yang ada di git.

## Step 1: Deploy via Blueprint (recommended)

`render.yaml` di root repo sudah ada — Render baca dan auto-create 3 services.

1. Login ke Render Dashboard
2. **New +** -> **Blueprint**
3. Connect repo `DYmazeh/TKI-KOS`
4. Render parse `render.yaml`, show preview 3 services:
   - `tki-kos-pg` (PostgreSQL)
   - `tki-kos-backend` (Web Service)
   - `tki-kos-frontend` (Static Site)
5. Click **Apply**

Render mulai provisioning. Order:
1. PG database (~30 detik)
2. Backend deploy (~5-10 menit — pip install heavy deps termasuk torch/sentence-transformers)
3. Frontend deploy (~2-3 menit — npm install + vite build)

## Step 2: Update CORS + API URL (post-deploy)

Setelah deploy selesai, Render kasih URL aktual seperti:
- Backend: `https://tki-kos-backend-xxxx.onrender.com`
- Frontend: `https://tki-kos-frontend-xxxx.onrender.com`

Update env vars di kedua service supaya saling kenal:

### Backend env: `CORS_ORIGINS`
1. Render Dashboard -> Backend service -> Environment tab
2. Edit `CORS_ORIGINS`:
   ```json
   ["https://tki-kos-frontend-xxxx.onrender.com"]
   ```
3. Save -> service auto-redeploy

### Frontend env: `VITE_API_URL`
1. Render Dashboard -> Frontend service -> Environment tab
2. Edit `VITE_API_URL`:
   ```
   https://tki-kos-backend-xxxx.onrender.com
   ```
3. Save -> trigger manual redeploy (env var change perlu rebuild)

## Step 3: Run Migrations (DB Schema)

Render PG empty by default. Buka **Shell** di backend service:

```bash
cd /opt/render/project/src/backend
alembic upgrade head
```

Output expected:
```
INFO  [alembic.runtime.migration] Running upgrade  -> 001, initial schema
```

## Step 4: Seed Database

Upload `data/raw/mamikos.jsonl` ke backend service (via Render Shell or curl
endpoint kalau ada upload route). Atau, kalau JSONL sudah di-commit:

```bash
cd /opt/render/project/src/backend
python -m scripts.seed_db --input ../data/raw/mamikos.jsonl
```

Output expected:
```
[load] 1542 listings raw
[filter] 1487/1542 listings pass quality bar
[insert] 1487 listings upserted
```

## Step 5: Verify Deploy

1. Hit health check:
   ```bash
   curl https://tki-kos-backend-xxxx.onrender.com/health
   # -> {"status":"ok","service":"tki-kos-backend"}
   ```

2. Hit Swagger UI:
   - Browser: `https://tki-kos-backend-xxxx.onrender.com/docs`
   - Test `/search?q=kos putra dekat unila&model=bm25`

3. Frontend:
   - Browser: `https://tki-kos-frontend-xxxx.onrender.com`
   - Should load TKI-KOS search UI
   - Try search "kos putra dekat unila"
   - Verify result cards muncul

## Cold Start Mitigation

Free tier backend spin down setelah 15 menit idle. Cold start ~30s + index
load ~30-60s = total 60-90s. **Bad untuk demo presentasi.**

### Option A: Keep service warm dengan UptimeRobot

1. Sign up free di https://uptimerobot.com
2. Add new monitor: HTTP(s), URL = `https://tki-kos-backend.onrender.com/health`, interval = 5 menit
3. Free tier UptimeRobot = 50 monitors, 5-min interval -> total 12 ping/hour cukup

**Cost:** UptimeRobot ping count towards Render free tier 750 hours/month
(720 hours = sebulan penuh). Aman tapi sempit. Kalau idle 14 jam/hari, total
~310 hours/month idle saved.

### Option B: Pre-warm sebelum demo

15 menit sebelum presentasi, akses backend URL di browser. Service spin up,
warm sampai ~30 menit setelah last request.

### Option C: Switch ke Render Starter plan ($7/month)

Always-on, no spin down. Worth it kalau project akan demo berulang kali.

## Troubleshooting

### Backend deploy fail: "out of memory" saat pip install

Render free tier hanya 512MB RAM. `sentence-transformers` + `faiss-cpu` +
`torch` (dependency) total ~600MB di build cache.

**Fix:** swap `requirements.txt` untuk separate `requirements-prod.txt`
tanpa heavy deps yang gak digunakan, atau upgrade ke paid plan.

### Backend startup timeout

Kalau index loading >5 menit, Render anggap unhealthy dan restart.

**Fix:**
- Pakai MiniLM (118MB) instead of IndoBERT-base (440MB)
- Precompute FAISS index, commit ke git supaya tinggal load dari disk
- Tambah `app.state.tfidf_loaded` flag dan return 503 dari `/search` sampai ready

### Frontend can't reach backend (CORS error di browser console)

**Fix:** verify `CORS_ORIGINS` di backend env match frontend URL exactly
(include `https://`, no trailing slash). Restart backend.

### Postgres free tier expired (after 90 days)

Render kasih warning 7 hari sebelum expire. Backup data:
```bash
pg_dump $DATABASE_URL > backup.sql
```

Create new free PG instance, restore:
```bash
psql $NEW_DATABASE_URL < backup.sql
```

Atau upgrade ke paid plan ($7/month basic) — recommended untuk after course
selesai kalau project mau di-keep up.

## Monitoring Post-Deploy

Render Dashboard kasih:
- **Logs**: real-time stream service logs (stdout/stderr)
- **Metrics**: CPU/RAM/Network usage (free tier limited)
- **Events**: deploy history, restarts, crashes

Tambah lagi:
- **UptimeRobot**: external uptime monitor
- **Sentry** (free tier): error tracking (kalau ada budget waktu di Week 4)

## Manual Re-Deploy

Setelah commit baru ke main, Render auto-deploy. Untuk **manual trigger**:
- Dashboard -> Service -> **Manual Deploy** -> **Deploy latest commit**

Useful kalau auto-deploy gagal dan kamu udah fix issue di repo.

## Summary Checklist

- [ ] PostgreSQL provisioned, connection string captured
- [ ] Backend deployed, `/health` returns 200
- [ ] Frontend deployed, accessible via public URL
- [ ] CORS_ORIGINS updated dengan frontend URL
- [ ] VITE_API_URL updated dengan backend URL
- [ ] Migrations run via Shell (`alembic upgrade head`)
- [ ] DB seeded (`python -m scripts.seed_db --input ...`)
- [ ] Indexes available (committed ke git atau download script)
- [ ] Search end-to-end works dari frontend
- [ ] UptimeRobot configured (kalau perlu warm)
- [ ] Screenshot UI live untuk laporan / slide
- [ ] URL ditulis di root README

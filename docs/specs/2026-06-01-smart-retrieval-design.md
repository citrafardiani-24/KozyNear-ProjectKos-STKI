# Design Spec: Smart Retrieval (Query Understanding + Geo + BM25)

Tanggal: 2026-06-01
Status: Draft (menunggu review user)
Konteks: UAS Temu Kembali Informasi (COM620321). Bukan skripsi. Deadline 17 Jun 2026.

## 1. Konteks & Tujuan

Sistem pencarian kos KozyNear sekarang mengirim seluruh query mentah ke satu model
teks (TF-IDF / BM25 / neural / hybrid), dengan filter (harga/tipe/kecamatan) sebagai
parameter terpisah dan TANPA ranking jarak, padahal tiap listing punya koordinat dan
inti use-case-nya adalah "cari kos dekat X + tampilkan di Maps".

Temuan: untuk query kos (pendek, padat atribut, hampir selalu ada lokasi), relevansi
sebenarnya didominasi **struktur (gender/harga/fasilitas) + kedekatan geografis**, yang
tidak ditangani embedding model neural. Bagian teksnya pendek dan keyword-heavy, di mana
BM25 sudah menang (P@5 0.61 vs neural 0.11 pada eval 15 query).

Tujuan: bangun pipeline retrieval bertingkat yang (a) memahami query jadi atribut +
anchor lokasi, (b) merangking dengan gabungan teks + geo + atribut, (c) ringan (tanpa
model neural di runtime, jadi muat di Render free tier 512MB), dan (d) transparan
(tiap langkah bisa dijelaskan saat presentasi, memenuhi syarat tugas "WAJIB dipahami").

## 2. Non-Goals (batas scope, YAGNI)

- TIDAK pakai Google Places / Geocoding API live (berbayar, butuh key, titik gagal).
- TIDAK enrich tiap listing dengan POI sekitar (future work).
- TIDAK fine-tune / ganti embedder neural (data label cuma 15 query, kekecilan).
- TIDAK handle multi-anchor kompleks; ambil anchor pertama yang dikenali.
- Model neural TETAP ADA di repo + notebook (untuk bab Model & Evaluasi), hanya tidak
  di-load di server produksi.

## 3. Arsitektur (alur data)

```
query "kos cewe ac deket unila murah"
  |
  v
[1. Query Parser] -> ParsedQuery {
      gender: "putri",            # hard filter
      harga_max: 1_000_000,       # hard filter (eksplisit) / heuristik "murah"
      fasilitas: ["ac"],          # boost
      anchor: {nama:"unila", lat,lng},  # boost (geo)
      residual_text: "kos"        # ke BM25
  }
  |
  v
[2. BM25] cari kandidat dari residual_text (overshoot 3x)
  |
  v
[3. Skor gabungan tiap kandidat]:
   skor = w_text*BM25 + w_geo*proximity(anchor) + w_attr*match(fasilitas)
   (tiap komponen di-min-max normalize per query dulu)
  |
  v
[4. Hard filter (gender, harga) -> sort -> top_k -> hydrate DB -> response]
   + fallback longgarin filter kalau hasil kosong
   + sertakan "understood": {gender, fasilitas, anchor, harga} untuk transparansi UI
```

## 4. Komponen (tiap unit satu tugas)

### 4.1 Query Parser (`app/search/query_parser.py`, baru)
- Input: string query. Output: `ParsedQuery` (dataclass / pydantic).
- Reuse:
  - `KOS_JARGON_DICT` ([app/preprocessing/jargon.py](../../backend/app/preprocessing/jargon.py)):
    `TYPE_SLANG` (gender), `LOCATIONS` (alias kampus/area), `ABBREVIATIONS` (fasilitas).
  - `extract_prices_inline()` ([app/preprocessing/normalizer.py](../../backend/app/preprocessing/normalizer.py))
    untuk harga eksplisit.
- Aturan:
  - gender: cari token TYPE_SLANG / {putra,putri,campur}. Bentrok -> drop gender.
  - harga: angka eksplisit -> harga_min/max. Qualifier "murah" tanpa angka ->
    harga_max default = Rp 1.000.000 (tunable, didokumentasikan sebagai heuristik).
  - fasilitas: cocokkan token ke vocabulary fasilitas (diturunkan dari `Listing.fasilitas`).
  - anchor: cocokkan ke gazetteer (4.2); alias fakultas (fmipa, ft unila) -> koordinat
    kampus induk. Kalau tak dikenal -> biarkan jadi residual_text (BM25 tetap cocokkan).
  - residual_text: sisa setelah token terkenali dilepas.
- Output `understood` ikut dikembalikan ke API untuk transparansi.

### 4.2 Landmark Gazetteer (`app/search/gazetteer.py`, baru)
- Kamus `nama_kanonik -> {lat, lng, aliases[]}` untuk ~30 tempat populer Bandar Lampung:
  kampus (UNILA + fakultas, ITERA, Teknokrat, Darmajaya, UIN, Polinela) + landmark
  (mall: MBK/Mall Boemi Kedaton, Chandra, Transmart, Central Plaza; RS: Abdul Moeloek,
  Urip Sumoharjo; pasar: Tugu, Bambu Kuning; terminal: Rajabasa).
- Koordinat dibangun SEKALI saat build via OpenStreetMap Nominatim (gratis, tanpa key),
  hasilnya disimpan statis di repo. Tidak ada API call saat runtime.
- Sumber awal koordinat kampus: angkat dari [scripts/enrich_geo.py](../../backend/scripts/enrich_geo.py)
  + [scripts/extract_mamikos_detail.py](../../backend/scripts/extract_mamikos_detail.py).
- Fungsi `haversine_km(lat1,lng1,lat2,lng2)` (angkat dari enrich_geo, jadi shared util).

### 4.3 Geo Scorer (bagian `app/search/ranker.py`)
- `proximity_score(listing, anchor) = 1 / (1 + haversine_km)` (0..1, makin dekat makin
  tinggi; sederhana, monotonic, mudah dijelaskan; tunable).
- Listing tanpa koordinat (`koordinat_lat/lng` null, ~1 listing) -> skor geo netral
  (0), jangan crash, masih bisa muncul lewat skor teks/atribut.
- Kalau query tak punya anchor -> w_geo = 0, bobotnya dialihkan ke teks + atribut.

### 4.4 Ranker / Fusion (`app/search/ranker.py`, baru)
- Input: kandidat BM25 + ParsedQuery. 
- `attr_score = (jumlah fasilitas diminta yang ada di listing.fasilitas) / (jumlah diminta)`.
- Normalisasi min-max per query untuk tiap komponen (pola dari `app/indexing/hybrid.py`);
  kalau semua sama (normalize jadi nol) -> komponen netral.
- `final = w_text*text + w_geo*geo + w_attr*attr`. Default `w_text=0.4, w_geo=0.4,
  w_attr=0.2` (tunable, akan di-grid-search di eval).
- Hard filter gender + harga eksplisit DITERAPKAN (buang yang tak cocok).
- Fallback: kalau hasil setelah hard filter kosong -> longgarkan (buang harga dulu, lalu
  gender), tandai `relaxed: [...]` di response.

### 4.5 Endpoint (`app/api/search.py`, dirombak)
- Tambah model baru `smart` (default produksi). Tetap sediakan `bm25`/`tfidf` untuk
  perbandingan. Hapus ketergantungan neural di jalur produksi.
- Response tambah field `understood` + `relaxed`.

## 5. Keputusan desain yang dikunci
- Filter: gender + harga eksplisit = **hard filter**; fasilitas + kedekatan = **boost**;
  fallback longgarkan kalau kosong. (disetujui user 2026-06-01)
- Anchor: kampus + ~30 landmark via gazetteer statis, no live API. (disetujui)
- Tampilan Maps: link/embed `https://www.google.com/maps?q=lat,lng` (gratis, tanpa key).

## 6. Edge cases / error handling
- Query tanpa anchor -> w_geo=0, redistribusi bobot.
- Kos tanpa koordinat -> geo netral, tidak crash.
- Parser kosong ("kos bagus") -> turun anggun ke perilaku BM25.
- Hard filter kosongkan hasil -> fallback longgarkan + tanda `relaxed`.
- Gender bentrok -> drop gender.
- "murah" tanpa angka -> harga_max default (heuristik terdokumentasi).
- Alias kampus/fakultas -> map ke koordinat induk; tak dikenal -> residual text.
- Multi-anchor -> ambil yang pertama dikenali (future: nearest-of-any).

## 7. Testing (pytest, reuse `backend/tests/`)
- Unit Query Parser: gender slang, fasilitas abbrev, alias kampus, pola harga + "murah",
  ekstraksi residual, kasus bentrok.
- Unit gazetteer + haversine: koordinat dikenal -> jarak benar (toleransi).
- Unit geo: makin dekat skor makin tinggi; koordinat hilang -> netral.
- Unit ranker: kos dekat+cocok ngalahin jauh; hard filter buang gender salah; fallback nyala.
- Integration `/search`: end-to-end 2-3 query via TestClient.

## 8. Evaluasi: cara membuktikan lebih baik (DUA LENSA, jujur)

Tambah pipeline `smart` sebagai sistem di harness eval, sejajar tfidf/bm25/indobert/hybrid.

**Lensa 1 - Relevansi teks (qrels lama):** P@5/MAP/NDCG di `eval/ground_truth.csv`.
Untuk perbandingan antar model teks (sudah ada).

**Caveat penting (pelajaran pooling bias jilid 2):** qrels lama dibangun untuk menilai
KECOCOKAN TEKS dari pool BM25/TF-IDF. Qrels itu belum tentu menghargai "lebih dekat
kampus" atau "gender benar". Jadi pipeline yang pintar geo+atribut bisa TIDAK terlihat
menang di P@5 lama, walau lebih berguna untuk user. Prinsip: metrik harus mengukur yang
dioptimalkan.

**Lensa 2 - Constraint Satisfaction @K (metrik baru, murah, jujur):** untuk query
berkendala ("kos putri deket unila < 1jt"), ukur persentase top-K yang BENAR-BENAR
memenuhi gender + harga + fasilitas + dalam X km dari anchor. BM25 mentah akan sering
melanggar; pipeline `smart` mestinya mendekati 100%. Tidak butuh re-anotasi pool.

Lapor KEDUANYA apa adanya: "BM25 unggul di relevansi teks; pipeline kami unggul di
constraint satisfaction (lebih dekat ke kebutuhan user nyata)."

Tuning bobot fusion via grid kecil (seperti eksperimen alpha), dengan caveat jujur:
15 query itu kecil -> rawan overfit, lapor sebagai indikatif bukan mutlak.

Eval hidup di notebook/Colab; pipeline importable dari `app`.

## 9. Dampak RAM / deploy
- Pipeline tidak load model neural -> RSS turun jauh, muat di Render free 512MB.
- Lanjutkan trim: buang `fastembed` + `faiss-cpu` dari `requirements-runtime.txt`
  (penghematan RAM nyata, gabung dengan split runtime/dev yang sudah staged).
- BM25 (rank-bm25, pure python) + parser (dict/regex) + geo (math) = nyaris nol overhead.
- Verifikasi setelah deploy via `GET /api/status` (RSS).

## 10. Aset existing yang direuse
- `Listing.koordinat_lat/lng`, `Listing.fasilitas`, `tipe`, `harga_per_bulan`
  ([app/models/listing.py](../../backend/app/models/listing.py)).
- `KOS_JARGON_DICT` + `extract_prices_inline()` (preprocessing).
- Koordinat kampus + `haversine_km` (scripts/enrich_geo.py).
- BM25 index ([app/indexing/bm25.py](../../backend/app/indexing/bm25.py)).
- Pola min-max normalize ([app/indexing/hybrid.py](../../backend/app/indexing/hybrid.py)).

## 11. Risiko & catatan jujur
- Gazetteer statis = cakupan terbatas pada landmark terkurasi; di luar itu turun ke BM25.
- Heuristik "murah" = ambang tetap, bukan personalisasi; didokumentasikan.
- Constraint Satisfaction butuh definisi konstrain per query (perlu nyusun set query uji
  berkendala; bisa pakai/luaskan 15 query eval).
- Bobot fusion hasil tuning 15 query = indikatif.

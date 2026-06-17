# Temuan Eksplorasi (12 Jun 2026)

Empat eksplorasi untuk MEMAHAMI sistem yang sudah dibangun (bukan menambah
fitur). Skrip: `backend/scripts/explore_*.py`. Catatan ini terpisah dari
LAPORAN.md.

## 1. Batas lexical vs semantic (`explore_lexical_vs_semantic.py`)

Query paraphrase yang sengaja hindari kosakata dokumen, relevansi via
oracle metadata (gender/geo/fasilitas/harga).

| mode | BM25 | TF-IDF | Neural |
|---|---|---|---|
| literal (berbagi kata) | 0.775 | 0.750 | 0.450 |
| semantic (paraphrase) | 0.700 | 0.725 | 0.550 |

Gap neural-BM25: literal -0.325 → semantic -0.150. **Keunggulan teoretis
semantic muncul persis di tempat yang diprediksi** (kosakata mismatch:
BM25 turun, neural naik), TAPI MiniLM terlalu lemah untuk benar-benar
menyalip. Kasus mencolok: "hunian muslimah" → neural 0.20→0.60 (paham
muslimah≈putri); "dekat itera" → neural 0.00 (geo TIDAK bisa dari teks,
ini ranah gazetteer). **Kesimpulan: keputusan membuang neural dari runtime
dan memakai smart (lexical + sinyal terstruktur) tepat untuk corpus ini.**

## 2. Error analysis (`explore_error_analysis.py`)

5 query AP-smart terendah dibaca manual. Temuan: **"kegagalan" smart
hampir semuanya artefak pooling bias, bukan kesalahan sistem.**
- q26 "kos putra labuhan ratu": smart kembalikan kos PUTRA (benar) tapi
  GT=? (tak ter-judge) → AP=0; BM25 top-5 semua PUTRI (gender SALAH)
  berlabel GT=0. Smart lebih benar, tapi jawabannya tak pernah masuk pool.
- q24/q03: smart kembalikan campur ≤budget (constraint benar) tapi GT=?;
  BM25 "menang" dgn dokumen yang justru langgar harga/gender tapi terlanjur
  ter-judge relevan (karena pool dibangun dari BM25).
**Kesimpulan: skor standard UNDERSTATE kualitas smart; CS@5 (bebas pool)
adalah lensa yang jujur. Justifikasi smart-as-live makin kuat.**

## 3. Sinyal popularitas (`explore_popularity_signal.py`)

| sinyal | Spearman rho vs label GT | p |
|---|---|---|
| view_count | -0.006 | 0.862 |
| available_room | -0.033 | 0.327 |

Blend popularitas ke smart: CS@5 datar lalu turun, MAP turun.
**Kesimpulan: kos populer ≠ kos relevan. Benar bahwa sinyal ini TIDAK
dipakai; tidak ada sinyal gratis yang terlewat.**

## 4. Stress-test query understanding (`explore_query_stress.py`)

| kategori | hasil |
|---|---|
| normal | semua constraint terekstrak ✓ |
| typo ringan ("dket unila wifii murh") | anchor+gender OK; wifi+murah hilang |
| typo berat ("univ lampng") | **anchor geo hilang** (nama kampus typo) |
| code-switch ("for girls near unila") | wifi+anchor OK; gender "girls" tak terbaca |
| over-specified (8 constraint) | semua terekstrak, tetap balik 5 hasil ✓ |
| minimalis/nonsense/kosong | fallback degenerate (termurah), tidak crash ✓ |
| `'; DROP TABLE...` | aman (diperlakukan teks; query DB parameterized) ✓ |

**Kesimpulan: degradasi anggun di mana-mana, tidak pernah crash, aman dari
injection. Titik rapuh = ketergantungan token persis (typo nama kampus
mematikan geo; kata gender Inggris tak dikenal). Perbaikan termurah kalau
mau: fuzzy-match alias gazetteer + tambah kata gender EN ke jargon.**
Bonus: fix bug "juta" terbukti jalan live ("maksimal 1,5 juta" → harga_max
1.500.000).

## Tindak lanjut eksplorasi 4 (sudah dikerjakan)

Dua titik rapuh ditambal + diverifikasi loop tertutup (stress-test ulang,
eval nol regресi, +10 test):
- **Gender code-switch EN**: "girls/boys/female/..." kini terbaca (di
  parser saja, BUKAN jargon dokumen — ablation §4.5 melarang). "for girls
  near unila" sekarang gender=putri ✓.
- **Fuzzy gazetteer**: typo nama kampus single-token (unilla→unila,
  itra→itera) kini ke-resolve via fallback difflib (cutoff 0.84, exact
  match tetap menang dulu = nol risiko di query ejaan benar). Guard:
  kata umum + token 4-huruf tidak memicu anchor palsu.
- Sisa batasan jujur: singkatan multi-kata ("univ lampng") tetap tak
  ter-anchor (mengejarnya berisiko false-positive).

## 5. Kuantifikasi pooling bias (`explore_pooling_bias.py`)

Operasionalisasi temuan #2 jadi angka: GT diagnostik dengan pool GABUNGAN
5 model (vs BM25-only), heuristik annotator yang sama, lalu skor standard
dibandingkan. Pool union rata-rata 38.1 dok/query.

| model | MAP@BM25-pool | MAP@union-pool | delta |
|---|---|---|---|
| **smart** | 0.359 | **0.431** | **+0.072** |
| neural | 0.044 | 0.107 | +0.064 |
| bm25 | 0.296 | 0.237 | -0.059 |
| hybrid | 0.285 | 0.230 | -0.055 |
| tfidf | 0.253 | 0.229 | -0.024 |

**Smart & neural paling ditekan pooling bias BM25** (delta positif besar:
jawaban benar mereka tadinya unjudged). BM25/hybrid/tfidf turun karena
kehilangan "home advantage" pool. Jarak smart vs BM25 MELEBAR dari +0.064
(BM25-pool) jadi +0.194 (union-pool). Caveat: GT union ini tetap pakai
heuristik annotator (sirkularitas heuristik≈filter-smart belum hilang),
jadi diagnostik, bukan headline. Tapi arah + magnitudonya menegaskan:
skor standard di laporan UNDERSTATE smart, dan kit anotasi manusia
(§6.2 laporan) adalah penutup penuhnya.

## Sintesis

Keempatnya bermuara ke satu kesimpulan: **desain sistem sudah benar; yang
membatasi angka headline adalah EVALUASI (pooling bias + GT simulasi),
bukan sistemnya.** Maka satu-satunya langkah bernilai tinggi yang tersisa
adalah anotasi manusia (kit sudah siap), sisanya poles. Keputusan berhenti
mengejar data terbukti tepat.

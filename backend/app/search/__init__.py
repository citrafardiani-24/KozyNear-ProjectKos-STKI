"""Smart retrieval: query understanding + geo + fusion ranking.

Modul:
- gazetteer.py: koordinat kampus + landmark (statis), haversine, lookup anchor
- query_parser.py: parse query jadi ParsedQuery (gender/harga/fasilitas/anchor)
- ranker.py: fusion skor teks(BM25) + geo + atribut, hard filter + fallback
- pipeline.py: orkestrasi smart_search (BM25 kandidat, fetch DB, rank)
"""

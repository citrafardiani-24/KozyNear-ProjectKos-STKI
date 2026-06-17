"""Smoke test endpoint insights (/api/preprocess, /api/stats) + eval file-based.

TestClient menjalankan lifespan (load pipeline + index dari data/). DB lokal
tidak dianggap tersedia: /api/stats harus fallback ke corpus.json.
"""
from fastapi.testclient import TestClient

from app.main import app


def test_preprocess_trace_endpoint():
    with TestClient(app) as client:
        res = client.get("/api/preprocess", params={"text": "Kos KM dlm AC dkt UNILA 800rb"})
        assert res.status_code == 200
        body = res.json()
        assert body["raw"].startswith("Kos")
        stage_names = [s["stage"] for s in body["stages"]]
        # 9 stage default semua aktif
        assert stage_names == [
            "strip_html", "normalize_whitespace", "extract_prices", "lowercase",
            "apply_jargon_dict", "correct_spelling", "tokenize",
            "remove_stopwords", "stem",
        ]
        assert body["processed"]  # non-empty
        assert 800000 in body["extracted_prices"]


def test_eval_summary_file_based():
    with TestClient(app) as client:
        res = client.get("/api/eval/summary")
        assert res.status_code == 200
        body = res.json()
        models = {m["model"] for m in body["standard"]}
        # smart harus ikut ter-evaluasi (model live)
        assert {"bm25", "smart", "tfidf"} <= models
        assert body["total_queries"] == 30
        assert body["constraints"]["mean_cs_at_5"]["smart"] > 0
        assert len(body["significance"]) == 10
        # standard diurutkan MAP desc
        maps = [m["map"] for m in body["standard"]]
        assert maps == sorted(maps, reverse=True)


def test_eval_per_query_drilldown():
    with TestClient(app) as client:
        res = client.get("/api/eval/query/q01")
        assert res.status_code == 200
        rows = res.json()
        assert {r["model"] for r in rows} >= {"bm25", "smart"}
        missing = client.get("/api/eval/query/q99")
        assert missing.status_code == 404


def test_stats_fallback_corpus_when_db_down():
    with TestClient(app) as client:
        res = client.get("/api/stats")
        assert res.status_code == 200
        body = res.json()
        assert body["total_listings"] == 227
        assert body["vocab_size"] and body["vocab_size"] > 100
        assert "smart" in body["models_loaded"]

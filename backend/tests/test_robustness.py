"""Test robustness fix dari eksplorasi 4: gender EN + fuzzy gazetteer."""
from app.search.gazetteer import Gazetteer
from app.search.query_parser import parse

GZ = Gazetteer.load()


# --- Gender bahasa Inggris (code-switch) ---
def test_gender_english_girls_putri():
    assert parse("boarding house for girls near unila", GZ).gender == "putri"


def test_gender_english_boys_putra():
    assert parse("kos for boys with wifi", GZ).gender == "putra"


def test_gender_english_female_putri():
    assert parse("female only room", GZ).gender == "putri"


def test_gender_id_slang_masih_jalan():
    # regресi: slang ID lama harus tetap bekerja
    assert parse("kos cewe deket unila", GZ).gender == "putri"
    assert parse("kost cowok murah", GZ).gender == "putra"


def test_gender_konflik_drop():
    # girls + boys -> ambigu -> None (mekanisme lama dipertahankan)
    assert parse("kos for boys and girls", GZ).gender is None


# --- Fuzzy gazetteer (typo nama kampus) ---
def test_fuzzy_typo_unila():
    a = GZ.lookup("kos dekat unilla")  # typo 1 huruf
    assert a is not None and a.name == "universitas lampung"


def test_fuzzy_typo_itera():
    a = GZ.lookup("kos deket itra murah")  # typo 'itera' -> 'itra'
    assert a is not None and a.name == "itera"


def test_exact_masih_menang_tanpa_fuzzy():
    a = GZ.lookup("kos dekat unila")
    assert a is not None and a.name == "universitas lampung"


def test_fuzzy_tidak_false_match_kata_umum():
    # kata umum (termasuk 4-huruf) TIDAK boleh memicu anchor palsu
    for q in ("kos nyaman bersih murah", "kamar mandi dalam parkir motor",
              "hunian strategis terjangkau", "kos baru saja ada dkat",
              "cari kos luas aman tipe putri"):
        assert GZ.lookup(q) is None, f"false anchor pada: {q!r}"


def test_fuzzy_bisa_dimatikan():
    assert GZ.lookup("kos dekat unilla", fuzzy=False) is None

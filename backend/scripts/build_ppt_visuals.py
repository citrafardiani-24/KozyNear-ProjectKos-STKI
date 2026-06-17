"""Generate gambar visual untuk PPT (tema gelap, cocok slide template):
1. comparison.png  : tabel "Pencarian Kata Kunci Biasa vs Smart" (kemampuan + CS@5)
2. stack.png       : diagram tech stack berlapis
3. data_flow.png   : alur akuisisi data (Google -> ekstrak -> quality gate -> 227)

Disimpan ke frontend/public/ supaya ikut ter-deploy dan punya URL publik
(https://dymazeh-kozynear.hf.space/<nama>.png) untuk di-upload ke Canva.

Jalankan: cd backend && python -m scripts.build_ppt_visuals
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "frontend" / "public"

BG = "#262626"      # dark slide bg
FG = "#F5F0E8"      # cream text
MUTE = "#B8B0A4"    # muted text
ACCENT = "#E8A98C"  # terracotta accent (dari template)
OK = "#86EFAC"      # hijau check
NO = "#F0A3A3"      # merah muted cross


def _save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor=BG,
                pad_inches=0.25)
    plt.close(fig)
    print(f"[saved] frontend/public/{name}")


def comparison():
    rows = [
        ("Paham gender (putri/putra)",        False, True),
        ("Paham batas budget",               False, True),
        ("Paham jarak ke kampus",            False, True),
        ("Tahan typo & bahasa campur",       False, True),
        ("Constraint Satisfaction@5",        "0,53", "0,87"),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")
    fig.patch.set_facecolor(BG)

    ax.text(0.2, 6.5, "Pencarian Biasa vs Smart", color=FG, fontsize=21,
            fontweight="bold", va="center")
    # header
    ax.text(4.9, 5.7, "Kata Kunci\nBiasa (BM25)", color=MUTE, fontsize=12,
            ha="center", va="center", linespacing=1.2)
    ax.text(7.6, 5.7, "Smart", color=ACCENT, fontsize=14, ha="center",
            va="center", fontweight="bold")

    y = 5.0
    for label, a, b in rows:
        is_score = isinstance(a, str)
        ax.plot([0.2, 9.6], [y + 0.42, y + 0.42], color="#3d3d3d", lw=1)
        ax.text(0.3, y, label, color=FG, fontsize=12.5, va="center")
        if is_score:
            ax.text(4.9, y, a, color=MUTE, fontsize=15, ha="center",
                    va="center", fontweight="bold")
            ax.text(7.6, y, b, color=OK, fontsize=19, ha="center",
                    va="center", fontweight="bold")
        else:
            ax.text(4.9, y, "✗", color=NO, fontsize=20, ha="center", va="center")
            ax.text(7.6, y, "✓", color=OK, fontsize=20, ha="center", va="center")
        y -= 0.92
    ax.text(0.3, y + 0.25, "CS@5 = proporsi 5 hasil teratas yang memenuhi "
            "semua kebutuhan user", color=MUTE, fontsize=9.5, va="center",
            style="italic")
    _save(fig, "comparison.png")


def stack():
    layers = [
        ("Frontend", "React 18 + TypeScript + Vite", ACCENT),
        ("Backend", "FastAPI + Python 3.11 (TF-IDF, BM25, MiniLM, Smart)", "#9DC3E6"),
        ("Data", "PostgreSQL (Supabase) + index FAISS / pickle", "#C5A3E0"),
        ("Deploy", "Docker -> HuggingFace Spaces + Render (backup)", "#E0C56E"),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 8.4); ax.axis("off")
    fig.patch.set_facecolor(BG)
    ax.text(0.2, 7.9, "Arsitektur & Tech Stack", color=FG, fontsize=21,
            fontweight="bold", va="center")
    y = 6.3
    for name, detail, c in layers:
        box = FancyBboxPatch((0.4, y - 0.6), 9.2, 1.25,
                             boxstyle="round,pad=0.02,rounding_size=0.12",
                             linewidth=0, facecolor="#303030")
        ax.add_patch(box)
        ax.add_patch(FancyBboxPatch((0.4, y - 0.6), 0.16, 1.25,
                     boxstyle="round,pad=0,rounding_size=0.05",
                     linewidth=0, facecolor=c))
        ax.text(0.95, y + 0.22, name, color=c, fontsize=15, fontweight="bold", va="center")
        ax.text(0.95, y - 0.25, detail, color=FG, fontsize=11.5, va="center")
        y -= 1.55
    _save(fig, "stack.png")


def data_flow():
    steps = [
        ("Penemuan tautan", "Google: site:mamikos\n313 slug ditemukan"),
        ("Ekstraksi detail", "Halaman detail Mamikos\n240 berhasil di-parse"),
        ("Quality gate", "Buang deskripsi kosong\n+ validasi field"),
        ("Corpus final", "227 kos nyata\n= 88,7% populasi"),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4.2); ax.axis("off")
    fig.patch.set_facecolor(BG)
    ax.text(0.2, 3.8, "Cara Data Didapat", color=FG, fontsize=20,
            fontweight="bold", va="center")
    x = 0.3; w = 2.6; gap = 0.45; y = 1.0
    for i, (title, body) in enumerate(steps):
        c = ACCENT if i == len(steps) - 1 else "#303030"
        tc = BG if i == len(steps) - 1 else FG
        box = FancyBboxPatch((x, y), w, 1.7,
                             boxstyle="round,pad=0.02,rounding_size=0.12",
                             linewidth=0, facecolor=c)
        ax.add_patch(box)
        ax.text(x + w / 2, y + 1.32, title, color=tc, fontsize=12.5,
                fontweight="bold", ha="center", va="center")
        ax.text(x + w / 2, y + 0.62, body, color=tc if i == len(steps) - 1 else MUTE,
                fontsize=10, ha="center", va="center", linespacing=1.3)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.05, y + 0.85),
                         (x + w + gap - 0.05, y + 0.85),
                         arrowstyle="-|>", mutation_scale=18, color=ACCENT, lw=2))
        x += w + gap
    _save(fig, "data_flow.png")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    comparison(); stack(); data_flow()
    return 0


if __name__ == "__main__":
    sys.exit(main())

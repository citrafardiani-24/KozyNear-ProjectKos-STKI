"""Convert LAPORAN.md ke LAPORAN.pdf dengan embedded charts.

Pakai markdown-pdf (pure Python, no external pandoc/LaTeX).

Usage (dari root project):
    cd D:/Project TKI Kos
    python backend/scripts/generate_laporan_pdf.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from markdown_pdf import MarkdownPdf, Section


def main() -> int:
    root = Path(__file__).resolve().parent.parent.parent
    laporan_md = root / "LAPORAN.md"
    laporan_pdf = root / "LAPORAN.pdf"

    if not laporan_md.exists():
        print(f"ERROR: {laporan_md} not found")
        return 1

    md_content = laporan_md.read_text(encoding="utf-8")

    pdf = MarkdownPdf(toc_level=2)
    pdf.meta["title"] = "LAPORAN AKHIR — KozyNear (STKI UNILA)"
    pdf.meta["author"] = "DYmazeh + AI Agent (Claude)"
    pdf.meta["subject"] = "Information Retrieval Final Project"

    # root parameter buat resolve relative image paths
    pdf.add_section(Section(md_content, toc=True, root=str(root)))
    pdf.save(str(laporan_pdf))

    size_kb = laporan_pdf.stat().st_size / 1024
    print(f"Saved {laporan_pdf} ({size_kb:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

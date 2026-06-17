"""Indonesian text preprocessing pipeline untuk kos listings.

Public API:
    from app.preprocessing import (
        PreprocessingPipeline, PipelineConfig, PipelineResult, preprocess,
        KOS_JARGON_DICT, jargon_count,
    )

Pipeline ORDER (PENTING):
1. HTML strip            (BeautifulSoup)
2. Whitespace normalize
3. Price extraction      (SEBELUM lowercase — preserve `Rp`)
4. Lowercase
5. Jargon dict substitution
6. Spelling correction
7. Tokenize
8. Stopword removal      (Sastrawi + custom)
9. Stem                  (Sastrawi StemmerFactory, cached)
"""

from .jargon import KOS_JARGON_DICT, jargon_count
from .pipeline import (
    PipelineConfig,
    PipelineResult,
    PreprocessingPipeline,
    preprocess,
)

__all__ = [
    "PreprocessingPipeline",
    "PipelineConfig",
    "PipelineResult",
    "preprocess",
    "KOS_JARGON_DICT",
    "jargon_count",
]

"""Indonesian preprocessing pipeline orchestrator.

Stage-based dengan toggle via PipelineConfig — tim Anggota B pakai untuk
benchmark `BEFORE vs AFTER tiap stage` di notebook 02_preprocessing_experiment.

Analogi Laravel: ini seperti chain of Middleware. Request (raw text) lewat
chain, tiap middleware transform.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .jargon import KOS_JARGON_DICT
from .normalizer import (
    extract_prices_inline,
    lowercase,
    normalize_whitespace,
    strip_html,
)
from .spelling import correct_spelling
from .stemmer import SastrawiStemmer
from .stopwords import StopwordRemover
from .tokenizer import simple_tokenize


@dataclass
class PipelineConfig:
    """Toggle setiap stage untuk experiment.

    Tim Anggota B: pakai config ini di notebooks/02_preprocessing_experiment.ipynb
    untuk benchmark IR metric BEFORE vs AFTER tiap stage.
    """
    strip_html: bool = True
    normalize_whitespace: bool = True
    extract_prices: bool = True  # always run — non-destructive
    lowercase: bool = True
    apply_jargon_dict: bool = True
    correct_spelling: bool = True
    tokenize: bool = True
    remove_stopwords: bool = True
    stem: bool = True
    custom_stopwords: list[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    """Hasil preprocessing dengan tracking per-stage."""
    raw: str
    processed: str
    tokens: list[str]
    extracted_prices: list[int] = field(default_factory=list)
    stages_applied: list[str] = field(default_factory=list)
    # Snapshot output tiap stage (untuk visualisasi /api/preprocess).
    # Diisi hanya kalau process(trace=True) — list of {stage, output}.
    trace: list[dict] = field(default_factory=list)


class PreprocessingPipeline:
    """Indonesian preprocessing pipeline untuk kos listings.

    Sastrawi factory + stopword set di-init sekali (heavy) — instance shared
    antar dokumen. Untuk batch processing, instantiate sekali, panggil
    `process()` per dokumen.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        jargon_dict: Optional[dict[str, str]] = None,
    ):
        self.config = config or PipelineConfig()
        self.jargon_dict = jargon_dict if jargon_dict is not None else KOS_JARGON_DICT
        # Lazy init heavy components
        self._stemmer: Optional[SastrawiStemmer] = (
            SastrawiStemmer() if self.config.stem else None
        )
        self._stopword_remover: Optional[StopwordRemover] = (
            StopwordRemover(custom=self.config.custom_stopwords)
            if self.config.remove_stopwords
            else None
        )
        # Pre-compile jargon patterns (sorted longest-first untuk hindari
        # "dlm" match dulu sebelum "km dlm")
        self._jargon_patterns: list[tuple[re.Pattern[str], str]] = []
        if self.jargon_dict:
            sorted_variants = sorted(self.jargon_dict.keys(), key=len, reverse=True)
            for variant in sorted_variants:
                pattern = re.compile(
                    r"\b" + re.escape(variant) + r"\b", re.IGNORECASE
                )
                self._jargon_patterns.append((pattern, self.jargon_dict[variant]))

    def process(self, text: str, trace: bool = False) -> PipelineResult:
        """Run full pipeline. Return PipelineResult dengan tracking stages.

        trace=True merekam snapshot output tiap stage (untuk visualisasi
        step-by-step di /api/preprocess); default False supaya jalur serving
        tidak bayar overhead.
        """
        result = PipelineResult(raw=text, processed=text or "", tokens=[])
        if not text:
            return result

        current = text

        def snap(stage: str, output) -> None:
            if trace:
                result.trace.append({"stage": stage, "output": output})

        # Stage 1: HTML strip
        if self.config.strip_html:
            current = strip_html(current)
            result.stages_applied.append("strip_html")
            snap("strip_html", current)

        # Stage 2: Whitespace normalize
        if self.config.normalize_whitespace:
            current = normalize_whitespace(current)
            result.stages_applied.append("normalize_whitespace")
            snap("normalize_whitespace", current)

        # Stage 3: Price extraction (PENTING — sebelum lowercase)
        if self.config.extract_prices:
            result.extracted_prices = extract_prices_inline(current)
            result.stages_applied.append("extract_prices")
            snap("extract_prices", result.extracted_prices)

        # Stage 4: Lowercase
        if self.config.lowercase:
            current = lowercase(current)
            result.stages_applied.append("lowercase")
            snap("lowercase", current)

        # Stage 5: Jargon dictionary substitution
        if self.config.apply_jargon_dict and self._jargon_patterns:
            current = self._apply_jargon(current)
            result.stages_applied.append("apply_jargon_dict")
            snap("apply_jargon_dict", current)

        # Stage 6: Spelling correction
        if self.config.correct_spelling:
            current = correct_spelling(current)
            result.stages_applied.append("correct_spelling")
            snap("correct_spelling", current)

        # Stage 7: Tokenize
        if self.config.tokenize:
            tokens = simple_tokenize(current)
            result.stages_applied.append("tokenize")
            snap("tokenize", list(tokens))
        else:
            tokens = current.split()

        # Stage 8: Stopword removal
        if self.config.remove_stopwords and self._stopword_remover:
            tokens = self._stopword_remover.remove(tokens)
            result.stages_applied.append("remove_stopwords")
            snap("remove_stopwords", list(tokens))

        # Stage 9: Stem
        if self.config.stem and self._stemmer:
            tokens = [self._stemmer.stem(tok) for tok in tokens]
            result.stages_applied.append("stem")
            snap("stem", list(tokens))

        result.tokens = tokens
        result.processed = " ".join(tokens)
        return result

    def _apply_jargon(self, text: str) -> str:
        """Replace jargon variants → canonical, longest-first dengan word boundary."""
        for pattern, canonical in self._jargon_patterns:
            text = pattern.sub(canonical, text)
        return text


def preprocess(text: str, config: Optional[PipelineConfig] = None) -> str:
    """Shorthand: run pipeline dengan config default, return processed string.

    Untuk batch lebih efisien, instantiate PreprocessingPipeline sekali,
    panggil `.process()` per dokumen.
    """
    pipeline = PreprocessingPipeline(config)
    return pipeline.process(text).processed

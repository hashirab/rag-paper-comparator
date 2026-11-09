"""Turn a research-paper PDF into a sequence of typed elements.

Flat-text chunking (just sliding a window over page text) destroys the one
thing that makes a paper a paper: its section structure, and the fact that
equations/tables/captions are not prose. This module classifies each line
of text using font metadata PyMuPDF already exposes (size + font name),
so later chunking can respect section boundaries and tag non-prose
elements instead of mashing everything into uniform windows.
"""
from __future__ import annotations

import collections
import re
from dataclasses import dataclass

import fitz

MATH_FONT_PREFIXES = ("CM", "MSAM", "MSBM", "ESINT", "LINE", "SFTT", "SFRM")
CAPTION_RE = re.compile(r"^(Table|Figure)\s+\d+", re.IGNORECASE)
TRAILING_EQ_NUM_RE = re.compile(r"\(\d+\)\s*$")
WATERMARK_RE = re.compile(r"^arXiv:\S+")


@dataclass
class Element:
    page: int
    text: str
    kind: str  # "heading" | "subheading" | "body" | "caption" | "equation"
    size: float


def _line_text_and_fonts(line: dict) -> tuple[str, list[str], list[float]]:
    text = "".join(span["text"] for span in line["spans"])
    fonts = [span["font"] for span in line["spans"]]
    sizes = [span["size"] for span in line["spans"]]
    return text, fonts, sizes


def _is_math_font(font: str) -> bool:
    return any(font.startswith(p) for p in MATH_FONT_PREFIXES)


_NUMERIC_TOKEN_RE = re.compile(r"^-?\d+(\.\d+)?%?$")


def _is_numeric_dense(text: str) -> bool:
    tokens = text.split()
    if len(tokens) < 6:
        return False
    numeric = sum(1 for t in tokens if _NUMERIC_TOKEN_RE.match(t.strip(",.")))
    return numeric / len(tokens) >= 0.35


def parse_pdf(path: str) -> list[Element]:
    doc = fitz.open(path)

    # Pass 1: collect every line with its dominant font/size so we can work
    # out what "body text" looks like for *this* paper (templates differ).
    raw_lines = []
    size_votes = collections.Counter()
    for page in doc:
        d = page.get_text("dict")
        for block in d["blocks"]:
            for line in block.get("lines", []):
                text, fonts, sizes = _line_text_and_fonts(line)
                stripped = text.strip()
                if not stripped or WATERMARK_RE.match(stripped):
                    continue
                # Stray diagram glyphs (arrows, bullets, lone symbols) carry
                # no retrievable meaning and otherwise get misread as headings
                # purely because they're rendered in a large font.
                if len(stripped) <= 3 and not any(ch.isalnum() for ch in stripped):
                    continue
                size = max(sizes, key=lambda s: sizes.count(s)) if sizes else 0.0
                size_votes[round(size, 1)] += len(stripped)
                raw_lines.append((page.number, stripped, fonts, sizes, size))

    body_size = size_votes.most_common(1)[0][0] if size_votes else 10.0

    elements: list[Element] = []
    for page_num, text, fonts, sizes, size in raw_lines:
        avg_size = sum(sizes) / len(sizes)
        math_frac = sum(1 for f in fonts if _is_math_font(f)) / max(len(fonts), 1)
        whole_line_bold = all("Medi" in f or "Bold" in f for f in fonts)

        if avg_size > body_size * 1.15 and len(text) < 100:
            kind = "heading"
        elif CAPTION_RE.match(text):
            kind = "caption"
        elif (
            whole_line_bold
            and len(text) < 90
            and body_size * 0.85 <= avg_size <= body_size * 1.15
        ):
            kind = "subheading"
        elif (math_frac > 0.5 or TRAILING_EQ_NUM_RE.search(text)) and len(text) < 120:
            kind = "equation"
        elif _is_numeric_dense(text):
            # A row/column of a results table, flattened to a single line of
            # text. PDF extraction destroys the 2D grid - these tokens are
            # real, but which model/column a given number belongs to is no
            # longer reliable from text order alone. Tag it so the prompt can
            # warn the model rather than let it state false-precise claims.
            kind = "table_data"
        else:
            kind = "body"

        elements.append(Element(page=page_num, text=text, kind=kind, size=avg_size))

    return elements

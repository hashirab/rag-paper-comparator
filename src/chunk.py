"""Group parsed elements into retrievable chunks along section boundaries.

Two deliberate departures from "split every N tokens":
1. A chunk never crosses a section/subsection boundary. The chunk carries
   its section path as metadata, so retrieval can cite "Section 3.2" and
   the LLM isn't stitching together text from two unrelated parts of the
   paper just because they happened to be adjacent in a fixed window.
2. Captions and equations are never merged into surrounding prose. They
   become their own chunk, tagged by kind, so a question like "what does
   Table 3 show" can match the caption directly instead of hoping it got
   swept into a 500-token prose window along with four paragraphs that
   have nothing to do with it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from parse import Element

MAX_CHUNK_CHARS = 1100


@dataclass
class Chunk:
    paper: str
    section: str
    subsection: str | None
    kind: str  # "prose" | "caption" | "equation"
    text: str
    page: int

    def citation(self) -> str:
        loc = self.section if not self.subsection else f"{self.section} → {self.subsection}"
        return f"{self.paper}, {loc} (p.{self.page + 1})"


def _looks_like_shouty_title(text: str) -> bool:
    letters = [ch for ch in text if ch.isalpha()]
    return bool(letters) and all(ch.isupper() for ch in letters) and len(text) < 100


def _merge_heading_run(elements: list[Element], start: int) -> tuple[str, int]:
    """Consecutive heading-kind lines (e.g. '2' then 'Methods') are one heading.

    Also absorbs a following ALL-CAPS title line even if its own font size
    didn't clear the heading threshold (e.g. ICLR-style '1' at 12pt followed
    by 'INTRODUCTION' rendered barely larger than body text).
    """
    parts = [elements[start].text]
    i = start + 1
    while i < len(elements) and elements[i].kind == elements[start].kind:
        parts.append(elements[i].text)
        i += 1
    if i < len(elements) and elements[i].kind == "body" and _looks_like_shouty_title(elements[i].text):
        parts.append(elements[i].text)
        i += 1
    return " ".join(parts), i


def chunk_elements(elements: list[Element], paper: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    section = "Front matter"
    subsection: str | None = None
    buf_text: list[str] = []
    buf_page: int | None = None
    buf_kind = "prose"

    def flush():
        nonlocal buf_text, buf_page, buf_kind
        if buf_text:
            chunks.append(
                Chunk(
                    paper=paper,
                    section=section,
                    subsection=subsection,
                    kind=buf_kind,
                    text=" ".join(buf_text).strip(),
                    page=buf_page or 0,
                )
            )
        buf_text, buf_page, buf_kind = [], None, "prose"

    i = 0
    while i < len(elements):
        el = elements[i]

        if el.kind == "heading":
            flush()
            section, i = _merge_heading_run(elements, i)
            subsection = None
            continue

        if el.kind == "subheading":
            flush()
            subsection, i = _merge_heading_run(elements, i)
            continue

        if el.kind in ("caption", "equation"):
            flush()
            chunks.append(
                Chunk(
                    paper=paper,
                    section=section,
                    subsection=subsection,
                    kind=el.kind,
                    text=el.text,
                    page=el.page,
                )
            )
            i += 1
            continue

        # body / table_data: accumulate runs of the same kind so a block of
        # consecutive table rows stays together as one "table_data" chunk
        # instead of being spliced into surrounding prose.
        new_kind = "prose" if el.kind == "body" else el.kind
        if buf_text and new_kind != buf_kind:
            flush()
        buf_kind = new_kind
        buf_page = buf_page if buf_page is not None else el.page
        buf_text.append(el.text)
        if sum(len(t) for t in buf_text) >= MAX_CHUNK_CHARS:
            flush()
        i += 1

    flush()
    return [c for c in chunks if c.text.strip()]

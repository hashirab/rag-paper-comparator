"""Parse both papers, chunk them, embed every chunk, and persist the index.

Run from the src/ directory:
    ../.venv/bin/python build_index.py
"""
from __future__ import annotations

import json
import os

import numpy as np

from chunk import chunk_elements
from embed import embed_texts
from parse import parse_pdf

PAPERS = [
    {"path": "../papers/rag_lewis2020.pdf", "name": "RAG (Lewis et al., 2020)"},
    {"path": "../papers/self_rag_asai2023.pdf", "name": "Self-RAG (Asai et al., 2023)"},
]

INDEX_DIR = "../index"


def main() -> None:
    all_chunks = []
    for paper in PAPERS:
        elements = parse_pdf(paper["path"])
        chunks = chunk_elements(elements, paper["name"])
        print(f"{paper['name']}: {len(chunks)} chunks "
              f"({sum(1 for c in chunks if c.kind == 'prose')} prose, "
              f"{sum(1 for c in chunks if c.kind == 'equation')} equation, "
              f"{sum(1 for c in chunks if c.kind == 'caption')} caption, "
              f"{sum(1 for c in chunks if c.kind == 'table_data')} table_data)")
        all_chunks.extend(chunks)

    texts = [c.text for c in all_chunks]
    print(f"Embedding {len(texts)} chunks...")
    vectors = embed_texts(texts)

    os.makedirs(INDEX_DIR, exist_ok=True)
    np.save(os.path.join(INDEX_DIR, "vectors.npy"), vectors)

    metadata = [
        {
            "paper": c.paper,
            "section": c.section,
            "subsection": c.subsection,
            "kind": c.kind,
            "text": c.text,
            "page": c.page,
        }
        for c in all_chunks
    ]
    with open(os.path.join(INDEX_DIR, "chunks.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved index: {vectors.shape[0]} vectors x {vectors.shape[1]} dims -> {INDEX_DIR}/")


if __name__ == "__main__":
    main()

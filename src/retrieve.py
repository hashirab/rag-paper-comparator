"""Flat numpy cosine-similarity search over the saved index.

No vector database, no ANN index - with 445 chunks a brute-force dot
product against a normalized matrix is sub-millisecond and exact, which
is the point of doing this without a framework: at this scale the
"vector database" is genuinely just a matrix multiply.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from embed import embed_texts

INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "index")


@dataclass
class Hit:
    score: float
    paper: str
    section: str
    subsection: str | None
    kind: str
    text: str
    page: int

    def citation(self) -> str:
        loc = self.section if not self.subsection else f"{self.section} → {self.subsection}"
        return f"{self.paper}, {loc} (p.{self.page + 1})"


class Index:
    def __init__(self):
        self.vectors = np.load(os.path.join(INDEX_DIR, "vectors.npy"))
        with open(os.path.join(INDEX_DIR, "chunks.json")) as f:
            self.chunks = json.load(f)
        assert len(self.chunks) == self.vectors.shape[0]

    def search(self, query: str, k: int = 6, paper: str | None = None) -> list[Hit]:
        q_vec = embed_texts([query])[0]
        sims = self.vectors @ q_vec  # vectors are normalized -> dot product == cosine sim

        candidate_idx = np.arange(len(self.chunks))
        if paper is not None:
            candidate_idx = np.array(
                [i for i in candidate_idx if paper.lower() in self.chunks[i]["paper"].lower()]
            )

        ranked = candidate_idx[np.argsort(-sims[candidate_idx])][:k]
        hits = []
        for i in ranked:
            c = self.chunks[i]
            hits.append(
                Hit(
                    score=float(sims[i]),
                    paper=c["paper"],
                    section=c["section"],
                    subsection=c["subsection"],
                    kind=c["kind"],
                    text=c["text"],
                    page=c["page"],
                )
            )
        return hits

    def search_each_paper(self, query: str, k_per_paper: int = 5) -> list[Hit]:
        """Retrieve top-k independently from each paper, so a comparison
        question can't end up with all its evidence pulled from one paper
        just because that paper happens to discuss the topic more."""
        papers = sorted(set(c["paper"] for c in self.chunks))
        hits: list[Hit] = []
        for p in papers:
            hits.extend(self.search(query, k=k_per_paper, paper=p))
        return hits

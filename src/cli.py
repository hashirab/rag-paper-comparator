"""Ask questions that get answered by retrieving from both papers and
reconciling what they say, with citations.

Usage:
    ../.venv/bin/python cli.py "how does Self-RAG decide when to retrieve, versus RAG?"
    ../.venv/bin/python cli.py --k 4 "what training data do they each use?"
"""
from __future__ import annotations

import argparse

from compare import answer
from retrieve import Index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Question to ask across both papers")
    parser.add_argument("--k", type=int, default=5, help="Chunks retrieved per paper")
    parser.add_argument("--show-context", action="store_true", help="Print retrieved chunks before the answer")
    args = parser.parse_args()

    index = Index()
    hits = index.search_each_paper(args.question, k_per_paper=args.k)

    if args.show_context:
        print("=== Retrieved context ===")
        for h in hits:
            print(f"[{h.score:.3f}] {h.citation()} ({h.kind})")
            print(f"    {h.text[:160]}")
        print()

    print("=== Answer ===")
    print(answer(args.question, hits))


if __name__ == "__main__":
    main()

"""Public entry point for the Smart Inquiry Triage Assistant.

`triage_inquiry` is what the Streamlit frontend calls. The knowledge base and
the compiled graph are built once and reused, so only the first inquiry of a
session pays the startup cost.
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from typing import Iterable

from . import config
from .graph import build_graph
from .knowledge_base import KnowledgeBase


@lru_cache(maxsize=1)
def _pipeline():
    """Build (and ingest, if needed) the knowledge base and graph once."""
    kb = KnowledgeBase()
    kb.ingest()
    return kb, build_graph(kb)


def triage_inquiry(
    query: str,
    top_k: int = config.DEFAULT_TOP_K,
    confidence_threshold: float = config.DEFAULT_CONFIDENCE_THRESHOLD,
    exclude_ids: Iterable[str] = (),
) -> dict:
    """Triage one customer inquiry.

    Returns the fixed output format required by the case study, plus a
    `confidence_parts` breakdown that the UI shows for explainability.
    """
    _kb, graph = _pipeline()

    final = graph.invoke(
        {
            "query": query,
            "top_k": top_k,
            "confidence_threshold": confidence_threshold,
            "exclude_ids": tuple(exclude_ids),
        }
    )

    return {
        "query": query,
        "category": final.get("category", "other"),
        "priority": final.get("priority", "medium"),
        "routed_queue": final.get("routed_queue", ""),
        "confidence": final.get("confidence", 0.0),
        "resolution_notes": final.get("resolution_notes", ""),
        "retrieved_past_cases": [
            n.case.inquiry_text for n in final.get("neighbours", [])
        ],
        "escalated": final.get("escalated", True),
        # Extra, not part of the required format: shown in the UI so a human
        # reviewer can see why the confidence landed where it did.
        "confidence_parts": final.get("confidence_parts", {}),
        "retrieved_details": [
            {
                "case_id": n.case.case_id,
                "text": n.case.inquiry_text,
                "category": n.case.category,
                "priority": n.case.priority,
                "similarity": round(n.similarity, 4),
            }
            for n in final.get("neighbours", [])
        ],
    }


def main() -> None:
    """Command-line use, handy for smoke tests and for the ingestion step."""
    parser = argparse.ArgumentParser(description="Smart Inquiry Triage")
    parser.add_argument("query", nargs="?", help="customer inquiry to triage")
    parser.add_argument("--top-k", type=int, default=config.DEFAULT_TOP_K)
    parser.add_argument(
        "--threshold", type=float, default=config.DEFAULT_CONFIDENCE_THRESHOLD
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="build the vector store and exit",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="with --ingest, rebuild from scratch",
    )
    args = parser.parse_args()

    if args.ingest:
        kb = KnowledgeBase()
        written = kb.ingest(force=args.force)
        print(f"ingested {written} cases into {config.CHROMA_DIR}")
        return

    if not args.query:
        parser.error("provide an inquiry, or use --ingest")

    result = triage_inquiry(args.query, args.top_k, args.threshold)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

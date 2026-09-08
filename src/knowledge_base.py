"""Data ingestion and retrieval.

Loads the historical cases and the taxonomy, embeds the case texts with the
local embedding model, and stores them in a persistent ChromaDB collection.

The routing table is derived from the data rather than hard-coded: in
past_cases.csv every category maps to exactly one queue (verified for all
300 rows), so routing is a lookup, not a prediction.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from typing import Iterable

import chromadb
from langchain_ollama import OllamaEmbeddings

from . import config


# --- Types ---------------------------------------------------------------

@dataclass(frozen=True)
class PastCase:
    """One historical inquiry with its human-assigned labels."""

    case_id: str
    inquiry_text: str
    category: str
    priority: str
    routed_queue: str


@dataclass(frozen=True)
class Neighbour:
    """A retrieved past case together with its similarity to the query."""

    case: PastCase
    similarity: float  # 0.0 - 1.0, higher is more similar


# --- Loading -------------------------------------------------------------

def load_past_cases() -> list[PastCase]:
    """Read past_cases.csv into typed records."""
    with open(config.PAST_CASES_CSV, newline="", encoding="utf-8") as fh:
        return [
            PastCase(
                case_id=row["case_id"],
                inquiry_text=row["inquiry_text"],
                category=row["category"],
                priority=row["priority"],
                routed_queue=row["routed_queue"],
            )
            for row in csv.DictReader(fh)
        ]


def load_taxonomy() -> list[dict]:
    """Read the canonical category list."""
    with open(config.TAXONOMY_JSON, encoding="utf-8") as fh:
        return json.load(fh)["categories"]


def build_routing_table(cases: Iterable[PastCase]) -> dict[str, str]:
    """Derive category -> queue from the historical data.

    Raises if a category maps to more than one queue, so that a change in the
    data surfaces as a loud failure instead of a silently wrong route.
    """
    table: dict[str, str] = {}
    for case in cases:
        existing = table.get(case.category)
        if existing is not None and existing != case.routed_queue:
            raise ValueError(
                f"category {case.category!r} maps to both {existing!r} and "
                f"{case.routed_queue!r}; routing is no longer deterministic"
            )
        table[case.category] = case.routed_queue
    return table


# --- Embeddings ----------------------------------------------------------

def get_embeddings() -> OllamaEmbeddings:
    """The local embedding model, shared by ingestion and query time."""
    return OllamaEmbeddings(
        model=config.EMBED_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )


# --- Vector store --------------------------------------------------------

class KnowledgeBase:
    """Persistent vector store over the historical cases.

    Chroma is used directly rather than through a LangChain wrapper so that
    the ingestion step and the distance-to-similarity conversion stay
    explicit and easy to explain.
    """

    def __init__(self) -> None:
        self.cases = load_past_cases()
        self.taxonomy = load_taxonomy()
        self.routing_table = build_routing_table(self.cases)
        self._by_id = {c.case_id: c for c in self.cases}

        self._embeddings = get_embeddings()
        self._client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        # Cosine distance: the embeddings are not length-normalised, and we
        # want angular similarity rather than raw magnitude.
        self._collection = self._client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # -- ingestion --

    def is_ingested(self) -> bool:
        return self._collection.count() == len(self.cases)

    def ingest(self, force: bool = False) -> int:
        """Embed and store every past case. Idempotent unless force=True."""
        if force:
            self._client.delete_collection(config.COLLECTION_NAME)
            self._collection = self._client.get_or_create_collection(
                name=config.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        elif self.is_ingested():
            return 0

        texts = [c.inquiry_text for c in self.cases]
        vectors = self._embeddings.embed_documents(texts)
        self._collection.upsert(
            ids=[c.case_id for c in self.cases],
            documents=texts,
            embeddings=vectors,
            metadatas=[
                {
                    "category": c.category,
                    "priority": c.priority,
                    "routed_queue": c.routed_queue,
                }
                for c in self.cases
            ],
        )
        return len(self.cases)

    # -- retrieval --

    def search(
        self,
        query: str,
        top_k: int,
        exclude_ids: Iterable[str] = (),
    ) -> list[Neighbour]:
        """Return the Top-K most similar past cases.

        `exclude_ids` exists for evaluation: when scoring a case that is
        itself in the store, it must not be allowed to retrieve itself.
        """
        exclude = set(exclude_ids)
        # Over-fetch so that exclusions cannot shrink the result below top_k.
        n_fetch = min(top_k + len(exclude), self._collection.count())
        if n_fetch == 0:
            return []

        vector = self._embeddings.embed_query(query)
        result = self._collection.query(
            query_embeddings=[vector],
            n_results=n_fetch,
            include=["distances"],
        )

        neighbours: list[Neighbour] = []
        for case_id, distance in zip(result["ids"][0], result["distances"][0]):
            if case_id in exclude:
                continue
            neighbours.append(
                Neighbour(
                    case=self._by_id[case_id],
                    similarity=cosine_distance_to_similarity(distance),
                )
            )
            if len(neighbours) == top_k:
                break
        return neighbours


def cosine_distance_to_similarity(distance: float) -> float:
    """Chroma returns cosine distance in [0, 2]; map it to similarity [0, 1].

    Clamped because floating-point noise can push the value marginally
    outside the theoretical range.
    """
    return max(0.0, min(1.0, 1.0 - (distance / 2.0)))

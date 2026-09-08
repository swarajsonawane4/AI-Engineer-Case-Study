"""The individual steps of the triage pipeline.

Each function here is one node in the LangGraph workflow. They are plain
functions over the shared state so that every step can be unit-tested and
explained on its own.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, TypedDict

from langchain_ollama import ChatOllama

from . import config
from .knowledge_base import KnowledgeBase, Neighbour


# --- Shared state --------------------------------------------------------

class TriageState(TypedDict, total=False):
    """State passed between graph nodes."""

    # Inputs
    query: str
    top_k: int
    confidence_threshold: float
    exclude_ids: tuple[str, ...]  # used by the evaluation harness only

    # Produced by the two parallel branches
    llm_category: str
    neighbours: list[Neighbour]

    # Produced by the fuse step onwards
    category: str
    confidence: float
    confidence_parts: dict[str, float]
    agreed: bool
    priority: str
    routed_queue: str
    resolution_notes: str
    escalated: bool


# --- Model handle --------------------------------------------------------

def get_chat_model(json_mode: bool = False) -> ChatOllama:
    """The local chat model. JSON mode is used for the classifier."""
    return ChatOllama(
        model=config.CHAT_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=config.CHAT_TEMPERATURE,
        format="json" if json_mode else None,
    )


# --- 1. Classify ---------------------------------------------------------

CLASSIFY_PROMPT = """You are a triage classifier for customer inquiries at an \
automotive company. Assign the inquiry to exactly one category.

Categories:
{taxonomy}

Customer inquiry:
"{query}"

Reply with JSON only, in this exact shape:
{{"category": "<one category name from the list above>"}}"""


def _taxonomy_block(kb: KnowledgeBase) -> str:
    """Compact category descriptions for the classifier prompt."""
    lines = []
    for entry in kb.taxonomy:
        # The full descriptions are long; the first sentence plus keywords
        # carries the distinguishing information and keeps the prompt small
        # enough for a 3B model to attend to reliably.
        first_sentence = entry["description"].split(". ")[0].strip()
        keywords = ", ".join(entry["keywords"][:6])
        lines.append(f"- {entry['name']}: {first_sentence}. Keywords: {keywords}.")
    return "\n".join(lines)


def make_classify_node(kb: KnowledgeBase):
    """Classify the inquiry using the taxonomy alone.

    This node deliberately does NOT see the retrieved neighbours. Keeping the
    classifier independent of retrieval is what makes agreement between the
    two a meaningful confidence signal rather than a circular one.
    """
    valid = {entry["name"] for entry in kb.taxonomy}
    taxonomy_text = _taxonomy_block(kb)
    model = get_chat_model(json_mode=True)

    def classify(state: TriageState) -> dict[str, Any]:
        prompt = CLASSIFY_PROMPT.format(
            taxonomy=taxonomy_text, query=state["query"]
        )
        raw = model.invoke(prompt).content
        category = _parse_category(raw, valid)
        return {"llm_category": category}

    return classify


def _parse_category(raw: str, valid: set[str]) -> str:
    """Pull a valid category out of the model's reply.

    Small models occasionally wrap the JSON or return a near-miss label, so
    parsing degrades gracefully instead of raising: an empty string here is
    handled by the fuse step, which falls back to the retrieved neighbours.
    """
    try:
        parsed = json.loads(raw)
        candidate = str(parsed.get("category", "")).strip().lower()
        if candidate in valid:
            return candidate
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass

    lowered = str(raw).lower()
    for name in valid:
        if name in lowered:
            return name
    return ""


# --- 2. Retrieve Top-K ---------------------------------------------------

def make_retrieve_node(kb: KnowledgeBase):
    """Similarity search over the historical cases."""

    def retrieve(state: TriageState) -> dict[str, Any]:
        neighbours = kb.search(
            state["query"],
            top_k=state.get("top_k", config.DEFAULT_TOP_K),
            exclude_ids=state.get("exclude_ids", ()),
        )
        return {"neighbours": neighbours}

    return retrieve


# --- 3. Fuse: final category + confidence --------------------------------

def normalise_similarity(similarity: float) -> float:
    """Stretch raw cosine similarity onto a usable [0, 1] scale.

    The embedding model never returns values near 0 or 1 for this corpus, so
    the raw number is a poor confidence term. The bounds in config were
    measured on the actual data rather than assumed; see README.
    """
    span = config.SIMILARITY_CEILING - config.SIMILARITY_FLOOR
    scaled = (similarity - config.SIMILARITY_FLOOR) / span
    return max(0.0, min(1.0, scaled))


def _weighted_category_vote(neighbours: list[Neighbour]) -> tuple[str, float]:
    """Similarity-weighted vote over neighbour categories.

    Returns the winning category and the share of total weight behind it.
    """
    if not neighbours:
        return "", 0.0
    scores: dict[str, float] = defaultdict(float)
    for n in neighbours:
        scores[n.case.category] += n.similarity
    total = sum(scores.values())
    winner = max(scores, key=scores.get)
    return winner, (scores[winner] / total if total else 0.0)


def fuse(state: TriageState) -> dict[str, Any]:
    """Combine the classifier and the neighbours into a category + confidence.

    Three signals feed the confidence score:

      cross-check  do the two independent predictors agree?
      agreement    how much of the retrieved weight supports the final label?
      similarity   are the neighbours actually close to this inquiry?

    Each covers a failure the others miss. Similarity alone is high for a
    novel inquiry that merely resembles old ones; agreement alone is high
    when retrieval is confidently wrong; the cross-check catches both but is
    binary, so on its own it is too coarse to threshold against.
    """
    neighbours = state.get("neighbours", [])
    llm_category = state.get("llm_category", "")
    knn_category, agreement = _weighted_category_vote(neighbours)

    # Prefer the classifier: it reads the taxonomy definitions, while
    # retrieval only knows what past inquiries looked like. On disagreement
    # the cross-check term collapses, which is what drives escalation.
    category = llm_category or knn_category
    agreed = bool(llm_category and knn_category and llm_category == knn_category)

    mean_similarity = (
        sum(normalise_similarity(n.similarity) for n in neighbours) / len(neighbours)
        if neighbours
        else 0.0
    )
    # Recompute agreement against the category actually chosen.
    if neighbours and category:
        total = sum(n.similarity for n in neighbours)
        supporting = sum(
            n.similarity for n in neighbours if n.case.category == category
        )
        agreement = supporting / total if total else 0.0

    parts = {
        "cross_check": 1.0 if agreed else 0.0,
        "agreement": round(agreement, 4),
        "similarity": round(mean_similarity, 4),
    }
    confidence = (
        config.W_CROSSCHECK * parts["cross_check"]
        + config.W_AGREEMENT * parts["agreement"]
        + config.W_SIMILARITY * parts["similarity"]
    )
    return {
        "category": category,
        "agreed": agreed,
        "confidence": round(min(1.0, max(0.0, confidence)), 4),
        "confidence_parts": parts,
    }


# --- 4. Determine priority ----------------------------------------------

def determine_priority(state: TriageState) -> dict[str, Any]:
    """Derive priority from the retrieved neighbours.

    The data justifies doing this from neighbours rather than from category:
    every category in past_cases.csv spans multiple priorities, so the
    category alone cannot decide urgency.

    A safety bias is applied on top of the vote. Missing an urgent inquiry
    (a brake fault sitting in a low-priority queue) is far more costly than
    raising a routine one, so a strongly similar high-priority neighbour
    lifts the result by one level.
    """
    neighbours = state.get("neighbours", [])
    if not neighbours:
        return {"priority": "medium"}  # no evidence: sit in the middle

    scores: dict[str, float] = defaultdict(float)
    for n in neighbours:
        scores[n.case.priority] += n.similarity
    priority = max(scores, key=scores.get)

    has_urgent_match = any(
        n.case.priority == "high" and n.similarity >= config.SAFETY_BIAS_SIMILARITY
        for n in neighbours
    )
    if has_urgent_match and priority != "high":
        rank = min(config.PRIORITY_RANK[priority] + 1, 2)
        priority = config.PRIORITY_LEVELS[rank]

    return {"priority": priority}


# --- 5. Route ------------------------------------------------------------

def make_route_node(kb: KnowledgeBase):
    """Map category to queue.

    Deterministic by design. In past_cases.csv each category maps to exactly
    one queue across all 300 rows, so a model here could only introduce
    errors that a dictionary lookup cannot make.
    """
    fallback = kb.routing_table.get("other", "General Correspondence / Triage Queue")

    def route(state: TriageState) -> dict[str, Any]:
        queue = kb.routing_table.get(state.get("category", ""), fallback)
        return {"routed_queue": queue}

    return route


# --- 6. Resolution notes -------------------------------------------------

NOTES_PROMPT = """You are a customer service agent at an automotive company. \
Write a short internal resolution note for the agent who will handle this \
inquiry.

Inquiry: "{query}"
Category: {category}
Priority: {priority}
Assigned to: {queue}

Similar past inquiries:
{examples}

Write at most two short sentences, 35 words in total, describing the concrete \
next action for the agent. No greeting, no sign-off, no bullet points, and do \
not restate the inquiry."""


def make_notes_node():
    """Draft the 1-2 line resolution note."""
    model = get_chat_model()

    def notes(state: TriageState) -> dict[str, Any]:
        neighbours = state.get("neighbours", [])
        examples = (
            "\n".join(
                f"- {n.case.inquiry_text} (priority: {n.case.priority})"
                for n in neighbours
            )
            or "- none"
        )
        prompt = NOTES_PROMPT.format(
            query=state["query"],
            category=state.get("category", "unknown"),
            priority=state.get("priority", "medium"),
            queue=state.get("routed_queue", "unassigned"),
            examples=examples,
        )
        text = str(model.invoke(prompt).content).strip()
        return {"resolution_notes": _tidy_note(text)}

    return notes


MAX_NOTE_WORDS = 35


def _tidy_note(text: str) -> str:
    """Trim the note to the one or two lines the output format asks for.

    Small models tend to keep elaborating past the instruction, so the length
    is enforced here rather than trusted to the prompt.
    """
    cleaned = " ".join(text.split())
    sentences = [s.strip() for s in cleaned.split(". ") if s.strip()]

    kept: list[str] = []
    words = 0
    for sentence in sentences[:2]:
        n = len(sentence.split())
        if kept and words + n > MAX_NOTE_WORDS:
            break
        kept.append(sentence)
        words += n
        if words >= MAX_NOTE_WORDS:
            break

    note = ". ".join(kept).rstrip(".") + "."
    # A single runaway sentence still has to be cut somewhere.
    if len(note.split()) > MAX_NOTE_WORDS + 8:
        note = " ".join(note.split()[:MAX_NOTE_WORDS]).rstrip(",.;") + "."
    return note


# --- 7. Escalation decision ---------------------------------------------

def decide_escalation(state: TriageState) -> dict[str, Any]:
    """Flag for human review when confidence falls below the threshold."""
    threshold = state.get("confidence_threshold", config.DEFAULT_CONFIDENCE_THRESHOLD)
    return {"escalated": state.get("confidence", 0.0) < threshold}

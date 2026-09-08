"""The LangGraph triage workflow.

    START ──┬──> classify ──┐
            │               ├──> fuse ──> determine_priority ──> route
            └──> retrieve ──┘                                      │
                                                                   v
                       END <── escalation <── resolution_notes ────┘

Classification and retrieval fan out in parallel and join at `fuse`. This is
the reason the pipeline is a graph rather than a chain: the two steps do not
depend on each other, so running them as independent branches both saves a
round trip and, more importantly, keeps the classifier blind to the retrieved
neighbours. That independence is what turns their agreement into a usable
confidence signal.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .knowledge_base import KnowledgeBase
from .nodes import (
    TriageState,
    decide_escalation,
    determine_priority,
    fuse,
    make_classify_node,
    make_notes_node,
    make_retrieve_node,
    make_route_node,
)


def build_graph(kb: KnowledgeBase):
    """Wire the nodes together and compile the workflow."""
    builder = StateGraph(TriageState)

    builder.add_node("classify", make_classify_node(kb))
    builder.add_node("retrieve", make_retrieve_node(kb))
    builder.add_node("fuse", fuse)
    builder.add_node("determine_priority", determine_priority)
    builder.add_node("route", make_route_node(kb))
    builder.add_node("resolution_notes", make_notes_node())
    builder.add_node("escalation", decide_escalation)

    # Parallel branches, joined at fuse.
    builder.add_edge(START, "classify")
    builder.add_edge(START, "retrieve")
    builder.add_edge("classify", "fuse")
    builder.add_edge("retrieve", "fuse")

    # Sequential tail.
    builder.add_edge("fuse", "determine_priority")
    builder.add_edge("determine_priority", "route")
    builder.add_edge("route", "resolution_notes")
    builder.add_edge("resolution_notes", "escalation")
    builder.add_edge("escalation", END)

    return builder.compile()

"""
Smart Inquiry Triage Assistant — Streamlit chat interface.

The frontend was provided with the case study. It has been wired to the
triage pipeline in `src/main.py` and extended in two places:

  * the sidebar reports whether the vector store is warm and which models
    are in use, so the state of the system is visible during a demo;
  * each result can be expanded to show the retrieved neighbours with their
    similarity scores and the breakdown of the confidence figure.

Everything else, including the fixed output format, is unchanged.
"""

import sys
from pathlib import Path

import streamlit as st

# Make the project importable regardless of where streamlit is launched from.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.main import triage_inquiry as run_triage  # noqa: E402

# --- Configuration (sidebar controls) ------------------------------------

st.set_page_config(page_title="Smart Inquiry Triage", page_icon="📨")
st.title("📨 Smart Inquiry Triage Assistant")

with st.sidebar:
    st.header("Settings")
    # Defaults come from config so the UI matches what was measured:
    # Top-K=5 is where accuracy stops improving in the sweep (see README).
    top_k = st.slider(
        "Top-K past cases", min_value=1, max_value=10, value=config.DEFAULT_TOP_K
    )
    confidence_threshold = st.slider(
        "Confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=config.DEFAULT_CONFIDENCE_THRESHOLD,
        step=0.05,
    )

    st.divider()
    st.caption("**Models** (local, via Ollama)")
    st.caption(f"chat · `{config.CHAT_MODEL}`")
    st.caption(f"embeddings · `{config.EMBED_MODEL}`")


# --- Backend hook --------------------------------------------------------

@st.cache_resource(show_spinner="Warming up the knowledge base...")
def _warm_pipeline():
    """Build the vector store and compile the graph once per session.

    Without this the first inquiry would pay the full ingestion cost and
    Streamlit would repeat it on every rerun.
    """
    from src.main import _pipeline

    kb, _graph = _pipeline()
    return len(kb.cases)


def triage_inquiry(query: str, top_k: int, confidence_threshold: float) -> dict:
    """Run the triage pipeline for one inquiry."""
    _warm_pipeline()
    return run_triage(query, top_k=top_k, confidence_threshold=confidence_threshold)


# --- Result rendering -----------------------------------------------------

def render_result(result: dict) -> None:
    """Render a triage result in the fixed output format."""
    st.markdown(f"**query:** {result.get('query', '')}")
    st.markdown(f"**category:** {result.get('category', '')}")
    st.markdown(f"**priority:** {result.get('priority', '')}")
    st.markdown(f"**routed queue:** {result.get('routed_queue', '')}")
    st.markdown(f"**confidence:** {result.get('confidence', '')}")
    st.markdown(f"**resolution notes:** {result.get('resolution_notes', '')}")

    past_cases = result.get("retrieved_past_cases", [])
    st.markdown("**retrieved past cases:**")
    if past_cases:
        for case in past_cases:
            st.markdown(f"- {case}")
    else:
        st.markdown("- _none_")

    if result.get("escalated"):
        st.warning("⚠️ Escalated to human review (confidence below threshold).")

    _render_explanation(result)


def _render_explanation(result: dict) -> None:
    """Show how the confidence score was arrived at.

    Kept behind an expander so the required output format stays the first
    thing a reader sees.
    """
    parts = result.get("confidence_parts") or {}
    details = result.get("retrieved_details") or []
    if not parts and not details:
        return

    with st.expander("Why this result?"):
        if parts:
            st.markdown("**Confidence breakdown**")
            cols = st.columns(3)
            labels = [
                ("cross_check", "Classifier ↔ neighbours", config.W_CROSSCHECK),
                ("agreement", "Neighbour agreement", config.W_AGREEMENT),
                ("similarity", "Mean similarity", config.W_SIMILARITY),
            ]
            for col, (key, label, weight) in zip(cols, labels):
                col.metric(label, f"{parts.get(key, 0.0):.2f}", f"weight {weight:g}")

        if details:
            st.markdown("**Retrieved neighbours**")
            st.dataframe(
                [
                    {
                        "case": d["case_id"],
                        "similarity": d["similarity"],
                        "category": d["category"],
                        "priority": d["priority"],
                        "inquiry": d["text"],
                    }
                    for d in details
                ],
                hide_index=True,
                use_container_width=True,
            )


# --- Chat / session view --------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []  # list of {"query": str, "result": dict|None}

# Replay history for this session.
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(turn["query"])
    with st.chat_message("assistant"):
        if turn["result"] is not None:
            render_result(turn["result"])
        else:
            st.error(turn.get("error", "No result."))

# New inquiry input.
query = st.chat_input("Paste a customer inquiry...")
if query:
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Triaging..."):
                result = triage_inquiry(query, top_k, confidence_threshold)
            render_result(result)
            st.session_state.history.append({"query": query, "result": result})
        except Exception as exc:  # noqa: BLE001 - surface any failure in the UI
            msg = f"Triage failed: {exc}"
            st.error(msg)
            st.session_state.history.append(
                {"query": query, "result": None, "error": msg}
            )

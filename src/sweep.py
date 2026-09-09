"""Retrieval-only sweeps behind the constants in `config`.

Two numbers in the README are not produced by `src.evaluate`, because both
isolate the retrieval component and deliberately leave the chat model out:

  * the Top-K table, which is why DEFAULT_TOP_K is 5
  * the in-domain similarity distribution, which is where
    SAFETY_BIAS_SIMILARITY, SIMILARITY_FLOOR and SIMILARITY_CEILING come from

Both are leave-one-out over all 300 past cases: each case is looked up with
itself excluded, so it cannot retrieve its own answer. Only the embedding
model runs, so the whole thing takes a few seconds.

Usage:
    python -m src.sweep
"""

from __future__ import annotations

import statistics
from collections import defaultdict

from . import config
from .knowledge_base import KnowledgeBase, Neighbour

MAX_K = 10

# Text that has nothing to do with the domain, used to find the floor below
# which a similarity score means "no real match".
OFF_TOPIC = (
    "a good recipe for banana bread with walnuts",
    "the history of the Roman Republic in three paragraphs",
    "how do I train for a marathon",
)


def knn_predict(neighbours: list[Neighbour]) -> tuple[str, str]:
    """Category and priority from the neighbours alone, no chat model.

    Mirrors the vote in `nodes.determine_priority` and the neighbour half of
    `nodes.fuse`, including the safety bias, so the sweep measures the same
    retrieval behaviour the pipeline actually uses.
    """
    cats: dict[str, float] = defaultdict(float)
    pris: dict[str, float] = defaultdict(float)
    for n in neighbours:
        cats[n.case.category] += n.similarity
        pris[n.case.priority] += n.similarity

    priority = max(pris, key=pris.get)
    urgent = any(
        n.case.priority == "high" and n.similarity >= config.SAFETY_BIAS_SIMILARITY
        for n in neighbours
    )
    if urgent and priority != "high":
        rank = min(config.PRIORITY_RANK[priority] + 1, 2)
        priority = config.PRIORITY_LEVELS[rank]

    return max(cats, key=cats.get), priority


def main() -> None:
    kb = KnowledgeBase()
    kb.ingest()

    # One retrieval pass at the largest K, reused for every smaller K.
    retrieved = {
        case.case_id: kb.search(case.inquiry_text, top_k=MAX_K, exclude_ids=[case.case_id])
        for case in kb.cases
    }

    # --- similarity distribution ---
    top1 = sorted(retrieved[c.case_id][0].similarity for c in kb.cases)
    q1, _, q3 = statistics.quantiles(top1, n=4)
    print("In-domain top-1 similarity, leave-one-out over all 300 cases")
    print(f"  min {top1[0]:.4f}   p25 {q1:.4f}   median {statistics.median(top1):.4f}"
          f"   p75 {q3:.4f}   max {top1[-1]:.4f}")
    print(f"  -> SAFETY_BIAS_SIMILARITY is the p75: {config.SAFETY_BIAS_SIMILARITY}")

    print("\nOff-topic text, for comparison")
    for text in OFF_TOPIC:
        sims = [n.similarity for n in kb.search(text, top_k=5)]
        print(f"  top1 {sims[0]:.4f}  mean {sum(sims)/len(sims):.4f}  {text!r}")
    print(f"  -> SIMILARITY_FLOOR {config.SIMILARITY_FLOOR} sits just above this band,")
    print(f"     SIMILARITY_CEILING {config.SIMILARITY_CEILING} near the in-domain p95")

    # --- Top-K sweep ---
    print("\nTop-K sweep (retrieval only, no chat model)")
    print(f"  {'K':>3} {'category':>10} {'priority':>10} {'under-triage':>14}")
    for k in (1, 3, 5, 7, 10):
        n = len(kb.cases)
        cat_ok = pri_ok = under = 0
        for case in kb.cases:
            cat, pri = knn_predict(retrieved[case.case_id][:k])
            cat_ok += cat == case.category
            pri_ok += pri == case.priority
            under += config.PRIORITY_RANK[pri] < config.PRIORITY_RANK[case.priority]
        marker = "  <- default" if k == config.DEFAULT_TOP_K else ""
        print(f"  {k:>3} {cat_ok/n:9.1%} {pri_ok/n:10.1%} {under/n:13.1%}{marker}")

    # --- how often the safety bias fires ---
    print("\nSafety bias firing rate at Top-K = 5")
    true_high = sum(c.priority == "high" for c in kb.cases) / len(kb.cases)
    for threshold in (0.75, config.SAFETY_BIAS_SIMILARITY):
        fired = sum(
            any(n.case.priority == "high" and n.similarity >= threshold
                for n in retrieved[c.case_id][:5])
            for c in kb.cases
        )
        print(f"  threshold {threshold:<6} fires on {fired/len(kb.cases):.1%} of cases")
    print(f"  true high-priority rate in the data: {true_high:.1%}")


if __name__ == "__main__":
    main()

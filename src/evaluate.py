"""Offline evaluation of the triage pipeline.

The case study does not ask for an evaluation, but the presentation asks what
works well and what the limitations are, and the confidence score has to be
justified. Both answers are guesswork without measurement.

Method: leave-one-out over a stratified sample of past_cases.csv. Each sampled
case is triaged with itself excluded from retrieval, so it cannot retrieve its
own answer. Its human-assigned labels are the ground truth.

The resolution-notes step is skipped here: it is a generative step with no
ground truth to score against, and skipping it roughly halves the runtime.

Usage:
    python -m src.evaluate --sample 60
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

from . import config
from .knowledge_base import KnowledgeBase, PastCase
from .nodes import determine_priority, fuse, make_classify_node, make_route_node


def stratified_sample(cases: list[PastCase], n: int, seed: int) -> list[PastCase]:
    """Sample n cases while keeping the category mix of the full set."""
    rng = random.Random(seed)
    by_category: dict[str, list[PastCase]] = defaultdict(list)
    for case in cases:
        by_category[case.category].append(case)

    picked: list[PastCase] = []
    for category, group in sorted(by_category.items()):
        share = max(1, round(n * len(group) / len(cases)))
        picked.extend(rng.sample(group, min(share, len(group))))
    rng.shuffle(picked)
    return picked[:n]


def evaluate(sample_size: int, top_k: int, seed: int) -> dict:
    """Run leave-one-out evaluation and return a metrics dictionary."""
    kb = KnowledgeBase()
    kb.ingest()

    classify = make_classify_node(kb)
    route = make_route_node(kb)
    sample = stratified_sample(kb.cases, sample_size, seed)

    records = []
    for i, case in enumerate(sample, 1):
        neighbours = kb.search(
            case.inquiry_text, top_k=top_k, exclude_ids=[case.case_id]
        )
        state = {
            "query": case.inquiry_text,
            "neighbours": neighbours,
            **classify({"query": case.inquiry_text}),
        }
        state.update(fuse(state))
        state.update(determine_priority(state))
        state.update(route(state))

        records.append(
            {
                "case_id": case.case_id,
                "confidence": state["confidence"],
                "category_correct": state["category"] == case.category,
                "priority_correct": state["priority"] == case.priority,
                "priority_gap": abs(
                    config.PRIORITY_RANK[state["priority"]]
                    - config.PRIORITY_RANK[case.priority]
                ),
                "under_triaged": (
                    config.PRIORITY_RANK[state["priority"]]
                    < config.PRIORITY_RANK[case.priority]
                ),
                "queue_correct": state["routed_queue"] == case.routed_queue,
                "true_category": case.category,
                "pred_category": state["category"],
            }
        )
        print(f"\r  {i}/{len(sample)} cases", end="", flush=True)
    print()

    return summarise(records, top_k)


def summarise(records: list[dict], top_k: int) -> dict:
    """Turn per-case records into the numbers that go on a slide."""
    n = len(records)
    share = lambda key: sum(r[key] for r in records) / n  # noqa: E731

    # Does confidence actually track correctness? If the score is worth
    # thresholding against, accuracy must rise with it.
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        edge = min(int(r["confidence"] * 5) / 5, 0.8)
        buckets[f"{edge:.1f}-{edge + 0.2:.1f}"].append(r)
    calibration = {
        label: {
            "n": len(rows),
            "category_accuracy": round(
                sum(x["category_correct"] for x in rows) / len(rows), 3
            ),
        }
        for label, rows in sorted(buckets.items())
    }

    # The operational trade-off: escalate more, and what is left is safer.
    tradeoff = {}
    for threshold in (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        auto = [r for r in records if r["confidence"] >= threshold]
        tradeoff[f"{threshold:.1f}"] = {
            "auto_handled_share": round(len(auto) / n, 3),
            "category_accuracy_on_auto": (
                round(sum(r["category_correct"] for r in auto) / len(auto), 3)
                if auto
                else None
            ),
        }

    per_category: dict[str, dict] = {}
    by_true: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_true[r["true_category"]].append(r)
    for category, rows in sorted(by_true.items()):
        per_category[category] = {
            "n": len(rows),
            "accuracy": round(sum(x["category_correct"] for x in rows) / len(rows), 3),
        }

    return {
        "n_cases": n,
        "top_k": top_k,
        "chat_model": config.CHAT_MODEL,
        "embed_model": config.EMBED_MODEL,
        "category_accuracy": round(share("category_correct"), 3),
        "priority_exact_accuracy": round(share("priority_correct"), 3),
        "priority_within_one_level": round(
            sum(r["priority_gap"] <= 1 for r in records) / n, 3
        ),
        "under_triage_rate": round(share("under_triaged"), 3),
        "queue_accuracy": round(share("queue_correct"), 3),
        "confidence_calibration": calibration,
        "threshold_tradeoff": tradeoff,
        "per_category_accuracy": per_category,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the triage pipeline")
    parser.add_argument("--sample", type=int, default=60, help="cases to score")
    parser.add_argument("--top-k", type=int, default=config.DEFAULT_TOP_K)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="evaluation_results.json")
    args = parser.parse_args()

    metrics = evaluate(args.sample, args.top_k, args.seed)
    print(json.dumps(metrics, indent=2))

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()

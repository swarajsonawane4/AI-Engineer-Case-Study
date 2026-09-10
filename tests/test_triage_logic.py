"""Unit tests for the deterministic parts of the triage pipeline.

The split here is deliberate. Anything whose output is decided by arithmetic
or a lookup is tested with assertions; anything decided by a language model is
measured statistically by `src/evaluate.py` instead. Asserting an exact
category for a generated answer would be testing the model, not the code, and
it would fail for reasons that are not bugs.

None of these tests need Ollama running, so they are fast and can run in CI.
"""

from __future__ import annotations

import pytest

from src import config
from src.knowledge_base import (
    PastCase,
    Neighbour,
    build_routing_table,
    cosine_distance_to_similarity,
    load_past_cases,
    load_taxonomy,
)
from src.nodes import (
    _parse_category,
    _tidy_note,
    _weighted_category_vote,
    determine_priority,
    fuse,
    normalise_similarity,
)


# --- helpers -------------------------------------------------------------

def case(cid="CASE-0001", text="text", category="service",
         priority="low", queue="Service Scheduling Team") -> PastCase:
    return PastCase(cid, text, category, priority, queue)


def neighbour(similarity, category="service", priority="low") -> Neighbour:
    return Neighbour(case(category=category, priority=priority), similarity)


# --- routing -------------------------------------------------------------

class TestRouting:
    def test_table_is_derived_from_the_data(self):
        table = build_routing_table([
            case(category="billing", queue="Billing & Payments Team"),
            case(category="service", queue="Service Scheduling Team"),
        ])
        assert table == {
            "billing": "Billing & Payments Team",
            "service": "Service Scheduling Team",
        }

    def test_ambiguous_routing_is_a_loud_failure(self):
        """A category mapping to two queues must raise, not pick one."""
        with pytest.raises(ValueError, match="no longer deterministic"):
            build_routing_table([
                case(category="billing", queue="Billing & Payments Team"),
                case(category="billing", queue="Some Other Team"),
            ])

    def test_real_data_is_still_one_to_one(self):
        """The design rests on this being true; fail if the data changes."""
        table = build_routing_table(load_past_cases())
        assert len(table) == 8

    def test_every_taxonomy_category_can_be_routed(self):
        table = build_routing_table(load_past_cases())
        for entry in load_taxonomy():
            assert entry["name"] in table


# --- similarity ----------------------------------------------------------

class TestSimilarity:
    @pytest.mark.parametrize("distance,expected", [
        (0.0, 1.0),   # identical
        (1.0, 0.5),   # orthogonal
        (2.0, 0.0),   # opposite
    ])
    def test_distance_maps_to_similarity(self, distance, expected):
        assert cosine_distance_to_similarity(distance) == pytest.approx(expected)

    @pytest.mark.parametrize("distance", [-0.0001, 2.0001, -5.0, 99.0])
    def test_floating_point_noise_is_clamped(self, distance):
        assert 0.0 <= cosine_distance_to_similarity(distance) <= 1.0

    def test_rescaling_stretches_the_useful_band(self):
        """Off-topic text scores ~0.70-0.76 raw and must land near zero."""
        assert normalise_similarity(0.70) == 0.0
        assert normalise_similarity(config.SIMILARITY_FLOOR) == 0.0
        assert normalise_similarity(config.SIMILARITY_CEILING) == 1.0
        assert 0.0 < normalise_similarity(0.82) < 1.0

    def test_rescaling_is_monotonic(self):
        values = [normalise_similarity(x / 100) for x in range(60, 100)]
        assert values == sorted(values)


# --- category vote -------------------------------------------------------

class TestCategoryVote:
    def test_empty_neighbours_yield_no_winner(self):
        assert _weighted_category_vote([]) == ("", 0.0)

    def test_vote_is_weighted_by_similarity_not_count(self):
        """Two weak neighbours must not outvote one very strong one."""
        winner, _ = _weighted_category_vote([
            neighbour(0.95, "service"),
            neighbour(0.30, "billing"),
            neighbour(0.30, "billing"),
        ])
        assert winner == "service"

    def test_unanimous_neighbours_give_full_agreement(self):
        winner, agreement = _weighted_category_vote([
            neighbour(0.9, "billing"), neighbour(0.8, "billing"),
        ])
        assert winner == "billing"
        assert agreement == pytest.approx(1.0)


# --- priority ------------------------------------------------------------

class TestPriority:
    def test_no_evidence_sits_in_the_middle(self):
        assert determine_priority({"neighbours": []})["priority"] == "medium"

    def test_priority_follows_the_weighted_vote(self):
        state = {"neighbours": [
            neighbour(0.80, priority="low"),
            neighbour(0.79, priority="low"),
            neighbour(0.50, priority="high"),
        ]}
        assert determine_priority(state)["priority"] == "low"

    def test_a_very_similar_urgent_case_lifts_priority_one_level(self):
        """The safety bias: errors move to the cheap direction."""
        below = config.SAFETY_BIAS_SIMILARITY - 0.01
        above = config.SAFETY_BIAS_SIMILARITY + 0.01

        # The low-priority neighbour must win the vote on its own, so that
        # any change comes from the bias rather than from the vote itself.
        quiet = {"neighbours": [
            neighbour(0.95, priority="low"), neighbour(below, priority="high"),
        ]}
        loud = {"neighbours": [
            neighbour(0.95, priority="low"), neighbour(above, priority="high"),
        ]}
        assert determine_priority(quiet)["priority"] == "low"
        assert determine_priority(loud)["priority"] == "medium"

    def test_the_bias_never_pushes_past_high(self):
        state = {"neighbours": [neighbour(0.99, priority="high")]}
        assert determine_priority(state)["priority"] == "high"


# --- fuse: category choice and confidence -------------------------------

class TestFuse:
    def test_agreement_produces_high_confidence(self):
        out = fuse({
            "llm_category": "service",
            "neighbours": [neighbour(0.88, "service"), neighbour(0.86, "service")],
        })
        assert out["category"] == "service"
        assert out["agreed"] is True
        assert out["confidence_parts"]["cross_check"] == 1.0
        assert out["confidence"] > 0.7

    def test_disagreement_collapses_the_cross_check(self):
        """The demo case: classifier says billing, neighbours say service."""
        out = fuse({
            "llm_category": "billing",
            "neighbours": [neighbour(0.85, "service"), neighbour(0.84, "warranty")],
        })
        assert out["category"] == "billing"      # classifier is preferred
        assert out["agreed"] is False
        assert out["confidence_parts"]["cross_check"] == 0.0
        assert out["confidence"] < 0.5           # so it escalates

    def test_falls_back_to_the_neighbours_when_the_model_fails(self):
        """An unparseable model reply must not take the pipeline down."""
        out = fuse({
            "llm_category": "",
            "neighbours": [neighbour(0.9, "ordering"), neighbour(0.7, "ordering")],
        })
        assert out["category"] == "ordering"

    def test_no_evidence_at_all_is_survivable(self):
        out = fuse({"llm_category": "", "neighbours": []})
        assert out["category"] == ""
        assert out["confidence"] == 0.0

    def test_confidence_never_leaves_the_unit_interval(self):
        for llm in ("service", "billing", ""):
            for sims in ([], [neighbour(1.0)], [neighbour(0.0), neighbour(1.0)]):
                out = fuse({"llm_category": llm, "neighbours": sims})
                assert 0.0 <= out["confidence"] <= 1.0

    def test_weights_sum_to_one(self):
        total = config.W_CROSSCHECK + config.W_AGREEMENT + config.W_SIMILARITY
        assert total == pytest.approx(1.0)


# --- classifier output parsing ------------------------------------------

class TestParseCategory:
    VALID = {"service", "billing", "other"}

    def test_clean_json(self):
        assert _parse_category('{"category": "billing"}', self.VALID) == "billing"

    def test_case_and_whitespace_are_tolerated(self):
        assert _parse_category('{"category": "  BILLING "}', self.VALID) == "billing"

    def test_prose_around_the_answer_still_parses(self):
        assert _parse_category("I think this is billing.", self.VALID) == "billing"

    @pytest.mark.parametrize("raw", ["", "{", "null", '{"category": "nonsense"}'])
    def test_unusable_replies_return_empty_rather_than_raising(self, raw):
        assert _parse_category(raw, self.VALID) == ""


# --- resolution notes ----------------------------------------------------

class TestTidyNote:
    def test_a_short_note_is_left_alone(self):
        text = "Schedule a brake inspection."
        assert _tidy_note(text) == text

    def test_at_most_two_sentences_survive(self):
        note = _tidy_note("One thing. Two things. Three things. Four things.")
        assert note.count(".") <= 2

    def test_a_rambling_note_is_cut_to_length(self):
        note = _tidy_note(" ".join(["word"] * 120))
        assert len(note.split()) <= 45

    def test_whitespace_is_normalised(self):
        assert "  " not in _tidy_note("Do   this.\n\nThen  that.")

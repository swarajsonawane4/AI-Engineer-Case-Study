# Smart Inquiry Triage Assistant

A prototype that takes a customer inquiry, classifies it, assigns a priority,
routes it to the right queue, and drafts a short resolution note, using similar
past cases as context.

Built for the BMW Group AI Engineer case study.

---

## Quick start

Requires Python 3.10 or newer and [Ollama](https://ollama.com).

```bash
# 1. Install Ollama and pull the two models
brew install ollama          # or see ollama.com for other platforms
ollama serve &               # leave this running
ollama pull llama3.2:3b      # chat model, ~2 GB
ollama pull nomic-embed-text # embedding model, ~274 MB

# 2. Set up the project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Build the vector store (about 10 seconds for 300 cases)
python -m src.main --ingest

# 4. Run the app
streamlit run app/app.py
```

Then open http://localhost:8501.

You can also triage a single inquiry from the command line:

```bash
python -m src.main "My brakes are grinding and the pedal feels soft."
```

And reproduce the evaluation numbers below:

```bash
python -m src.evaluate --sample 120 --top-k 5   # full pipeline, ~30 s
python -m src.sweep                             # retrieval only, ~5 s
```

---

## Models

| Role | Model | Why |
|---|---|---|
| Chat | `llama3.2:3b` | The brief recommends 1B to 4B. At 3B it follows a JSON output format reliably and answers in about 1.3 seconds on an M-series Mac. |
| Embeddings | `nomic-embed-text` | 768-dimensional, purpose-built for retrieval, and small enough to embed all 300 cases in around 10 seconds. |
| Vector store | ChromaDB | Persistent local storage, cosine distance, no server to run. |

Both models run locally through Ollama. That was a deliberate choice over a
hosted API for three reasons. Customer inquiries are personal data, and keeping
them on the machine avoids sending them to a third party, which matters for a
European company under GDPR. The live demo does not depend on a network
connection or an API key that could fail in the room. And the cost of running
the pipeline is zero, so the evaluation could be re-run as often as needed
while tuning.

The trade-off is answer quality. A 3B model writes blunter resolution notes
than a frontier model would. The brief states that reply quality is secondary,
so that is an acceptable price.

Both model names can be overridden with the `TRIAGE_CHAT_MODEL` and
`TRIAGE_EMBED_MODEL` environment variables.

---

## Architecture

```
          ┌──> classify ──┐
  START ──┤               ├──> fuse ──> priority ──> route ──> notes ──> END
          └──> retrieve ──┘                                        │
                                                              escalation
```

| Node | What it does |
|---|---|
| `classify` | Asks the chat model to pick one category, given the taxonomy definitions and keywords. Returns JSON. |
| `retrieve` | Cosine similarity search over the 300 embedded past cases. Returns the Top-K with similarity scores. |
| `fuse` | Picks the final category and computes the confidence score from three signals. |
| `priority` | Similarity-weighted vote over the neighbours' priorities, with a safety bias. |
| `route` | Deterministic category to queue lookup. |
| `notes` | Chat model drafts a one or two line note, given the category, priority and neighbours. |
| `escalation` | Flags the inquiry for human review if confidence is below the threshold. |

### Why a graph and not a chain

Classification and retrieval do not depend on each other. Both only need the
raw inquiry text. Running them as parallel branches that join at `fuse` gives
two benefits.

The first is speed. The two slowest steps overlap instead of queueing.

The second matters more. Because the classifier never sees the retrieved
neighbours, the two predictions are genuinely independent. When they agree,
that agreement is evidence. If the classifier had been shown the neighbours
first, it would tend to copy them, and the agreement would be circular and
worth nothing. This independence is what makes the confidence score meaningful,
and it is only expressible as a graph.

---

## Key design decisions

### Routing is a lookup, not a prediction

In `past_cases.csv`, every one of the eight categories maps to exactly one
queue, across all 300 rows. There is no ambiguity for a model to resolve.

So the routing table is built from the data at load time, and
`build_routing_table` raises if a category is ever seen mapping to two
different queues. A language model at this step could only introduce errors
that a dictionary cannot make, while adding latency and cost.

### Priority comes from the neighbours, not the category

Category alone cannot decide urgency. Every category in the data spans multiple
priority levels:

| Category | low | medium | high |
|---|---|---|---|
| service | 20 | 18 | 17 |
| technical | 16 | 17 | 12 |
| warranty | 13 | 12 | 10 |
| ordering | 16 | 17 | 7 |
| billing | 14 | 18 | 8 |
| configurator | 22 | 13 | 0 |
| general | 26 | 4 | 0 |
| other | 19 | 1 | 0 |

A billing question can be routine or a duplicate charge worth thousands. So
priority is a similarity-weighted vote over the retrieved neighbours, which
carries the urgency signal that the category label does not.

### The safety bias is deliberately asymmetric

On top of the vote, if a high-priority neighbour is retrieved with similarity
at or above 0.858, the predicted priority is raised by one level.

The reason is that the two errors do not cost the same. Over-triaging a routine
question wastes a few minutes of an agent's time. Under-triaging a brake fault
leaves a safety issue sitting in a low-priority queue. The bias trades some
overall accuracy for a lower under-triage rate, on purpose.

An earlier version used a threshold of 0.75. Measurement showed that was wrong:
0.75 sits below the *minimum* in-domain similarity of 0.775, so the rule fired
on 61% of real cases against a true high-priority rate of 18%. Raising the
threshold to the measured 75th percentile fixed it; the rule now fires on 9%.

### Similarity is rescaled before it is used as confidence

Raw cosine similarity from this embedding model is compressed. Measured by
leave-one-out retrieval over all 300 cases, genuine in-domain matches score
between 0.77 and 0.93, while completely unrelated text ("recipe for banana
bread") still scores around 0.70 to 0.76.

Used raw, the similarity term would sit near 0.8 for everything and contribute
almost nothing. It is therefore rescaled onto [0, 1] using bounds read off that
measured distribution, which is what makes it able to separate an off-topic
message from a real one.

The raw scores are still what the UI displays, because those are the honest
numbers a reviewer would want to see.

---

## How confidence is computed

The brief leaves this open and asks for the reasoning. The score is a weighted
sum of three signals:

| Signal | Weight | What it captures |
|---|---|---|
| Cross-check | 0.40 | Do the classifier and the neighbour vote independently agree on the category? |
| Agreement | 0.35 | What share of the retrieved similarity weight supports the chosen category? |
| Similarity | 0.25 | Are the neighbours actually close to this inquiry, after rescaling? |

Each one covers a failure the others miss.

Similarity alone is high for a novel inquiry that merely looks like old ones.
Agreement alone is high when retrieval is confidently wrong, because five
neighbours drawn from the same wrong region all agree with each other. The
cross-check catches both of those, but it is binary, so on its own it gives
only two possible scores and cannot be thresholded usefully.

**What was deliberately not used: asking the model for its own confidence.**
Self-reported confidence from a small instruction-tuned model is poorly
calibrated. It tends to answer 0.9 regardless, because the number is generated
text rather than a measured quantity. Every signal used here is computed from
observable evidence instead.

The trade-off is that the weights are reasoned rather than learned. With 300
labelled cases, they could be fitted, and the calibration table below is the
measurement that would drive that.

---

## Results

Measured by leave-one-out evaluation. Each case is triaged with itself excluded
from retrieval, so it cannot look up its own answer, and its human-assigned
labels are the ground truth. Reproduce with `python -m src.evaluate`.

### Headline numbers (120 cases, Top-K = 5)

| Metric | Result |
|---|---|
| Category accuracy | **85.8%** |
| Queue accuracy | 85.8% (follows from category by construction) |
| Priority, exact match | 63.3% |
| Priority, within one level | **97.5%** |
| Under-triage rate | 17.5% |
| Latency per inquiry | about 1.3 s |

### Does the confidence score actually work?

This is the question that matters, because the score decides what gets
escalated. If it does not track correctness, it is decoration. It does:

| Confidence band | Cases | Category accuracy |
|---|---|---|
| 0.0 - 0.2 | 18 | 22.2% |
| 0.2 - 0.4 | 5 | 80.0% |
| 0.6 - 0.8 | 48 | 95.8% |
| 0.8 - 1.0 | 49 | 100.0% |

Accuracy rises monotonically with confidence, from 22% in the lowest band to
100% in the highest. The pipeline knows when it does not know.

### The operating trade-off

That is what makes the escalation threshold useful in practice:

| Threshold | Handled automatically | Accuracy on those |
|---|---|---|
| 0.0 (escalate nothing) | 100% | 85.8% |
| **0.5 (default)** | **80.8%** | **97.9%** |
| 0.8 | 40.8% | 100.0% |

At the default threshold, sending the least confident 19% of inquiries to a
human lifts accuracy on everything else from 86% to 98%. That is the business
case in one line: most of the volume is handled automatically and reliably,
and the hard fifth still reaches a person.

One honest caveat. The table is flat between 0.3 and 0.6 because the
cross-check signal is binary and carries the largest weight, so scores cluster
into two groups rather than spreading evenly. Any threshold in that range
behaves identically. Fitting the weights would smooth this out.

### Accuracy by category

| Category | Cases | Accuracy |
|---|---|---|
| technical | 18 | 100.0% |
| general | 12 | 91.7% |
| ordering | 16 | 87.5% |
| service | 22 | 86.4% |
| configurator | 14 | 85.7% |
| warranty | 14 | 85.7% |
| billing | 16 | 75.0% |
| other | 8 | 62.5% |

The two weakest categories are the two that overlap most with their
neighbours. `billing` and `ordering` both involve money changing hands, and
the boundary between a deposit question and a payment question is genuinely
thin. `other` is a catch-all with only 20 examples in the whole dataset, so
retrieval has little to work with.

### Choosing Top-K

Top-K is configurable in the sidebar. The default of 5 was chosen by sweeping
it against ground truth over all 300 cases. Reproduce with `python -m src.sweep`,
which also prints the similarity distribution the constants in `config.py` were
read off.

Note that this sweep isolates the **retrieval component**: the category column
is the accuracy of the neighbour vote on its own, with no LLM involved, which
is why it reads lower than the 85.8% headline for the full pipeline. Isolating
it is the point, since Top-K only affects retrieval.

| Top-K | Category acc. (kNN only) | Priority accuracy | Under-triage rate |
|---|---|---|---|
| 1 | 77.7% | 52.7% | 17.3% |
| 3 | 80.7% | 54.3% | 16.7% |
| **5** | **83.7%** | **61.0%** | **15.0%** |
| 7 | 83.3% | 61.3% | 13.7% |
| 10 | 85.0% | 61.3% | 14.0% |

Accuracy climbs steeply to 5 and then flattens, while latency keeps growing
with every extra neighbour embedded into the vote. Five is the knee.

---

## What works and what does not

**Works well.** Category classification is reliable, and routing follows from
it exactly. Clear safety-critical inquiries are caught and marked high
priority. Off-topic messages get low confidence and are escalated rather than
forced into a category. The whole pipeline answers in roughly 1.3 seconds on a
laptop with no network call.

**Limitations.**

- Priority is the weakest step. It is genuinely ambiguous, and even human
  labellers would disagree on many of these cases. 97.5% of predictions land
  within one level of the truth, but exact agreement is only 63.3%.
- The knowledge base is 300 cases. Rare categories such as `other` have only
  20 examples, so retrieval for them is thin.
- The confidence weights are reasoned, not fitted.
- Resolution notes are not evaluated, because there is no ground truth to
  score them against. They are read for plausibility only.
- Everything is single-language English and single-tenant. There is no
  authentication, rate limiting, or persistence beyond the local Chroma store.

---

## Roadmap

1. **Close the feedback loop.** Every inquiry an agent corrects becomes a new
   labelled case. The knowledge base improves without retraining anything,
   which is the main advantage of a retrieval-based design over a fine-tuned
   classifier.
2. **Fit the confidence weights** on held-out data instead of reasoning about
   them, and pick the escalation threshold from the operating curve rather than
   by default.
3. **Multilingual support**, which matters immediately for a European customer
   base.
4. **Per-queue routing rules**, so that capacity and working hours affect
   routing rather than category alone.
5. **Monitoring**, tracking escalation rate and category drift over time. A
   sudden rise in one category is itself a useful business signal about a
   product problem.

---

## Presentation

The slide deck is in this repository at `presentation/index.html`. Open it in a
browser and use the arrow keys to move between slides. Press **N** to toggle
speaker notes.

---

## Project layout

```
presentation/index.html  Slide deck (Part B)
app/app.py             Streamlit chat UI, wired to the pipeline
src/config.py          All tunables and measured constants
src/knowledge_base.py  Loading, embedding, Chroma store, retrieval
src/nodes.py           The individual pipeline steps
src/graph.py           LangGraph wiring
src/main.py            triage_inquiry() entry point and CLI
src/evaluate.py        Leave-one-out evaluation harness
src/sweep.py           Top-K and similarity sweeps behind the config constants
data/                  Provided past cases and taxonomy
```

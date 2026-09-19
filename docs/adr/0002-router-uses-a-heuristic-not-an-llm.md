# ADR 0002 — The router uses a heuristic, not an LLM call

**Status:** Accepted · **Date:** 2026-09-05

## Context

The router node picks which Groq model synthesizes the answer: the 20b model
for simple questions, the 120b for multi-hop or table questions. It did this
by spending **one LLM call per query** to classify the question as
`simple` / `multi_hop` / `needs_table`, with a keyword heuristic
(`_heuristic`) as a fallback if that call failed or returned something
unrecognized.

That is an LLM call on the critical path of every single request, before any
retrieval happens. Whether it earns its cost is an empirical question, so it
was measured rather than argued about. Harness: `evals/router_ab.py`.

## Measurement 1 — the classifier had never worked

Running the harness against the 20-question golden set:

```
label_disagreement=20/20   model_changed=16/20   mean_router_latency=385ms
```

Every LLM label came back **unparseable**. Isolating the call showed why:

```
max_tokens=20    out_tokens=20   content=''
max_tokens=60    out_tokens=29   content='simple'
max_tokens=150   out_tokens=29   content='simple'
```

gpt-oss models spend output tokens on hidden reasoning before emitting visible
content. At `max_tokens=20` the whole budget went to reasoning and `content`
came back empty — every time. The whitelist check rejected it and the heuristic
silently took over on 100% of queries, while the request still paid ~385ms and
the call's tokens.

Three things kept this invisible:

- `except Exception: pass` — the intended graceful degradation, which also
  swallowed the signal.
- A whitelist that treated *"the model returned garbage"* and *"the model chose
  this label"* identically.
- A trace recording the chosen **label** but not its **source**, so the
  heuristic's output was reported as though the classifier had produced it.

Every trace and dashboard would have shown a healthy router.

## Measurement 2 — repaired, it was worse

With `max_tokens=120` the classifier worked (0 unparseable, 11/20 labels
differing from the heuristic, 8/20 changing the model). Running the full eval
against the identical corpus (index v3), dataset and judge:

| Metric | Heuristic only | Working LLM router |
|---|---|---|
| Mean faithfulness | **0.955** | 0.925 |
| **False refusals** | **0/17** | **1/17** |
| Mean tokens/query | **1,609** | 1,838 (+14%) |
| Latency p95 | **12,057 ms** | 12,307 ms |
| Retrieval hit-rate | 100% | 100% |
| Unanswerable refusal-rate | 100% | 100% |

The false refusal is the informative result, and it is explainable rather than
random. Golden question 2 scores faithfulness 1.00 with 0 repairs in 1,540 ms
under the heuristic. The classifier labelled it `needs_table`, which routed it
to the 120b model, which then **refused a question the small model answers
correctly** — 0.30 faithfulness, 2 repairs, 4,311 ms.

Escalating a simple question did not improve it; it broke it.

## Decision

Remove the LLM classification call. Keep `_heuristic` as the router's only
classifier. Keep the router node itself — it still selects the model, and
records the decision to the trace.

Keep `evals/router_ab.py` so reintroducing an LLM router is a measurable
decision rather than a matter of taste.

## Alternatives considered

**Fix `max_tokens` and keep the LLM router.** Rejected on the data above: no
measured quality gain, a measured quality *loss*, +14% tokens, and ~430 ms
added to every request.

**Keep it but only escalate on high classifier confidence.** Rejected as
unfounded complexity — the API returns a bare label, so "confidence" would have
to be invented, and there is no evidence a better-gated version of a component
with no demonstrated upside would produce one.

**Drop the router node entirely and always use the small model.** Not chosen
*yet*, because it is a separate question with its own experiment: the heuristic
does route the table/multi-hop questions to the larger model and those score
well. Worth measuring, but it is a change to answer quality rather than to
routing mechanism, so it should not ride along with this decision.

## Consequences

- One LLM call removed from every request's critical path.
- Routing is now deterministic — the same question always picks the same model,
  which also makes evals reproducible. The LLM classifier was observably
  unstable, returning different labels for the same question across runs.
- The heuristic is crude and will mis-route some questions. That is accepted:
  it is measurably no worse than the LLM alternative and costs nothing.
- `source` is recorded in the router's trace step, so if a classifier is ever
  reintroduced, a silent failure cannot hide the way this one did.

## Caveat, stated honestly

n = 20, single run, with real LLM variance between runs. This is not enough to
prove the heuristic is *better* — the faithfulness gap is a small number of
questions, and the false refusal is one sample.

It is, however, enough to conclude that the LLM router has **no demonstrated
benefit** while carrying a definite cost in latency and tokens. The burden of
proof sits on keeping an extra model call, not on removing one.

**The strongest evidence is contextual:** every quality number this system
reports — 0.955 faithfulness, 100% hit-rate, 100% refusal accuracy, 0 false
refusals — was achieved while the LLM router was silently broken and the
heuristic was doing all the work.

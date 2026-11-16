# sentinel-proof: catch RAG hallucinations before your users do

[![PyPI](https://img.shields.io/pypi/v/sentinel-proof.svg)](https://pypi.org/project/sentinel-proof/)
[![Python](https://img.shields.io/pypi/pyversions/sentinel-proof.svg)](https://pypi.org/project/sentinel-proof/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/92kareeem/sentinel-rag/blob/main/packages/sentinel-proof/LICENSE)
[![Dependencies: none](https://img.shields.io/badge/dependencies-none-brightgreen.svg)](https://github.com/92kareeem/sentinel-rag/blob/main/packages/sentinel-proof/pyproject.toml)

**Check every sentence of an LLM answer against the retrieved context, before
it ships.** `sentinel-proof` is a hallucination guardrail for
retrieval-augmented generation (RAG). It checks that every number in the answer
appears in the evidence and that every citation points at a passage that
supports its sentence. Wrong citations are re-pointed, unsupported sentences
are removed, and an answer that is mostly unsupported is refused with a reason
code instead of being shown.

Pure Python. No dependencies, no model calls, no API key, no GPU. It runs in
microseconds, so it can check every production answer rather than a sample.
It works with LangChain, LlamaIndex, the OpenAI or Anthropic SDKs, or your own
retriever.

```bash
pip install sentinel-proof
```

Or **[try it in your browser](https://92kareeem.github.io/sentinel-rag/)** first:
the same package runs in the page through Pyodide, and nothing you paste is
uploaded.

**Use it when:**

- your RAG chatbot or document assistant **makes up numbers**: prices, dates,
  limits, percentages that are not in your documents;
- the model **cites the wrong source**, so a user who clicks `[2]` reads a
  passage that says nothing of the kind;
- you need an answer that is **grounded or refused**, not a faithfulness score
  you then have to put a threshold on;
- you want a **fast, deterministic check in CI** over a golden set of answers,
  with no judge model to pay for or keep stable;
- you work in a regulated domain (legal, finance, healthcare, HR policy) and
  have to show **which passage each sentence came from**.

```python
from sentinel_proof import verify, passages_from_texts

passages = passages_from_texts([
    "Customers may request a refund within 30 days of purchase.",
    "Refund requests above 500 USD require manager approval.",
])

answer = (
    "Customers can get a refund within 30 days [1]. "
    "Requests above 500 USD need manager approval [1]. "
    "Refunds are processed in 2 business days [2]."
)

v = verify(answer, passages, citation_style="numeric")

v.ok                  # True: 2 of 3 claims survive, under the 50% strip limit
v.answer              # "Customers can get a refund within 30 days [1]. Requests above 500 USD need manager approval [2]."
v.repaired_citations  # 1: the second claim cited [1] but only [2] supports it
v.stripped            # the "2 business days" claim: no passage contains a 2
v.safe_text()         # the verified answer, or refusal prose if v.ok is False
```

Three things happened there, and all of them are things an LLM judge would
have scored rather than fixed:

1. A fabricated figure was **stripped**, and the rest of the answer still shipped.
2. A citation that pointed at the wrong passage was **repaired** to the right
   one, so a reader who clicks it lands on the text that supports the sentence.
3. The result is a **verdict with per-claim detail**, not a float.

## Why a deterministic gate

LLM-judge metrics (faithfulness, groundedness) are the right tool for *measuring*
a pipeline offline. They are the wrong *last line* in production: a judge can be
wrong in exactly the way the model was, it costs a model call per answer, and it
returns a score, which leaves you to pick a threshold and still ship the whole
answer or none of it.

This gate is string logic. It cannot hallucinate, it runs in microseconds, and it
edits the answer down to what the evidence supports. It is deliberately **weak**:
it checks lexical overlap and numbers, not semantic entailment, because a
deterministic checker that tried to be clever about paraphrase produces false
refusals on good answers. Use it *under* your judge, not instead of it. The two
catch different things.

The design and its measured trade-offs come from running it in production inside
[Sentinel](https://github.com/92kareeem/sentinel-rag), where it is the
last node before the user. The relevant decision record is
[ADR 0003: grounding is deterministic, and deliberately weak](https://github.com/92kareeem/sentinel-rag/blob/main/docs/adr/0003-grounding-is-deterministic-not-only-an-llm-judge.md).

## What it checks, per sentence

| Check | Outcome |
|---|---|
| Content words overlap the cited passages (≥ 25% by default) and every number in the sentence appears in one of them | **supported**: kept as written |
| The cited passages fail, but another retrieved passage passes | **repaired**: kept, citation re-pointed |
| No retrieved passage passes | **stripped**: removed from the answer |

Then: if more than `max_stripped_ratio` (default 50%) of the claims were
stripped, `ok` is `False`, `answer` is empty and `refusal` is
`UNVERIFIABLE_ANSWER`. An answer with no parseable citations at all fails
closed. The model's own refusal token (`INSUFFICIENT_CONTEXT`) passes through
as `is_refusal=True` with reason `INSUFFICIENT_EVIDENCE`.

Details that came from real failures, each with a test:

- Numbers echoed from the **question** are not fabrications ("is 20 days inside
  the window?" may restate 20 when the evidence says 30).
- A sentence citing **several passages at once** (`[1][2][3]` after a summary)
  is supported by their union, but each sentence keeps only the passages it
  actually rests on.
- **List markers** (`1.`), **lead-ins** ("The main ones are:") and passage ids
  the model wrote into the prose are not claims and are not counted.
- `【1】`, `[ chunk: id ]` and full-width brackets are read as citations;
  `[sic]` is not.
- `1,200` and `1200` are the same number.

## Citation styles

| `citation_style` | What the model writes | Mapped to |
|---|---|---|
| `"numeric"` | `[1]`, `[2, 3]`, `[1][2]` | 1-based position in `passages` |
| `"tag"` (default) | `[chunk:<id>]` | `Passage.id` |

Numeric is what most RAG prompts already use, so for most stacks nothing in the
prompt changes. The verified answer is written back in the same style.

## Bring your own retriever

```python
from sentinel_proof import from_langchain, from_llamaindex, Passage

passages = from_langchain(retriever.invoke(question))        # LangChain Documents
passages = from_llamaindex(retriever.retrieve(question))     # LlamaIndex nodes / NodeWithScore
passages = [Passage(text=row.text, id=row.id, source=row.url) for row in my_rows]
```

The adapters are duck-typed: this package imports neither library. A `Passage`
needs only `text`; `section` (a heading path such as `Item 7.2 > Refunds`) is
counted as evidence, and `source`/`page` ride along into the verdict for display.

## Provenance a reader can check

Give a passage its span and the source file's checksum, and it gets a hash
anyone can recompute in any language:

```python
p = Passage(text=quote, document_sha256=sha, char_start=120, char_end=410).with_provenance()
p.provenance_hash   # sha256(f"{sha}|120|410|{quote}")
```

Publish the hash with the citation. A client that holds the file can prove the
quote is that exact span of that exact file, without trusting the server.
Sentinel's web UI does this in the browser with WebCrypto on every citation.

## CLI

```bash
sentinel-proof check --answer answer.txt --passages passages.json --numeric
echo "Refunds take 30 days [1]." | sentinel-proof check --passages passages.json --numeric --json
```

Exit code 0 when verified, 1 when not. `--json` gives the full verdict, claims
included. `passages.json` is a list of strings or of objects with a `text`
field. Useful as a CI step over a golden set of answers.

## API

```python
verify(answer, passages, question="", *, citation_style="tag",
       max_stripped_ratio=0.5, min_lexical_overlap=0.25) -> Verdict
```

`Verdict`: `ok`, `answer`, `claims` (each with `text`, `status`, `cited`,
`supported_by`, `unsupported_numbers`), `stripped_ratio`, `cited_ids`,
`repaired_citations`, `refusal`, `is_refusal`, `passed`, `stripped`, `safe_text()`.

Also exported: `Passage`, `passages_from_texts`, `from_langchain`,
`from_llamaindex`, `provenance_hash`, `check_provenance`, `document_sha256`,
`normalize_citations`, `strip_citations`, `RefusalReason`, `refusal_text`,
`is_refusal`.

## How it fits with other tools

Most hallucination tooling *measures*: it returns a score from a judge model or
a trained classifier. `sentinel-proof` *enforces*: it edits the answer and
decides whether it ships. They are complementary, and the strongest setup uses
both.

| | Typical use | Needs a model | Output | Changes the answer |
|---|---|---|---|---|
| LLM-as-judge metrics (Ragas faithfulness, DeepEval) | Offline evaluation of a pipeline | Yes, a call per answer | A score | No |
| Hallucination classifiers (HHEM, Lynx, LettuceDetect) | Detection, monitoring | Yes, local or hosted | A score or labelled spans | No |
| Guardrail frameworks (NeMo Guardrails, Guardrails AI) | Policy rails around the model | Usually, for fact checks | Pass, fail or reask | Sometimes, by reasking |
| **sentinel-proof** | The last step before an answer is shown | **No** | Per-sentence verdict | **Yes**: re-points citations, strips claims, refuses |

Use a judge or classifier to catch paraphrase that misstates its source, which
this gate cannot see (see below). Use this gate so that a fabricated number or a
wrong citation never reaches a user, whatever the judge said.

## FAQ

### How do I detect hallucinations in a RAG answer without an LLM judge?

Ask the model to cite passages (`[1]`, `[2]`), then pass the answer and the
retrieved passages to `verify()`. Each sentence is checked against the
passages it cites: its numbers must appear there and its content words must
overlap. The verdict lists every claim as supported, repaired or stripped.

### How do I check that an LLM's citations are correct?

The same call. A citation is valid only if the passage it points at supports
the sentence. If another retrieved passage supports it instead, the citation is
rewritten to point there (`v.repaired_citations` counts them). A citation to a
passage that was never retrieved (`[9]` when there are three) is never
trusted: the sentence is re-pointed to a retrieved passage that supports it,
or removed.

### How do I stop my LLM from making up numbers?

Prompts reduce it; they do not prevent it. `verify()` removes any sentence
containing a figure (`30 days`, `$1,200`, `4.5%`) that is not in the retrieved
text, and refuses the answer if too much has to go. Numbers the user typed in
the question are allowed, because restating them is not fabrication.

### Does it work with LangChain, LlamaIndex or the OpenAI SDK?

Yes. `from_langchain(docs)` and `from_llamaindex(nodes)` turn retriever output
into passages, without importing either library. With a raw SDK, pass the
passage strings to `passages_from_texts` and the completion text to `verify`.
Nothing changes in your prompt if it already asks for `[1]`-style citations.

### Does it need a GPU, an API key or network access?

No. It is string processing in the standard library, with no dependencies, so
it runs anywhere Python 3.10+ runs, including AWS Lambda and air-gapped hosts.

### Is it a replacement for Ragas, DeepEval or an NLI model?

No. Those judge meaning; this gate checks evidence. Run them for evaluation and
for paraphrase, and run this on every answer as the deterministic floor.

### What happens when the answer can't be verified?

`v.ok` is `False`, `v.answer` is empty, and `v.refusal` says why
(`UNVERIFIABLE_ANSWER`, `INSUFFICIENT_EVIDENCE`). `v.safe_text()` returns
refusal prose that never leaks the unverified draft.

## What it is not

- Not an entailment model. "The policy allows returns for a month" against a
  passage saying "30 days" is **stripped**, because no overlap carries the
  claim. Pair the gate with an NLI model or an LLM judge for paraphrase.
- Not a retriever quality metric. If the right passage was never retrieved, the
  gate refuses correctly and tells you nothing about why. That is what the
  casebook and gap report in Sentinel are for.
- Not multilingual yet. Content words are `[A-Za-z]{2,}`; numbers are
  script-independent. Contributions welcome, with a test.

## License

Apache-2.0. The Sentinel platform this was extracted from is AGPL-3.0; this
package is licensed separately so it can be used anywhere.

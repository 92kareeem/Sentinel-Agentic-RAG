# sentinel-proof

**A deterministic grounding gate for RAG answers.** Every number in a claim
must appear in the passage it cites. Every citation must point at a passage
that supports the sentence in front of it. What cannot be verified is stripped,
and if too much is stripped the answer is refused with a reason code instead
of shipped.

Pure Python. No dependencies. No model calls. Works with any retrieval stack.

```bash
pip install sentinel-proof
```

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

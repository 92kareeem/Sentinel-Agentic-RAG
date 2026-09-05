# Router A/B — is the classification LLM call worth it?

_2026-09-05 09:47 · 20 questions · classifier: openai/gpt-oss-20b_

The router spends one LLM call per query to pick between the 20b and
120b synthesis model. router.py already contains a keyword heuristic
used whenever that call fails. This compares the two.

| Metric | Value |
|---|---|
| Questions | 20 |
| Label disagreements (LLM vs heuristic) | 11/20 (55%) |
| **Disagreements that change the MODEL** | **8/20 (40%)** |
| Unparseable LLM labels | 0/20 |
| Mean router latency | 428 ms |
| Total router latency across the set | 8559 ms |
| Mean router tokens | 139 |

Only the model-changing row matters for outcomes: `multi_hop` and
`needs_table` both select the complex model, so disagreement between
those two costs nothing. Every query pays the latency row regardless.

## Per-question

| # | Heuristic | LLM | Model changed | ms | Question |
|---|---|---|---|---|---|
| 1 | multi_hop | needs_table | no | 482 | How many days after purchase can a customer request a refund |
| 2 | simple | needs_table | **yes** | 458 | Under what condition are digital goods refundable? |
| 3 | simple | multi_hop | **yes** | 454 | What happens to subscriptions cancelled mid-cycle? |
| 4 | simple | simple | no | 433 | What restocking fee applies in India? |
| 5 | multi_hop | simple | **yes** | 441 | How many processing days do EU refunds take and by what meth |
| 6 | simple | simple | no | 465 | When are restocking fees waived entirely? |
| 7 | simple | simple | no | 445 | What should a customer do if a refund is not received in the |
| 8 | simple | needs_table | **yes** | 307 | Who handles escalations unresolved after 10 business days, a |
| 9 | multi_hop | needs_table | no | 332 | Within how many business days must new hires complete securi |
| 10 | simple | simple | no | 429 | What is the buddy program? |
| 11 | simple | simple | no | 483 | What is required for production database access? |
| 12 | simple | simple | no | 437 | How do system access requests work? |
| 13 | simple | simple | no | 411 | What laptop budget does a Designer get and with what monitor |
| 14 | simple | needs_table | **yes** | 429 | Which role has a 4-year refresh cycle? |
| 15 | simple | simple | no | 396 | Does unused equipment allowance roll over? |
| 16 | multi_hop | needs_table | no | 431 | How many remote days per week are allowed after probation? |
| 17 | simple | needs_table | **yes** | 451 | A defective item was bought in the US — what fee applies and |
| 18 | simple | simple | no | 459 | What is the company's parental leave policy? |
| 19 | simple | needs_table | **yes** | 417 | Which cloud provider hosts the company's infrastructure? |
| 20 | simple | needs_table | **yes** | 399 | What was the company's revenue last quarter? |
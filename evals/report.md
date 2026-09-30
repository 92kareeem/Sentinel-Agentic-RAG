# Sentinel — Eval Report

_2026-09-30 17:04 · 20 questions · judge: openai/gpt-oss-20b_

| Metric | Value | Gate |
|---|---|---|
| Mean faithfulness (vs retrieved context) | **1.000** | >= 0.75 |
| Mean completeness (vs reference) | 0.975 | not gated yet |
| Retrieval hit-rate (top-5) | 100% | — |
| Unanswerable refusal-rate | 100% (3/3) | >= 2/3 |
| False refusals (answerable) | 0/17 | — |
| Latency p50 / p95 | 1296 ms / 12211 ms | — |
| Mean tokens/query | 1308 | — |

## Repair loop
3 of 20 questions triggered repair (18, 19, 20). Faithfulness on repaired items: 1.00

## Per-question results

| # | Category | Faith | Compl | Hit | Refused | Repairs | ms |
|---|---|---|---|---|---|---|---|
| 1 | factual | 1.00 | 1.00 | True | False | 0 | 12211 |
| 2 | factual | 1.00 | 1.00 | True | False | 0 | 1054 |
| 3 | factual | 1.00 | 1.00 | True | False | 0 | 1002 |
| 4 | table | 1.00 | 1.00 | True | False | 0 | 1615 |
| 5 | table | 1.00 | 1.00 | True | False | 0 | 1296 |
| 6 | factual | 1.00 | 1.00 | True | False | 0 | 1211 |
| 7 | factual | 1.00 | 1.00 | True | False | 0 | 1007 |
| 8 | factual | 1.00 | 1.00 | True | False | 0 | 1107 |
| 9 | factual | 1.00 | 1.00 | True | False | 0 | 1392 |
| 10 | factual | 1.00 | 1.00 | True | False | 0 | 1671 |
| 11 | factual | 1.00 | 0.50 | True | False | 0 | 976 |
| 12 | factual | 1.00 | 1.00 | True | False | 0 | 1358 |
| 13 | table | 1.00 | 1.00 | True | False | 0 | 1121 |
| 14 | table | 1.00 | 1.00 | True | False | 0 | 1130 |
| 15 | factual | 1.00 | 1.00 | True | False | 0 | 1159 |
| 16 | factual | 1.00 | 1.00 | True | False | 0 | 1044 |
| 17 | multi-hop | 1.00 | 1.00 | True | False | 0 | 1395 |
| 18 | unanswerable | 1.00 | 1.00 | None | True | 1 | 1543 |
| 19 | unanswerable | 1.00 | 1.00 | None | True | 1 | 1638 |
| 20 | unanswerable | 1.00 | 1.00 | None | True | 1 | 1669 |

## Regression suite

0 enforced · 0 pending · 0 retired. Every enforced item must pass on its own; pending items are known failures, tracked but not blocking.

_No active regressions yet._
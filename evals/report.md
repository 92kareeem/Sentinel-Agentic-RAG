# Sentinel — Eval Report

_2026-09-05 09:43 · 20 questions · judge: openai/gpt-oss-20b_

| Metric | Value | Gate |
|---|---|---|
| Mean faithfulness | **0.955** | >= 0.75 |
| Retrieval hit-rate (top-5) | 100% | — |
| Unanswerable refusal-rate | 100% (3/3) | >= 2/3 |
| False refusals (answerable) | 0/17 | — |
| Latency p50 / p95 | 10228 ms / 12057 ms | — |
| Mean tokens/query | 1609 | — |

## Repair loop
5 of 20 questions triggered repair (4, 9, 18, 19, 20). Faithfulness on repaired items: 1.00

## Per-question results

| # | Category | Faith | Hit | Refused | Repairs | ms |
|---|---|---|---|---|---|---|
| 1 | factual | 1.00 | True | False | 0 | 12010 |
| 2 | factual | 1.00 | True | False | 0 | 1540 |
| 3 | factual | 1.00 | True | False | 0 | 1285 |
| 4 | table | 1.00 | True | False | 2 | 3570 |
| 5 | table | 1.00 | True | False | 0 | 1687 |
| 6 | factual | 1.00 | True | False | 0 | 7691 |
| 7 | factual | 0.90 | True | False | 0 | 10228 |
| 8 | factual | 1.00 | True | False | 0 | 10806 |
| 9 | factual | 1.00 | True | False | 1 | 10216 |
| 10 | factual | 1.00 | True | False | 0 | 10492 |
| 11 | factual | 0.70 | True | False | 0 | 9910 |
| 12 | factual | 0.50 | True | False | 0 | 11432 |
| 13 | table | 1.00 | True | False | 0 | 11083 |
| 14 | table | 1.00 | True | False | 0 | 11273 |
| 15 | factual | 1.00 | True | False | 0 | 7397 |
| 16 | factual | 1.00 | True | False | 0 | 3389 |
| 17 | multi-hop | 1.00 | True | False | 0 | 9815 |
| 18 | unanswerable | 1.00 | None | True | 1 | 12057 |
| 19 | unanswerable | 1.00 | None | True | 1 | 11233 |
| 20 | unanswerable | 1.00 | None | True | 1 | 10532 |
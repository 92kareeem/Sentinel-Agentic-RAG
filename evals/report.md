# Sentinel — Eval Report

_2026-09-05 10:04 · 20 questions · judge: openai/gpt-oss-20b_

| Metric | Value | Gate |
|---|---|---|
| Mean faithfulness (vs retrieved context) | **1.000** | >= 0.75 |
| Mean completeness (vs reference) | 0.975 | not gated yet |
| Retrieval hit-rate (top-5) | 100% | — |
| Unanswerable refusal-rate | 100% (3/3) | >= 2/3 |
| False refusals (answerable) | 0/17 | — |
| Latency p50 / p95 | 7807 ms / 9906 ms | — |
| Mean tokens/query | 1482 | — |

## Repair loop
5 of 20 questions triggered repair (4, 9, 18, 19, 20). Faithfulness on repaired items: 1.00

## Per-question results

| # | Category | Faith | Compl | Hit | Refused | Repairs | ms |
|---|---|---|---|---|---|---|---|
| 1 | factual | 1.00 | 1.00 | True | False | 0 | 9534 |
| 2 | factual | 1.00 | 1.00 | True | False | 0 | 1130 |
| 3 | factual | 1.00 | 1.00 | True | False | 0 | 1020 |
| 4 | table | 1.00 | 1.00 | True | False | 2 | 3853 |
| 5 | table | 1.00 | 1.00 | True | False | 0 | 1419 |
| 6 | factual | 1.00 | 1.00 | True | False | 0 | 8001 |
| 7 | factual | 1.00 | 1.00 | True | False | 0 | 8956 |
| 8 | factual | 1.00 | 1.00 | True | False | 0 | 9858 |
| 9 | factual | 1.00 | 1.00 | True | False | 1 | 9906 |
| 10 | factual | 1.00 | 1.00 | True | False | 0 | 5225 |
| 11 | factual | 1.00 | 0.50 | True | False | 0 | 8368 |
| 12 | factual | 1.00 | 1.00 | True | False | 0 | 7757 |
| 13 | table | 1.00 | 1.00 | True | False | 0 | 6959 |
| 14 | table | 1.00 | 1.00 | True | False | 0 | 7807 |
| 15 | factual | 1.00 | 1.00 | True | False | 0 | 6604 |
| 16 | factual | 1.00 | 1.00 | True | False | 0 | 4587 |
| 17 | multi-hop | 1.00 | 1.00 | True | False | 0 | 7893 |
| 18 | unanswerable | 1.00 | 1.00 | None | True | 1 | 9196 |
| 19 | unanswerable | 1.00 | 1.00 | None | True | 1 | 7778 |
| 20 | unanswerable | 1.00 | 1.00 | None | True | 1 | 9173 |
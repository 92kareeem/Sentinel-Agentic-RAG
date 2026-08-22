# Sentinel — Eval Report

_2026-08-22 22:19 · 20 questions · judge: openai/gpt-oss-20b_

| Metric | Value | Gate |
|---|---|---|
| Mean faithfulness | **0.975** | >= 0.75 |
| Retrieval hit-rate (top-5) | 100% | — |
| Unanswerable refusal-rate | 100% (3/3) | >= 2/3 |
| False refusals (answerable) | 0/17 | — |
| Latency p50 / p95 | 8990 ms / 25542 ms | — |
| Mean tokens/query | 1631 | — |

## Repair loop
4 of 20 questions triggered repair (11, 12, 18, 20). Faithfulness on repaired items: 0.88

## Per-question results

| # | Category | Faith | Hit | Refused | Repairs | ms |
|---|---|---|---|---|---|---|
| 1 | factual | 1.00 | True | False | 0 | 11800 |
| 2 | factual | 1.00 | True | False | 0 | 1385 |
| 3 | factual | 1.00 | True | False | 0 | 1356 |
| 4 | table | 1.00 | True | False | 0 | 5372 |
| 5 | table | 1.00 | True | False | 0 | 5355 |
| 6 | factual | 1.00 | True | False | 0 | 8981 |
| 7 | factual | 1.00 | True | False | 0 | 9100 |
| 8 | factual | 1.00 | True | False | 0 | 9309 |
| 9 | factual | 1.00 | True | False | 0 | 5041 |
| 10 | factual | 1.00 | True | False | 0 | 8990 |
| 11 | factual | 1.00 | True | False | 1 | 20292 |
| 12 | factual | 0.50 | True | False | 2 | 25542 |
| 13 | table | 1.00 | True | False | 0 | 9832 |
| 14 | table | 1.00 | True | False | 0 | 9135 |
| 15 | factual | 1.00 | True | False | 0 | 8755 |
| 16 | factual | 1.00 | True | False | 0 | 5135 |
| 17 | multi-hop | 1.00 | True | False | 0 | 8912 |
| 18 | unanswerable | 1.00 | None | True | 2 | 21557 |
| 19 | unanswerable | 1.00 | None | True | 0 | 8237 |
| 20 | unanswerable | 1.00 | None | True | 2 | 23555 |
# Sentinel — Eval Report

_2026-08-26 01:05 · 20 questions · judge: openai/gpt-oss-20b_

| Metric | Value | Gate |
|---|---|---|
| Mean faithfulness | **0.950** | >= 0.75 |
| Retrieval hit-rate (top-5) | 100% | — |
| Unanswerable refusal-rate | 100% (3/3) | >= 2/3 |
| False refusals (answerable) | 0/17 | — |
| Latency p50 / p95 | 9548 ms / 25341 ms | — |
| Mean tokens/query | 1950 | — |

## Repair loop
5 of 20 questions triggered repair (4, 9, 18, 19, 20). Faithfulness on repaired items: 1.00

## Per-question results

| # | Category | Faith | Hit | Refused | Repairs | ms |
|---|---|---|---|---|---|---|
| 1 | factual | 1.00 | True | False | 0 | 11191 |
| 2 | factual | 1.00 | True | False | 0 | 1767 |
| 3 | factual | 1.00 | True | False | 0 | 2570 |
| 4 | table | 1.00 | True | False | 2 | 3342 |
| 5 | table | 1.00 | True | False | 0 | 1683 |
| 6 | factual | 1.00 | True | False | 0 | 6919 |
| 7 | factual | 1.00 | True | False | 0 | 9212 |
| 8 | factual | 1.00 | True | False | 0 | 11451 |
| 9 | factual | 1.00 | True | False | 1 | 9548 |
| 10 | factual | 1.00 | True | False | 0 | 9951 |
| 11 | factual | 0.50 | True | False | 0 | 10192 |
| 12 | factual | 0.50 | True | False | 0 | 8685 |
| 13 | table | 1.00 | True | False | 0 | 9150 |
| 14 | table | 1.00 | True | False | 0 | 9138 |
| 15 | factual | 1.00 | True | False | 0 | 10106 |
| 16 | factual | 1.00 | True | False | 0 | 4672 |
| 17 | multi-hop | 1.00 | True | False | 0 | 9979 |
| 18 | unanswerable | 1.00 | None | True | 2 | 24141 |
| 19 | unanswerable | 1.00 | None | True | 2 | 24577 |
| 20 | unanswerable | 1.00 | None | True | 2 | 25341 |
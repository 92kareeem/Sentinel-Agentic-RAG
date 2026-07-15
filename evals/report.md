# Sentinel — Eval Report

_2026-07-15 23:50 · 20 questions · judge: llama-3.1-8b-instant_

| Metric | Value | Gate |
|---|---|---|
| Mean faithfulness | **0.905** | >= 0.75 |
| Retrieval hit-rate (top-8) | 100% | — |
| Unanswerable refusal-rate | 100% (3/3) | >= 2/3 |
| False refusals (answerable) | 1/17 | — |
| Latency p50 / p95 | 8003 ms / 37434 ms | — |
| Mean tokens/query | 1355 | — |

## Repair loop
1 of 20 questions triggered repair (18). Faithfulness on repaired items: 1.00

## Per-question results

| # | Category | Faith | Hit | Refused | Repairs | ms |
|---|---|---|---|---|---|---|
| 1 | factual | 0.30 | True | True | 0 | 37434 |
| 2 | factual | 1.00 | True | False | 0 | 824 |
| 3 | factual | 1.00 | True | False | 0 | 676 |
| 4 | table | 1.00 | True | False | 0 | 734 |
| 5 | table | 0.80 | True | False | 0 | 658 |
| 6 | factual | 1.00 | True | False | 0 | 7196 |
| 7 | factual | 1.00 | True | False | 0 | 2116 |
| 8 | factual | 1.00 | True | False | 0 | 8148 |
| 9 | factual | 1.00 | True | False | 0 | 8003 |
| 10 | factual | 1.00 | True | False | 0 | 8104 |
| 11 | factual | 1.00 | True | False | 0 | 6716 |
| 12 | factual | 0.00 | True | False | 0 | 8198 |
| 13 | table | 1.00 | True | False | 0 | 8237 |
| 14 | table | 1.00 | True | False | 0 | 8102 |
| 15 | factual | 1.00 | True | False | 0 | 7137 |
| 16 | factual | 1.00 | True | False | 0 | 6805 |
| 17 | multi-hop | 1.00 | True | False | 0 | 8440 |
| 18 | unanswerable | 1.00 | None | True | 2 | 21531 |
| 19 | unanswerable | 1.00 | None | True | 0 | 7999 |
| 20 | unanswerable | 1.00 | None | True | 0 | 8247 |
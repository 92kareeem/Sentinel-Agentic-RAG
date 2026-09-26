# Sentinel — Eval Report

_2026-09-20 06:49 · 20 questions · judge: openai/gpt-oss-20b_

| Metric | Value | Gate |
|---|---|---|
| Mean faithfulness (vs retrieved context) | **1.000** | >= 0.75 |
| Mean completeness (vs reference) | 1.000 | not gated yet |
| Retrieval hit-rate (top-5) | 100% | — |
| Unanswerable refusal-rate | 100% (3/3) | >= 2/3 |
| False refusals (answerable) | 0/17 | — |
| Latency p50 / p95 | 1139 ms / 18183 ms | — |
| Mean tokens/query | 1417 | — |

## Repair loop
4 of 20 questions triggered repair (4, 18, 19, 20). Faithfulness on repaired items: 1.00

## Per-question results

| # | Category | Faith | Compl | Hit | Refused | Repairs | ms |
|---|---|---|---|---|---|---|---|
| 1 | factual | 1.00 | 1.00 | True | False | 0 | 18183 |
| 2 | factual | 1.00 | 1.00 | True | False | 0 | 1075 |
| 3 | factual | 1.00 | 1.00 | True | False | 0 | 1082 |
| 4 | table | 1.00 | 1.00 | True | False | 2 | 3692 |
| 5 | table | 1.00 | 1.00 | True | False | 0 | 729 |
| 6 | factual | 1.00 | 1.00 | True | False | 0 | 1087 |
| 7 | factual | 1.00 | 1.00 | True | False | 0 | 1098 |
| 8 | factual | 1.00 | 1.00 | True | False | 0 | 1055 |
| 9 | factual | 1.00 | 1.00 | True | False | 0 | 1264 |
| 10 | factual | 1.00 | 1.00 | True | False | 0 | 1052 |
| 11 | factual | 1.00 | 1.00 | True | False | 0 | 1138 |
| 12 | factual | 1.00 | 1.00 | True | False | 0 | 1308 |
| 13 | table | 1.00 | 1.00 | True | False | 0 | 1175 |
| 14 | table | 1.00 | 1.00 | True | False | 0 | 1071 |
| 15 | factual | 1.00 | 1.00 | True | False | 0 | 1139 |
| 16 | factual | 1.00 | 1.00 | True | False | 0 | 1105 |
| 17 | multi-hop | 1.00 | 1.00 | True | False | 0 | 1385 |
| 18 | unanswerable | 1.00 | 1.00 | None | True | 1 | 1532 |
| 19 | unanswerable | 1.00 | 1.00 | None | True | 1 | 1524 |
| 20 | unanswerable | 1.00 | 1.00 | None | True | 1 | 1261 |
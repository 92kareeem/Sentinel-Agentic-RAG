# Load testing Sentinel

A JMeter test plan, a stand-in model, and two scripts: one drives the load,
one turns the results into the report in [`docs/benchmarks/`](../docs/benchmarks/).

## What is measured, and what is not

Sentinel's own cost per request: authentication, the guardrails, query
embedding, access-filtered hybrid retrieval, the LangGraph agent, the
deterministic grounding gate, the verified-answer cache, and the transport.

The model is **not** measured. A hosted model's free tier allows about 30
requests a minute, so a load test pointed at one measures the provider's rate
limiter. [`mock_llm.py`](mock_llm.py) answers in the OpenAI wire format with a
fixed latency (300 ms by default, a typical small hosted model) and with
answers good enough to pass Sentinel's real critic and grounding gate: it
quotes the top retrieved passage verbatim with its citation tag. Nothing in
Sentinel is mocked; only the network hop to a model is replaced.

Do not point this at AWS. At these rates the Lambda free tier (1M requests a
month) lasts minutes, and new accounts are capped at 10 concurrent executions.

## Scenarios

| Scenario | Request | Passes when |
|---|---|---|
| `health` | `GET /healthz` | 200 |
| `blocked` | prompt injection to `POST /v1/query` | 422, stopped before any model call |
| `cached` | the same question, repeatedly | 200 with citations, from the verified-answer cache |
| `answer` | a new question every time | 200 with citations: the full graph, two model calls |
| `stream` | a new question to `POST /v1/query/stream` | SSE that ends in a `result` event with citations |

## Method

- **Open model.** JMeter's Precise Throughput Timer schedules arrivals at the
  offered rate whatever the server does, so a slow server shows up as latency
  and errors instead of quietly lowering the load (the closed-loop mistake).
- **Staircase.** Each scenario climbs a ladder of rates, 30 seconds per step,
  the first 5 seconds excluded as warm-up. Climbing stops at the first step
  that fails: more than 1% errors, or less than 90% of the offered rate
  delivered. That failing step is reported too: where it breaks is the result.
- **One API process, 2 CPUs.** The API container is capped at 2 CPUs and runs
  one uvicorn worker, so a result is per-instance capacity. Sentinel scales
  out by instance (Lambda runs one request per instance; containers scale
  horizontally). Local mode keeps the cache and quota counters in process
  memory, so it is deliberately not run with several workers.
- **Admin key.** Ordinary users have a 50-question daily quota; the admin is
  exempt, so the quota does not cap the test.

## Run it

Requirements: Docker, Java 17+, [JMeter 5.6+](https://jmeter.apache.org/download_jmeter.cgi)
(no plugins needed).

```bash
make bench                                         # build, start, full run (~20 min)
make bench ARGS="--scenarios answer --steps 10,20" # one scenario, chosen rates
make bench-down                                    # remove the stack and its data
```

Without make:

```bash
BUILDX_BAKE_ENTITLEMENTS_FS=0 docker compose -f benchmarks/compose.bench.yml up -d --build --wait
python benchmarks/run.py --jmeter-home /path/to/apache-jmeter-5.6.3
python benchmarks/report.py --run benchmarks/results/<run-id> --out docs/benchmarks/<date>
```

`run.py` writes raw `.jtl` files and `summary.json` to `benchmarks/results/`
(gitignored). `report.py` builds the Markdown report and its SVG charts, and
takes `--baseline` to compare two runs. For JMeter's own HTML dashboard of
any step: `jmeter -g benchmarks/results/<run-id>/answer-20.jtl -o dashboard/`.

Settings: `BENCH_API_CPUS` (default 2) and `MOCK_LATENCY_MS` (default 300)
are read by both the compose file and the runner.

## Reading the numbers

The load generator shares the machine with the system under test, so treat
the results as comparable between runs on the same machine, not as a
production capacity guarantee. The report records the machine, the commit
and every setting next to the numbers.

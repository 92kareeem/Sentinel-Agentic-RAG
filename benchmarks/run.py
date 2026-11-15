"""Run the Sentinel load test: every scenario, a staircase of arrival rates.

    python benchmarks/run.py                          # full run, ~15 minutes
    python benchmarks/run.py --scenarios health --steps 50,100 --duration 10

Needs: the bench stack up (benchmarks/compose.bench.yml), Java 17+, and
JMeter 5.6+ (JMETER_HOME, --jmeter-home, or C:/jmeter/apache-jmeter-5.6.3).

For each scenario the rate climbs the staircase until a step fails — more
than 1% errors, or the server cannot keep up with 90% of the offered rate.
The failing step is recorded too: where a system breaks is the result.

Writes benchmarks/results/<run-id>/: raw .jtl per step (gitignored) and
summary.json, which benchmarks/report.py turns into the published report.
Standard library only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

SCENARIOS = ("health", "blocked", "cached", "answer", "stream")
# Rough service time per scenario, used only to size the thread pool so the
# plan can sustain the offered rate (threads ~ rate x latency, with headroom).
EXPECTED_LATENCY_S = {"health": 0.02, "blocked": 0.03, "cached": 0.03, "answer": 1.5, "stream": 1.5}
# Default staircases (offered requests/second). Paths that never reach a model
# are measured in the hundreds; paths that run the whole graph, in tens.
FAST_STEPS = "100,250,500,750,1000,1500,2000"
GRAPH_STEPS = "5,10,20,30,40,50,75,100,150,200"
DEFAULT_STEPS = {
    "health": FAST_STEPS,
    "blocked": FAST_STEPS,
    "cached": FAST_STEPS,
    "answer": GRAPH_STEPS,
    "stream": GRAPH_STEPS,
}
MAX_THREADS = 2000
WARMUP_S = 5  # dropped from the statistics: connection setup, JIT, ramp
MAX_ERROR_RATE = 0.01
MIN_DELIVERED = 0.90
# p95 latency a step must stay under. A server that still delivers the rate
# but makes every user wait seconds is not sustaining it. Paths that never
# reach a model must stay interactive; a full answer (two model calls at
# 300 ms each, ~0.7 s unloaded) may take up to 2 s.
SLO_P95_MS = {"health": 250, "blocked": 250, "cached": 250, "answer": 2000, "stream": 2000}


PASS_RULE = (
    f"errors <= {MAX_ERROR_RATE:.0%}, delivered >= {MIN_DELIVERED:.0%} of the offered rate, "
    "p95 under 250 ms (no model call) or 2 s (full answer)"
)


def fail_reason(step: dict[str, Any]) -> str:
    """Every way a step failed the rule, joined; empty if it passed."""
    if step["samples"] == 0:
        return "no samples"
    reasons = []
    if step["error_rate"] > MAX_ERROR_RATE:
        reasons.append(f"{step['error_rate']:.1%} errors")
    if step["achieved_rps"] < MIN_DELIVERED * step["target_rps"]:
        reasons.append(f"delivered {step['achieved_rps'] / step['target_rps']:.1%} of offered")
    if step["p95_ms"] > SLO_P95_MS[step["scenario"]]:
        reasons.append(f"p95 over {SLO_P95_MS[step['scenario']]:,} ms")
    return "; ".join(reasons)


def step_passes(step: dict[str, Any]) -> bool:
    """The one pass rule, shared with report.py so old summaries are re-judged."""
    return bool(
        step["samples"] > 0
        and step["error_rate"] <= MAX_ERROR_RATE
        and step["achieved_rps"] >= MIN_DELIVERED * step["target_rps"]
        and step["p95_ms"] <= SLO_P95_MS[step["scenario"]]
    )


API_CONTAINER = "sentinel-bench-api-1"


@dataclass
class Step:
    scenario: str
    target_rps: int
    threads: int
    duration_s: int
    samples: int = 0
    achieved_rps: float = 0.0
    error_rate: float = 0.0
    p50_ms: float = 0.0
    p90_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    max_ms: float = 0.0
    ttfb_p50_ms: float = 0.0
    ttfb_p95_ms: float = 0.0
    cpu_mean_pct: float = 0.0
    cpu_max_pct: float = 0.0
    mem_max_mib: float = 0.0
    codes: dict[str, int] = field(default_factory=dict)
    top_failure: str = ""
    passed: bool = False


# ------------------------------------------------------------------ helpers


def _http(
    url: str, *, key: str, body: dict[str, Any] | None = None, timeout: float = 60
) -> tuple[int, str]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if body is not None else "GET")
    req.add_header("x-api-key", key)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


def percentile(sorted_values: list[float], q: float) -> float:
    """Nearest-rank percentile; 0 for an empty list."""
    if not sorted_values:
        return 0.0
    rank = max(1, math.ceil(q / 100 * len(sorted_values)))
    return sorted_values[rank - 1]


def _mib(text: str) -> float:
    """'512.3MiB' / '1.2GiB' / '900KiB' -> MiB."""
    text = text.strip()
    for unit, factor in (("GiB", 1024.0), ("MiB", 1.0), ("KiB", 1 / 1024), ("B", 1 / 1024 / 1024)):
        if text.endswith(unit):
            return float(text[: -len(unit)]) * factor
    return 0.0


class StatsSampler(threading.Thread):
    """Samples the API container's CPU and memory while a step runs."""

    def __init__(self, container: str) -> None:
        super().__init__(daemon=True)
        self.container = container
        self.cpu: list[float] = []
        self.mem: list[float] = []
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.is_set():
            out = subprocess.run(
                [
                    "docker",
                    "stats",
                    "--no-stream",
                    "--format",
                    "{{.CPUPerc}}|{{.MemUsage}}",
                    self.container,
                ],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
            if "|" in out:
                cpu, mem = out.split("|", 1)
                try:
                    self.cpu.append(float(cpu.rstrip("%")))
                    self.mem.append(_mib(mem.split("/")[0]))
                except ValueError:
                    pass

    def stop(self) -> None:
        self._stop_event.set()


# ------------------------------------------------------------------ the run


def preflight(base: str, key: str, cached_question: str) -> dict[str, Any]:
    """Fail fast, with a reason, before spending minutes on a broken stack."""
    status, body = _http(f"{base}/healthz", key=key)
    if status != 200:
        sys.exit(f"API not healthy at {base} ({status}). Start benchmarks/compose.bench.yml first.")
    health = json.loads(body)
    # One full answer through the real graph: proves the stand-in model's
    # replies pass the critic and the grounding gate, so the answer scenarios
    # measure answers, not refusals.
    status, body = _http(f"{base}/v1/query", key=key, body={"query": cached_question})
    if status != 200 or '"citations":[{' not in body:
        sys.exit(f"Preflight answer was not a verified answer ({status}): {body[:300]}")
    # Prime the verified-answer cache for the `cached` scenario, and check it hit.
    started = time.perf_counter()
    status, body = _http(f"{base}/v1/query", key=key, body={"query": cached_question})
    primed_ms = (time.perf_counter() - started) * 1000
    if status != 200 or '"citations":[{' not in body:
        sys.exit(f"Cache priming failed ({status}): {body[:300]}")
    return {"health": health, "cache_hit_ms": round(primed_ms, 1)}


def jmeter_cmd(home: Path, jtl: Path, log: Path, props: dict[str, str]) -> list[str]:
    java = shutil.which("java") or sys.exit("java not found on PATH (JMeter needs Java 17+)")
    cmd = [
        java,
        "-Xms1g",
        "-Xmx3g",
        "-XX:+UseG1GC",
        "-jar",
        str(home / "bin" / "ApacheJMeter.jar"),
        "-n",
        "-t",
        str(HERE / "sentinel.jmx"),
        "-l",
        str(jtl),
        "-j",
        str(log),
        "-Jjmeter.save.saveservice.output_format=csv",
        "-Jjmeter.save.saveservice.response_code=true",
        "-Jjmeter.save.saveservice.latency=true",
        "-Jjmeter.save.saveservice.successful=true",
        "-Jjmeter.save.saveservice.assertion_results_failure_message=true",
        # uvicorn closes a keep-alive connection after 5 s idle. A JMeter
        # thread pacing a low rate idles longer than that and can send on a
        # socket the server has just closed (NoHttpResponseException): the
        # request never reaches Sentinel. Retire idle connections at 4 s.
        "-Jhttpclient4.idletimeout=4000",
        "-Jhttpclient4.validate_after_inactivity=1000",
    ]
    cmd += [f"-J{k}={v}" for k, v in props.items()]
    return cmd


def summarise(step: Step, jtl: Path) -> None:
    with jtl.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return
    t0 = min(int(r["timeStamp"]) for r in rows)
    lo, hi = t0 + WARMUP_S * 1000, t0 + step.duration_s * 1000
    window = [r for r in rows if lo <= int(r["timeStamp"]) < hi]
    if not window:
        return
    elapsed = sorted(float(r["elapsed"]) for r in window)
    ttfb = sorted(float(r["Latency"]) for r in window)
    failures = [r for r in window if r["success"] != "true"]
    codes: dict[str, int] = {}
    for r in window:
        codes[r["responseCode"]] = codes.get(r["responseCode"], 0) + 1
    messages: dict[str, int] = {}
    for r in failures:
        msg = (r.get("failureMessage") or r.get("responseMessage") or "")[:120]
        messages[msg] = messages.get(msg, 0) + 1

    step.samples = len(window)
    step.achieved_rps = round(len(window) / ((hi - lo) / 1000), 1)
    step.error_rate = round(len(failures) / len(window), 4)
    step.p50_ms, step.p90_ms = percentile(elapsed, 50), percentile(elapsed, 90)
    step.p95_ms, step.p99_ms = percentile(elapsed, 95), percentile(elapsed, 99)
    step.max_ms = elapsed[-1]
    step.ttfb_p50_ms, step.ttfb_p95_ms = percentile(ttfb, 50), percentile(ttfb, 95)
    step.codes = codes
    step.top_failure = max(messages, key=messages.__getitem__) if messages else ""
    step.passed = step_passes(asdict(step))


def environment(args: argparse.Namespace, home: Path) -> dict[str, Any]:
    def sh(*cmd: str) -> str:
        try:
            return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()
        except OSError:
            return ""

    cpu = platform.processor()
    if platform.system() == "Windows":
        cpu = (
            sh("powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name")
            or cpu
        )
    return {
        "date": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "commit": sh("git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"),
        "host_os": f"{platform.system()} {platform.release()}",
        "host_cpu": cpu,
        "host_logical_cpus": os.cpu_count(),
        "docker": sh("docker", "version", "--format", "{{.Server.Version}}"),
        "api_cpu_limit": os.environ.get("BENCH_API_CPUS", "2"),
        "api_workers": 1,
        "mock_llm_latency_ms": int(os.environ.get("MOCK_LATENCY_MS", "300")),
        "jmeter": home.name,
        "step_duration_s": args.duration,
        "warmup_excluded_s": WARMUP_S,
        "pass_rule": PASS_RULE,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument(
        "--steps", default=None, help="offered requests/second, e.g. 50,100 (default: per scenario)"
    )
    ap.add_argument("--duration", type=int, default=30, help="seconds per step")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", default="8100")
    ap.add_argument("--key", default="admin-local", help="admin: exempt from the daily quota")
    ap.add_argument("--cached-question", default="What is the refund window?")
    ap.add_argument(
        "--jmeter-home", default=os.environ.get("JMETER_HOME", "C:/jmeter/apache-jmeter-5.6.3")
    )
    ap.add_argument("--out", default=None, help="results directory (default results/<timestamp>)")
    args = ap.parse_args()

    home = Path(args.jmeter_home)
    if not (home / "bin" / "ApacheJMeter.jar").exists():
        sys.exit(f"JMeter not found at {home}; set JMETER_HOME or --jmeter-home")
    scenarios = [s for s in args.scenarios.split(",") if s]
    unknown = set(scenarios) - set(SCENARIOS)
    if unknown:
        sys.exit(f"unknown scenarios: {sorted(unknown)}")
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(args.out) if args.out else HERE / "results" / run_id
    out.mkdir(parents=True, exist_ok=True)
    base = f"http://{args.host}:{args.port}"

    print(f"preflight against {base} ...", flush=True)
    pre = preflight(base, args.key, args.cached_question)
    steps: list[dict[str, Any]] = []
    summary = {"environment": environment(args, home), "preflight": pre, "steps": steps}

    for scenario in scenarios:
        rates = [int(r) for r in (args.steps or DEFAULT_STEPS[scenario]).split(",")]
        for rate in rates:
            threads = min(MAX_THREADS, max(50, math.ceil(rate * EXPECTED_LATENCY_S[scenario] * 4)))
            step = Step(scenario, rate, threads, args.duration)
            jtl, log = out / f"{scenario}-{rate}.jtl", out / f"{scenario}-{rate}.log"
            props = {
                "host": args.host,
                "port": args.port,
                "key": args.key,
                "scenario": scenario,
                "rps": str(rate),
                "threads": str(threads),
                "duration": str(args.duration),
                "ramp": "2",
                "questions": str(HERE / "questions.csv"),
                "cached_question": args.cached_question,
            }
            print(f"{scenario:>8} @ {rate:>5}/s  threads={threads:<5}", end=" ", flush=True)
            sampler = StatsSampler(API_CONTAINER)
            sampler.start()
            subprocess.run(jmeter_cmd(home, jtl, log, props), capture_output=True, check=False)
            sampler.stop()
            sampler.join(timeout=10)
            summarise(step, jtl)
            if sampler.cpu:
                step.cpu_mean_pct = round(sum(sampler.cpu) / len(sampler.cpu), 1)
                step.cpu_max_pct = round(max(sampler.cpu), 1)
                step.mem_max_mib = round(max(sampler.mem), 1)
            steps.append(asdict(step))
            (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            verdict = "ok" if step.passed else "SATURATED"
            print(
                f"-> {step.achieved_rps:>7.1f}/s  p50 {step.p50_ms:>6.0f}  "
                f"p95 {step.p95_ms:>6.0f}  p99 {step.p99_ms:>6.0f} ms  "
                f"err {step.error_rate:.2%}  cpu {step.cpu_mean_pct:.0f}%  {verdict}",
                flush=True,
            )
            time.sleep(5)  # let queues drain before the next step
            if not step.passed:
                break

    print(f"\nwrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()

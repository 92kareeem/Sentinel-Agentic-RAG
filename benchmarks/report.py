"""Turn benchmark summaries into the published report: Markdown plus SVG charts.

    python benchmarks/report.py --run benchmarks/results/after \\
        --baseline benchmarks/results/baseline --out docs/benchmarks/2026-11-15

--baseline is optional; with it, every table and chart compares the two runs.
Charts are plain SVG with light and dark palettes, so GitHub renders them in
either theme. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import math
from html import escape
from pathlib import Path
from typing import Any

from run import PASS_RULE, SLO_P95_MS, fail_reason, step_passes  # one rule for runs old and new

SCENARIO_LABEL = {
    "health": "Health check",
    "blocked": "Prompt injection blocked",
    "cached": "Repeated question (cache hit)",
    "answer": "New question, full verified answer",
    "stream": "New question, streamed",
}
FAST = ("health", "blocked", "cached")
GRAPH = ("answer", "stream")

# Reference palette (validated): categorical slots 1-2, text and surface roles.
STYLE = """<style>
  .bg{fill:#fcfcfb} .t1{fill:#0b0b0b} .t2{fill:#52514e} .grid{stroke:#e4e3df} .ref{stroke:#52514e}
  .s1{fill:#2a78d6} .s2{fill:#eb6834} .l1{stroke:#2a78d6} .l2{stroke:#eb6834} .ring{stroke:#fcfcfb}
  @media (prefers-color-scheme: dark){
    .bg{fill:#1a1a19} .t1{fill:#f0efec} .t2{fill:#c3c2b7} .grid{stroke:#383835} .ref{stroke:#c3c2b7}
    .s1{fill:#3987e5} .s2{fill:#d95926} .l1{stroke:#3987e5} .l2{stroke:#d95926}
    .ring{stroke:#1a1a19}
  }
  text{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif}
</style>"""


def load(paths: str | None, code: str | None = None) -> dict[str, Any] | None:
    """One run from one or more result directories, comma-separated.

    A later directory replaces the scenarios it measured, so a run can be
    assembled from a full pass plus a re-measured subset. Every step is
    re-judged by the current pass rule.
    """
    if not paths:
        return None
    merged: dict[str, Any] | None = None
    for path in paths.split(","):
        data: dict[str, Any] = json.loads((Path(path) / "summary.json").read_text(encoding="utf-8"))
        if merged is None:
            merged = data
            continue
        redone = {s["scenario"] for s in data["steps"]}
        merged["steps"] = [s for s in merged["steps"] if s["scenario"] not in redone]
        merged["steps"] += data["steps"]
    assert merged is not None
    for step in merged["steps"]:
        step["passed"] = step_passes(step)
    merged["environment"]["pass_rule"] = PASS_RULE
    if code:
        merged["environment"]["code"] = code
    return merged


def steps_for(run: dict[str, Any], scenario: str) -> list[dict[str, Any]]:
    return [s for s in run["steps"] if s["scenario"] == scenario]


def sustained(run: dict[str, Any], scenario: str) -> tuple[float, bool, dict[str, Any] | None]:
    """(highest delivered rate among passing steps, staircase exhausted?, that step)."""
    steps = steps_for(run, scenario)
    passing = [s for s in steps if s["passed"]]
    if not passing:
        return 0.0, False, None
    best = max(passing, key=lambda s: s["achieved_rps"])
    exhausted = bool(steps) and steps[-1]["passed"]  # never failed: the real limit is higher
    return best["achieved_rps"], exhausted, best


def fmt_rate(value: float, exhausted: bool) -> str:
    return f"{value:,.0f}{'+' if exhausted else ''}"


# ------------------------------------------------------------------ charts


def _svg(width: int, height: int, body: list[str], title: str) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
            f"<title>{escape(title)}</title>",
            STYLE,
            f'<rect class="bg" width="{width}" height="{height}" rx="8"/>',
            *body,
            "</svg>",
        ]
    )


def _bar(x: float, y: float, w: float, h: float, cls: str) -> str:
    """Horizontal bar, square at the baseline, rounded (4px) at the data end."""
    r = min(4.0, h / 2, max(w, 0) / 2)
    if w <= 0:
        return ""
    return (
        f'<path class="{cls}" d="M{x:.1f},{y:.1f} h{w - r:.1f} a{r},{r} 0 0 1 {r},{r} '
        f'v{h - 2 * r:.1f} a{r},{r} 0 0 1 -{r},{r} h-{w - r:.1f} z"/>'
    )


def nice_ticks(vmax: float, target: int = 5) -> list[float]:
    """Round tick values (1, 2, 2.5 or 5 times a power of ten) covering vmax."""
    raw = max(vmax, 1e-9) / target
    power = 10 ** math.floor(math.log10(raw))
    step = next(m * power for m in (1, 2, 2.5, 5, 10) if m * power >= raw)
    count = math.ceil(vmax / step)
    return [i * step for i in range(count + 1)]


def _legend(x: float, y: float, labels: list[tuple[str, str]]) -> list[str]:
    out = []
    for cls, label in labels:
        out.append(f'<rect class="{cls}" x="{x}" y="{y - 9}" width="10" height="10" rx="2"/>')
        out.append(f'<text class="t2" x="{x + 15}" y="{y}" font-size="12">{escape(label)}</text>')
        x += 24 + 7 * len(label)
    return out


def throughput_chart(
    title: str, scenarios: tuple[str, ...], run: dict[str, Any], base: dict[str, Any] | None
) -> str:
    series = [("s1", "Before", base), ("s2", "After", run)] if base else [("s1", "Measured", run)]
    width, left, right, top = 760, 230, 70, 64
    bar_h, gap, group_gap = 14, 2, 18
    group_h = len(series) * bar_h + (len(series) - 1) * gap
    height = top + len(scenarios) * (group_h + group_gap) + 30
    values = {(name, sc): sustained(r, sc) for _, name, r in series if r for sc in scenarios}
    ticks = nice_ticks(max([v[0] for v in values.values()] + [1.0]))
    vmax = ticks[-1]
    scale = (width - left - right) / vmax

    body = [
        f'<text class="t1" x="20" y="28" font-size="15" font-weight="600">{escape(title)}</text>',
        *_legend(20, 50, [(cls, name) for cls, name, _ in series]),
    ]
    # Recessive grid at round values.
    for v in ticks:
        gx = left + v * scale
        body.append(
            f'<line class="grid" x1="{gx:.1f}" y1="{top - 6}" x2="{gx:.1f}" '
            f'y2="{height - 26}" stroke-width="1"/>'
        )
        body.append(
            f'<text class="t2" x="{gx:.1f}" y="{height - 10}" font-size="11" '
            f'text-anchor="middle">{v:,.0f}</text>'
        )
    y = top
    for sc in scenarios:
        body.append(
            f'<text class="t1" x="{left - 10}" y="{y + group_h / 2 + 4:.1f}" font-size="12" '
            f'text-anchor="end">{escape(SCENARIO_LABEL[sc])}</text>'
        )
        for i, (cls, name, _) in enumerate(series):
            value, exhausted, _step = values[(name, sc)]
            by = y + i * (bar_h + gap)
            body.append(_bar(left, by, value * scale, bar_h, cls))
            body.append(
                f'<text class="t2" x="{left + value * scale + 6:.1f}" y="{by + bar_h - 3}" '
                f'font-size="11">{fmt_rate(value, exhausted)}</text>'
            )
        y += group_h + group_gap
    return _svg(width, height, body, title)


def latency_chart(scenario: str, run: dict[str, Any], base: dict[str, Any] | None) -> str:
    title = f"{SCENARIO_LABEL[scenario]}: p95 latency as load rises"
    series = [("1", "Before", base), ("2", "After", run)] if base else [("1", "Measured", run)]
    rates = sorted({s["target_rps"] for _, _, r in series if r for s in steps_for(r, scenario)})
    if not rates:
        return ""
    width, height, left, right, top, bottom = 760, 340, 70, 30, 64, 46
    pmax = max(s["p95_ms"] for _, _, r in series if r for s in steps_for(r, scenario)) or 1.0
    ticks = nice_ticks(max(pmax, SLO_P95_MS[scenario]) * 1.05)
    ymax = ticks[-1]
    pw, ph = width - left - right, height - top - bottom

    def px(rate: int) -> float:
        return left + (rates.index(rate) + 0.5) * pw / len(rates)

    def py(ms: float) -> float:
        return top + ph - ms / ymax * ph

    body = [
        f'<text class="t1" x="20" y="28" font-size="15" font-weight="600">{escape(title)}</text>',
        *_legend(20, 50, [(f"s{k}", name) for k, name, _ in series]),
    ]
    for v in ticks:
        body.append(
            f'<line class="grid" x1="{left}" y1="{py(v):.1f}" x2="{width - right}" '
            f'y2="{py(v):.1f}" stroke-width="1"/>'
        )
        body.append(
            f'<text class="t2" x="{left - 8}" y="{py(v) + 4:.1f}" font-size="11" '
            f'text-anchor="end">{v / 1000:g} s</text>'
        )
    # The latency target a step must stay under, as a labelled reference line.
    slo = SLO_P95_MS[scenario]
    body.append(
        f'<line class="ref" x1="{left}" y1="{py(slo):.1f}" x2="{width - right}" '
        f'y2="{py(slo):.1f}" stroke-width="1.5" stroke-dasharray="6 4"/>'
    )
    body.append(
        f'<text class="t2" x="{width - right - 4}" y="{py(slo) - 6:.1f}" font-size="11" '
        f'text-anchor="end">p95 target {slo / 1000:g} s</text>'
    )
    for rate in rates:
        body.append(
            f'<text class="t2" x="{px(rate):.1f}" y="{height - bottom + 18}" '
            f'font-size="11" text-anchor="middle">{rate}</text>'
        )
    body.append(
        f'<text class="t2" x="{left + pw / 2:.1f}" y="{height - 8}" font-size="11" '
        f'text-anchor="middle">offered load (requests/second)</text>'
    )
    for k, name, r in series:
        if not r:
            continue
        pts = [(px(s["target_rps"]), py(s["p95_ms"]), s) for s in steps_for(r, scenario)]
        d = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y, _) in enumerate(pts))
        body.append(
            f'<path class="l{k}" d="{d}" fill="none" stroke-width="2" stroke-linejoin="round"/>'
        )
        for x, y, _s in pts:
            body.append(
                f'<circle class="s{k} ring" cx="{x:.1f}" cy="{y:.1f}" r="4.5" stroke-width="2"/>'
            )
        x, y, s = pts[-1]
        body.append(
            f'<text class="t2" x="{x + 8:.1f}" y="{y - 8:.1f}" font-size="11">{escape(name)}</text>'
        )
    return _svg(width, height, body, title)


# ------------------------------------------------------------------ markdown


def step_table(run: dict[str, Any], scenario: str) -> list[str]:
    ttfb = scenario == "stream"
    head = (
        "| Offered/s | Delivered/s | p50 | p95 | p99 | Errors | API CPU (100% = 1 core) |"
        + (" First byte p95 |" if ttfb else "")
        + " Result |"
    )
    sep = "|" + "---|" * (8 + (1 if ttfb else 0))
    rows = [head, sep]
    for s in steps_for(run, scenario):
        cells = [
            f"{s['target_rps']}",
            f"{s['achieved_rps']:.1f}",
            f"{s['p50_ms']:.0f} ms",
            f"{s['p95_ms']:.0f} ms",
            f"{s['p99_ms']:.0f} ms",
            f"{s['error_rate']:.2%}",
            f"{s['cpu_mean_pct']:.0f}%",
        ]
        if ttfb:
            cells.append(f"{s['ttfb_p95_ms']:.0f} ms")
        cells.append("ok" if s["passed"] else f"limit: {fail_reason(s)}")
        rows.append("| " + " | ".join(cells) + " |")
    return rows


def headline_table(run: dict[str, Any], base: dict[str, Any] | None) -> list[str]:
    if base:
        rows = ["| Scenario | Before | After | p95 at that rate (after) |", "|---|---:|---:|---:|"]
    else:
        rows = ["| Scenario | Highest sustained rate | p95 at that rate |", "|---|---:|---:|"]
    for sc in (*FAST, *GRAPH):
        if not steps_for(run, sc):
            continue
        value, exhausted, step = sustained(run, sc)
        p95 = f"{step['p95_ms']:.0f} ms" if step else "—"
        if base:
            bv, bex, _ = sustained(base, sc)
            rows.append(
                f"| {SCENARIO_LABEL[sc]} | {fmt_rate(bv, bex)}/s "
                f"| **{fmt_rate(value, exhausted)}/s** | {p95} |"
            )
        else:
            rows.append(f"| {SCENARIO_LABEL[sc]} | **{fmt_rate(value, exhausted)}/s** | {p95} |")
    return rows


def environment_table(env: dict[str, Any]) -> list[str]:
    labels = [
        ("date", "Date"),
        ("code", "Code under test"),
        ("commit", "Commit"),
        ("host_cpu", "Host CPU"),
        ("host_logical_cpus", "Host logical CPUs"),
        ("host_os", "Host OS"),
        ("docker", "Docker"),
        ("api_cpu_limit", "API container CPU limit"),
        ("api_workers", "API worker processes"),
        ("mock_llm_latency_ms", "Stand-in model latency per call (ms)"),
        ("jmeter", "Load generator"),
        ("step_duration_s", "Seconds per step"),
        ("warmup_excluded_s", "Warm-up excluded (s)"),
        ("pass_rule", "A step passes when"),
    ]
    return ["| | |", "|---|---|", *[f"| {label} | {env.get(key, '')} |" for key, label in labels]]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run", required=True, help="result dir(s), comma-separated")
    ap.add_argument("--baseline", help="result dir(s) to compare against")
    ap.add_argument("--run-code", help="what code --run measured, for the report")
    ap.add_argument("--baseline-code", help="what code --baseline measured")
    ap.add_argument("--out", required=True)
    ap.add_argument("--notes", help="Markdown file whose contents go under 'What changed'")
    args = ap.parse_args()
    run, base = load(args.run, args.run_code), load(args.baseline, args.baseline_code)
    assert run is not None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    charts = {
        "throughput-fast.svg": throughput_chart(
            "Highest sustained rate, no model call (req/s)", FAST, run, base
        ),
        "throughput-graph.svg": throughput_chart(
            "Highest sustained rate, full agent graph (req/s)", GRAPH, run, base
        ),
        "latency-answer.svg": latency_chart("answer", run, base),
        "latency-stream.svg": latency_chart("stream", run, base),
    }
    for name, svg in charts.items():
        if svg:
            (out / name).write_text(svg, encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    if base:
        (out / "baseline-summary.json").write_text(json.dumps(base, indent=2), encoding="utf-8")

    md: list[str] = [f"# Load test — {run['environment']['date'][:10]}", ""]
    md += ["## Highest sustained rate", "", *headline_table(run, base), ""]
    md += [
        "A step passes when errors stay at or under 1%, the server delivers at least 90% of the",
        "offered rate, and p95 latency stays under 250 ms for requests that never reach a model",
        "or 2 s for a full answer. `+` means the staircase ran out before the system did.",
        "",
        "![Highest sustained rate, no model call](throughput-fast.svg)",
        "",
        "![Highest sustained rate, full agent graph](throughput-graph.svg)",
        "",
    ]
    if args.notes:
        md += ["## What changed", "", Path(args.notes).read_text(encoding="utf-8").strip(), ""]
    md += ["## Latency under load", ""]
    for name in ("latency-answer.svg", "latency-stream.svg"):
        if charts[name]:
            md += [f"![{name}]({name})", ""]
    md += ["## Every step", ""]
    for sc in (*FAST, *GRAPH):
        if steps_for(run, sc):
            md += [f"### {SCENARIO_LABEL[sc]}", "", *step_table(run, sc), ""]
            if base and steps_for(base, sc):
                md += [
                    "<details><summary>Before</summary>",
                    "",
                    *step_table(base, sc),
                    "",
                    "</details>",
                    "",
                ]
    md += ["## Environment", "", *environment_table(run["environment"]), ""]
    md += [
        "Raw data: [`summary.json`](summary.json)"
        + (" · [`baseline-summary.json`](baseline-summary.json)" if base else ""),
        "",
    ]
    (out / "README.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {out / 'README.md'}")


if __name__ == "__main__":
    main()

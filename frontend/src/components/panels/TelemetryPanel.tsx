import { useEffect, useState } from "react";
import { getTelemetry } from "../../api";
import { NODE_LABEL } from "../../lib/traceGraph";
import type { TelemetryReport } from "../../types";
import { BarList, ColumnChart } from "../charts";
import { Kpi } from "./Kpi";

const WINDOWS = [
  { hours: 1, label: "1 hour" },
  { hours: 24, label: "24 hours" },
  { hours: 24 * 7, label: "7 days" },
];

const pct = (x: number) => `${Math.round(x * 100)}%`;
const secs = (ms: number) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${ms} ms`);
const usd = (x: number) => (x === 0 ? "$0" : x < 0.01 ? `$${x.toFixed(4)}` : `$${x.toFixed(2)}`);

type State = { kind: "loading" } | { kind: "ready"; report: TelemetryReport } | { kind: "error" };

export function TelemetryPanel() {
  const [hours, setHours] = useState(24);
  const [state, setState] = useState<State>({ kind: "loading" });

  useEffect(() => {
    let live = true;
    setState({ kind: "loading" });
    getTelemetry(hours)
      .then((report) => live && setState({ kind: "ready", report }))
      .catch(() => live && setState({ kind: "error" }));
    return () => {
      live = false;
    };
  }, [hours]);

  return (
    <div aria-busy={state.kind === "loading"}>
      <div className="panel-toolbar" role="group" aria-label="Time window">
        {WINDOWS.map((w) => (
          <button
            key={w.hours}
            type="button"
            className="btn btn-secondary btn-sm"
            aria-pressed={hours === w.hours}
            onClick={() => setHours(w.hours)}
          >
            {w.label}
          </button>
        ))}
      </div>

      {state.kind === "loading" && (
        <div className="notice" role="status">
          <span className="spinner" aria-hidden="true" />
          Loading telemetry…
        </div>
      )}
      {state.kind === "error" && (
        <div className="notice" role="alert">
          <p className="notice-title">Telemetry isn&rsquo;t available right now</p>
        </div>
      )}
      {state.kind === "ready" && <Report r={state.report} />}
    </div>
  );
}

function Report({ r }: { r: TelemetryReport }) {
  if (r.requests === 0) {
    return (
      <div className="notice">
        <p className="notice-title">No requests in this window</p>
        <p>Ask a question, or run the chaos demo, and the numbers appear here.</p>
      </div>
    );
  }

  // Bucket a week into days so the chart stays readable.
  const daily = r.window_hours > 48;
  const buckets = daily ? groupByDay(r.hourly) : r.hourly;

  return (
    <>
      <ul className="kpis" aria-label="Summary">
        <Kpi
          label="Requests"
          value={r.requests.toLocaleString()}
          sub={`${r.answered} answered · ${r.refused} refused · ${r.errors} errors`}
          tone="neutral"
        />
        <Kpi
          label="Latency p50 / p95"
          value={`${secs(r.latency_p50_ms)}`}
          sub={`p95 ${secs(r.latency_p95_ms)}`}
          tone={r.latency_p95_ms <= 15000 ? "good" : "warn"}
        />
        <Kpi
          label="Refusal rate"
          value={pct(r.refusal_rate)}
          sub="refusing beats guessing — see Gaps"
          tone={r.refusal_rate <= 0.2 ? "good" : "warn"}
        />
        <Kpi
          label="Estimated cost"
          value={usd(r.est_cost_usd)}
          sub={`${usd(r.est_cost_per_answer_usd)} per answer`}
          tone="neutral"
        />
        <Kpi
          label="Token burn"
          value={`${Math.round(r.token_burn_per_hour).toLocaleString()}/h`}
          sub={`${Math.round(r.tokens_per_request).toLocaleString()} per request`}
          tone="neutral"
        />
        <Kpi
          label="Cache hit rate"
          value={pct(r.cache_hit_rate)}
          sub="verified answers reused at $0"
          tone={r.cache_hit_rate > 0 ? "good" : "neutral"}
        />
        <Kpi
          label="Self-repair rate"
          value={pct(r.repair_rate)}
          sub="answers that needed a rewrite or escalation"
          tone="neutral"
        />
        <Kpi
          label="Budget downgrades"
          value={String(r.downgrades)}
          sub="calls switched to the cheap model"
          tone={r.downgrades ? "warn" : "neutral"}
        />
      </ul>

      <div className="chart-grid">
        <ColumnChart
          title={daily ? "Requests per day" : "Requests per hour"}
          unit="requests"
          bars={buckets.map((b) => ({
            key: b.hour,
            label: daily ? b.hour.slice(5, 10) : b.hour.slice(11, 16),
            value: b.requests,
            tooltip: `${daily ? b.hour.slice(0, 10) : b.hour.slice(11, 16) + " UTC"}: ${b.requests} requests, ${
              b.refused
            } refused${b.p95_ms ? `, p95 ${secs(b.p95_ms)}` : ""}`,
          }))}
        />
        <BarList
          title="Where the time goes (mean per call)"
          rows={r.stages.map((s) => ({
            key: s.name,
            label: NODE_LABEL[s.name] ?? s.name,
            sub: `${s.calls} calls · ${s.tokens.toLocaleString()} tokens`,
            value: s.mean_ms,
            display: secs(Math.round(s.mean_ms)),
          }))}
        />
      </div>

      {Object.keys(r.refusal_reasons).length > 0 && (
        <section className="panel-section">
          <h3 className="section-label">Why requests were refused</h3>
          <ul className="tag-list">
            {Object.entries(r.refusal_reasons).map(([reason, n]) => (
              <li key={reason} className="badge badge-neutral">
                {reason.replace(/_/g, " ").toLowerCase()} · {n}
              </li>
            ))}
          </ul>
        </section>
      )}
    </>
  );
}

function groupByDay(hourly: TelemetryReport["hourly"]): TelemetryReport["hourly"] {
  const days = new Map<string, { requests: number; refused: number; p95_ms: number }>();
  for (const h of hourly) {
    const d = h.hour.slice(0, 10);
    const cur = days.get(d) ?? { requests: 0, refused: 0, p95_ms: 0 };
    cur.requests += h.requests;
    cur.refused += h.refused;
    cur.p95_ms = Math.max(cur.p95_ms, h.p95_ms);
    days.set(d, cur);
  }
  return [...days.entries()].map(([d, v]) => ({ hour: `${d}T00:00Z`, ...v }));
}

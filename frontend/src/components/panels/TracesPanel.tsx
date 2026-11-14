import { useEffect, useState } from "react";
import { getRecentTraces, getTrace } from "../../api";
import type { TraceRecord, TraceSummary } from "../../types";
import { TraceGraph } from "../TraceGraph";

type State = { kind: "loading" } | { kind: "ready"; rows: TraceSummary[] } | { kind: "error" };

const statusTone: Record<string, string> = {
  answered: "badge-good",
  refused: "badge-warn",
  error: "badge-bad",
};

// The agent trace & state inspector: every recent request, and for the one
// selected, the exact path it took through the graph — which node ran, how
// long it took, what it cost, where it repaired and why it stopped.
export function TracesPanel() {
  const [state, setState] = useState<State>({ kind: "loading" });
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<TraceRecord | null>(null);

  useEffect(() => {
    getRecentTraces(30)
      .then((rows) => {
        setState({ kind: "ready", rows });
        if (rows.length) setSelected(rows[0].trace_id);
      })
      .catch(() => setState({ kind: "error" }));
  }, []);

  useEffect(() => {
    if (!selected) return;
    setDetail(null);
    getTrace(selected)
      .then(setDetail)
      .catch(() => setDetail(null));
  }, [selected]);

  if (state.kind === "loading") {
    return (
      <div className="notice" role="status">
        <span className="spinner" aria-hidden="true" />
        Loading traces…
      </div>
    );
  }
  if (state.kind === "error") {
    return (
      <div className="notice" role="alert">
        <p className="notice-title">Traces aren&rsquo;t available right now</p>
      </div>
    );
  }
  if (state.rows.length === 0) {
    return (
      <div className="notice">
        <p className="notice-title">No traces yet</p>
        <p>Every question you ask is traced node by node and appears here.</p>
      </div>
    );
  }

  return (
    <div className="trace-split">
      <ul className="trace-list" aria-label="Recent requests">
        {state.rows.map((t) => (
          <li key={t.trace_id}>
            <button
              type="button"
              className="trace-row"
              aria-current={selected === t.trace_id ? "true" : undefined}
              onClick={() => setSelected(t.trace_id)}
            >
              <span className="trace-q">{t.query_redacted || "(empty)"}</span>
              <span className="gap-meta">
                <span className={`badge ${statusTone[t.final_status] ?? "badge-neutral"}`}>{t.final_status}</span>
                {t.cache_hit && <span className="badge badge-good">cache</span>}
                <span>{(t.latency_ms / 1000).toFixed(1)} s</span>
                <span>{t.total_tokens} tok</span>
                {t.repair_count > 0 && <span>{t.repair_count} repair(s)</span>}
              </span>
            </button>
          </li>
        ))}
      </ul>
      <div className="trace-detail" aria-live="polite">
        {!detail ? (
          <div className="notice" role="status">
            <span className="spinner" aria-hidden="true" />
          </div>
        ) : (
          <>
            <p className="row-title">{detail.query_redacted}</p>
            <TraceGraph steps={detail.steps} />
            <dl className="kv">
              <dt>Trace ID</dt>
              <dd className="mono">{detail.trace_id}</dd>
              <dt>Asked by</dt>
              <dd>{detail.user_id}</dd>
              <dt>Model path</dt>
              <dd className="mono">{detail.model_path.join(" → ") || "—"}</dd>
              <dt>Critic scores</dt>
              <dd>
                {detail.critic_scores.length
                  ? detail.critic_scores
                      .map((c) => `#${c.attempt + 1}: ${c.faithfulness.toFixed(2)} / ${c.relevance.toFixed(2)}`)
                      .join(" · ")
                  : "—"}
              </dd>
              <dt>Cost</dt>
              <dd>
                {detail.total_tokens} tokens · ${Number(detail.est_cost_usd ?? 0).toFixed(5)}
              </dd>
            </dl>
          </>
        )}
      </div>
    </div>
  );
}

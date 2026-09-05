import type { CriticScores, TraceRecord } from "../types";

const STEP_LABEL: Record<string, string> = {
  router: "Router",
  retriever: "Retrieval",
  synthesizer: "Answer synthesis",
  critic: "Verification",
  repair_rewrite: "Repair (rewrite)",
  repair_escalate: "Repair (escalate)",
  grounding_check: "Grounding check",
};

interface Props {
  traceId: string;
  critic: CriticScores;
  repairCount: number;
  model: string;
  latencyMs: number;
  sourceCount: number;
  trace: TraceRecord | null;
}

// Everything a developer wants (trace id, per-step timings, raw critic
// scores) lives here, collapsed by default, instead of being permanently
// visible in the main thread — see brief section 13 ("Advanced / debug
// details"). Nothing here is fabricated: every value is a field the backend
// already returns (QueryResponse.critic/model_used/latency_ms/trace_id, or
// TraceRecord.steps fetched separately via GET /v1/traces/{id}).
export function ResponseDetails({ traceId, critic, repairCount, model, latencyMs, sourceCount, trace }: Props) {
  return (
    <details className="response-details">
      <summary>Response details</summary>
      <dl className="rd-grid">
        <dt>Trace ID</dt>
        <dd className="mono">{traceId}</dd>
        <dt>Retrieval</dt>
        <dd>Hybrid · dense + keyword search</dd>
        <dt>Sources used</dt>
        <dd>{sourceCount}</dd>
        <dt>Model</dt>
        <dd className="mono">{model}</dd>
        <dt>Faithfulness</dt>
        <dd>{critic.faithfulness.toFixed(2)}</dd>
        <dt>Relevance</dt>
        <dd>{critic.relevance.toFixed(2)}</dd>
        <dt>Repairs</dt>
        <dd>{repairCount}</dd>
        <dt>Latency</dt>
        <dd>{(latencyMs / 1000).toFixed(1)}s</dd>
      </dl>
      {trace && trace.steps.length > 0 && (
        <div className="rd-timeline">
          {trace.steps.map((s, i) => (
            <div key={i} className="rd-timeline-row">
              <span>{STEP_LABEL[s.name] ?? s.name}</span>
              <span className="mono">{s.duration_ms} ms</span>
            </div>
          ))}
        </div>
      )}
    </details>
  );
}

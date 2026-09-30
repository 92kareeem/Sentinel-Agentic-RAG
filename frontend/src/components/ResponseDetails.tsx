import type { CriticScores, TraceRecord } from "../types";

const STEP_LABEL: Record<string, string> = {
  router: "Router",
  retriever: "Retrieval",
  synthesizer: "Answer synthesis",
  critic: "Verification",
  repair_rewrite: "Repair (rewrite)",
  repair_escalate: "Repair (escalate)",
  grounding_check: "Grounding check",
  refusal: "Refusal",
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

// What a developer wants — trace id, per-step timings, raw critic scores —
// collapsed by default rather than permanently in the thread. Nothing here is
// invented: every value is a field the backend already returns.
export function ResponseDetails({ traceId, critic, repairCount, model, latencyMs, sourceCount, trace }: Props) {
  return (
    <details className="details">
      <summary>Response details</summary>
      <dl className="kv">
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
        <dd>{(latencyMs / 1000).toFixed(1)} s</dd>
      </dl>
      {trace && trace.steps.length > 0 && (
        <ul className="timeline" aria-label="Time spent per step">
          {trace.steps.map((s, i) => (
            <li key={i}>
              <span>{STEP_LABEL[s.name] ?? s.name}</span>
              <span className="mono">{s.duration_ms} ms</span>
            </li>
          ))}
        </ul>
      )}
    </details>
  );
}

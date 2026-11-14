import type { CriticScores, TraceRecord } from "../types";
import { TraceGraph } from "./TraceGraph";

interface Props {
  traceId: string;
  critic: CriticScores;
  repairCount: number;
  model: string;
  latencyMs: number;
  sourceCount: number;
  trace: TraceRecord | null;
}

function money(usd: number | undefined): string {
  if (!usd) return "$0";
  return usd < 0.01 ? `$${usd.toFixed(5)}` : `$${usd.toFixed(3)}`;
}

// What a developer wants — trace id, the path through the graph, raw critic
// scores, cost — collapsed by default rather than permanently in the thread.
// Nothing here is invented: every value is a field the backend returns.
export function ResponseDetails({ traceId, critic, repairCount, model, latencyMs, sourceCount, trace }: Props) {
  return (
    <details className="details">
      <summary>How this answer was produced</summary>
      {trace && trace.steps.length > 0 && <TraceGraph steps={trace.steps} />}
      <dl className="kv">
        <dt>Trace ID</dt>
        <dd className="mono">{traceId}</dd>
        <dt>Sources used</dt>
        <dd>{sourceCount}</dd>
        <dt>Model</dt>
        <dd className="mono">
          {model}
          {trace?.provider ? ` (${trace.provider})` : ""}
        </dd>
        <dt>Faithfulness</dt>
        <dd>{critic.faithfulness.toFixed(2)}</dd>
        <dt>Relevance</dt>
        <dd>{critic.relevance.toFixed(2)}</dd>
        <dt>Repairs</dt>
        <dd>{repairCount}</dd>
        <dt>Latency</dt>
        <dd>{(latencyMs / 1000).toFixed(1)} s</dd>
        {trace && (
          <>
            <dt>Tokens</dt>
            <dd>{trace.total_tokens}</dd>
            <dt>Estimated cost</dt>
            <dd>{trace.cache_hit ? "$0 (served from the verified-answer cache)" : money(trace.est_cost_usd)}</dd>
          </>
        )}
      </dl>
    </details>
  );
}

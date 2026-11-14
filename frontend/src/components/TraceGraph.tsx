import { buildLanes } from "../lib/traceGraph";
import type { TraceStep } from "../types";

interface Props {
  steps: TraceStep[];
  compact?: boolean;
}

// The agent's actual path through the graph for one request: one row per
// attempt, nodes left to right, and the repair that sent it round again.
// Rendered as an ordered list of lists so a screen reader hears it as
// "attempt 1: router, retriever, ... grounding, rejected; rewrite; attempt 2".
export function TraceGraph({ steps, compact = false }: Props) {
  const lanes = buildLanes(steps);
  if (steps.length === 0) return null;

  return (
    <ol className={`dag${compact ? " dag-compact" : ""}`} aria-label="Agent execution path">
      {lanes.map((lane) => (
        <li key={lane.attempt} className="dag-lane">
          <span className="dag-attempt">Attempt {lane.attempt}</span>
          <ol className="dag-nodes">
            {lane.nodes.map((n, i) => (
              <li key={`${n.name}-${i}`} className={`dag-node dag-${n.tone}`}>
                <span className="dag-name">{n.label}</span>
                {!compact && n.detail && <span className="dag-detail">{n.detail}</span>}
                <span className="dag-meta">
                  {n.durationMs} ms{n.tokens ? ` · ${n.tokens} tok` : ""}
                </span>
              </li>
            ))}
          </ol>
          {lane.repair && (
            <p className="dag-repair">
              <span aria-hidden="true">↻</span> {lane.repair.label}
              {lane.repair.detail ? ` · ${lane.repair.detail}` : ""} — trying again
            </p>
          )}
        </li>
      ))}
    </ol>
  );
}

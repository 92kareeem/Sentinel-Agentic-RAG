import type { TraceStep } from "../types";

const COLORS: Record<string, string> = {
  router: "#7c6ff0",
  retriever: "#3fa7d6",
  synthesizer: "#59cd90",
  critic: "#fac05e",
  repair_rewrite: "#ee6352",
  repair_escalate: "#ee6352",
  grounding_check: "#9c89b8",
};

interface Props {
  steps: TraceStep[];
}

export function TraceTimeline({ steps }: Props) {
  if (!steps.length) return null;
  const total = Math.max(
    ...steps.map((s) => s.started_ms + s.duration_ms),
    1,
  );
  return (
    <section className="timeline">
      <h3>Trace timeline</h3>
      {steps.map((s, i) => (
        <div key={i} className="timeline-row">
          <span className="timeline-label">{s.name}</span>
          <div className="timeline-track">
            <div
              className="timeline-bar"
              style={{
                marginLeft: `${(Math.max(s.started_ms, 0) / total) * 100}%`,
                width: `${Math.max((s.duration_ms / total) * 100, 0.8)}%`,
                background: COLORS[s.name] ?? "#888",
              }}
              title={`${s.name}: ${s.duration_ms} ms · ${s.tokens_in + s.tokens_out} tokens`}
            />
          </div>
          <span className="timeline-ms">{s.duration_ms} ms</span>
        </div>
      ))}
    </section>
  );
}

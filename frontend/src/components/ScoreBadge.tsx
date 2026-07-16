import type { CriticScores } from "../types";

function Dial({ label, value }: { label: string; value: number }) {
  const r = 26;
  const c = 2 * Math.PI * r;
  return (
    <div className="dial">
      <svg viewBox="0 0 64 64" width="64" height="64">
        <circle cx="32" cy="32" r={r} fill="none" stroke="#333" strokeWidth="6" />
        <circle
          cx="32"
          cy="32"
          r={r}
          fill="none"
          stroke={value >= 0.7 ? "#59cd90" : "#ee6352"}
          strokeWidth="6"
          strokeDasharray={`${c * value} ${c}`}
          strokeLinecap="round"
          transform="rotate(-90 32 32)"
        />
        <text x="32" y="37" textAnchor="middle" className="dial-value">
          {value.toFixed(2)}
        </text>
      </svg>
      <span className="dial-label">{label}</span>
    </div>
  );
}

interface Props {
  critic: CriticScores;
  repairCount: number;
  model: string;
  latencyMs: number;
}

export function ScoreBadge({ critic, repairCount, model, latencyMs }: Props) {
  return (
    <section className="scores">
      <Dial label="faithfulness" value={critic.faithfulness} />
      <Dial label="relevance" value={critic.relevance} />
      <div className="score-meta">
        <div>
          <b>{repairCount}</b> repair{repairCount === 1 ? "" : "s"}
        </div>
        <div>{model}</div>
        <div>{(latencyMs / 1000).toFixed(1)}s</div>
      </div>
    </section>
  );
}

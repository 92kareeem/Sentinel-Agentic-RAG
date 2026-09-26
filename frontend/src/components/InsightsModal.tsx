import { useEffect, useState } from "react";
import { getKnowledgeGaps } from "../api";
import type { KnowledgeGap, KnowledgeGapReport } from "../types";
import { DIAGNOSIS_LABEL } from "../types";

interface Props {
  onClose: () => void;
}

type State =
  | { kind: "loading" }
  | { kind: "ready"; report: KnowledgeGapReport }
  | { kind: "error" };

// The only screen in the product written for the person who OWNS the
// documents rather than the person asking questions. Everything else answers
// "what do my documents say?"; this answers "what are my documents missing?",
// which is the question a business is actually paying to have answered.
//
// Written to be read by someone non-technical: no chunk counts, no critic
// scores, no retrieval metrics. Every number here is one a manager could put
// in a status update, and every gap carries an action rather than a diagnosis
// code.
export function InsightsModal({ onClose }: Props) {
  const [state, setState] = useState<State>({ kind: "loading" });

  useEffect(() => {
    getKnowledgeGaps()
      .then((report) => setState({ kind: "ready", report }))
      .catch(() => setState({ kind: "error" }));
  }, []);

  return (
    <div className="modal-scrim" onClick={onClose}>
      <div
        className="modal modal-wide"
        role="dialog"
        aria-modal="true"
        aria-label="Coverage insights"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="modal-header">
          <div>
            <h3>Where your documents fall short</h3>
            <p className="modal-sub">
              Questions your team asked that the documents couldn&rsquo;t answer, grouped by topic.
            </p>
          </div>
          <button className="drawer-close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </header>

        {state.kind === "loading" && (
          <div className="insights-empty">
            <span className="spinner" aria-hidden /> Loading coverage…
          </div>
        )}

        {state.kind === "error" && (
          <div className="insights-empty">Coverage insights aren&rsquo;t available right now.</div>
        )}

        {state.kind === "ready" && <Report report={state.report} />}
      </div>
    </div>
  );
}

function Report({ report }: { report: KnowledgeGapReport }) {
  const pct = Math.round(report.answer_rate * 100);

  if (report.total_questions === 0) {
    // An empty workspace is not a failing one. Rendering 0% here would be a
    // scary, meaningless number — there is no denominator yet.
    return (
      <div className="insights-empty">
        <div className="insights-empty-title">No questions asked yet</div>
        <p>
          Once people start asking, anything your documents can&rsquo;t answer shows up here as a
          to-do list.
        </p>
      </div>
    );
  }

  return (
    <div className="insights">
      <div className="kpis">
        <Kpi
          label="Questions answered"
          value={`${pct}%`}
          tone={pct >= 90 ? "good" : pct >= 75 ? "warn" : "bad"}
          sub={`${report.answered} of ${report.total_questions} in the last ${report.window_days} days`}
        />
        <Kpi
          label="Topics to fix"
          value={String(report.gaps.length)}
          tone={report.gaps.length === 0 ? "good" : "warn"}
          // Not "need documentation": the list now also holds answers readers
          // flagged as wrong, and some of those are ours to fix, not the
          // owner's. Each row's action line says which.
          sub={report.gaps.length === 1 ? "topic needs attention" : "topics need attention"}
        />
        <Kpi
          label="Unanswered questions"
          value={String(report.unanswered)}
          tone={report.unanswered === 0 ? "good" : "warn"}
          sub="each one is someone who had to go ask a colleague"
        />
        <Kpi
          label="Flagged by readers"
          value={String(report.marked_wrong)}
          // Only "bad" when readers actually flagged something. With no
          // ratings at all this is neutral, not good: silence isn't evidence
          // that the answers were right.
          tone={report.answers_rated === 0 ? "neutral" : report.marked_wrong === 0 ? "good" : "bad"}
          sub={
            report.answers_rated === 0
              ? "no answers rated yet"
              : `of ${report.answers_rated} rated answer${report.answers_rated === 1 ? "" : "s"}`
          }
        />
      </div>

      {report.gaps.length === 0 ? (
        <div className="insights-empty">
          <div className="insights-empty-title">Nothing needs attention</div>
          <p>
            Every question in this period was answered from your documents, and no reader flagged an
            answer as wrong.
          </p>
        </div>
      ) : (
        <ol className="gap-list">
          {report.gaps.map((gap, i) => (
            <GapRow key={i} gap={gap} rank={i + 1} />
          ))}
        </ol>
      )}
    </div>
  );
}

function Kpi({
  label,
  value,
  sub,
  tone,
}: {
  label: string;
  value: string;
  sub: string;
  tone: "good" | "warn" | "bad" | "neutral";
}) {
  return (
    <div className={`kpi kpi-${tone}`}>
      <div className="kpi-value">{value}</div>
      <div className="kpi-label">{label}</div>
      <div className="kpi-sub">{sub}</div>
    </div>
  );
}

function GapRow({ gap, rank }: { gap: KnowledgeGap; rank: number }) {
  const [open, setOpen] = useState(false);
  // Only worth expanding when there is more than the topic line to show.
  const extras = gap.example_questions.filter((q) => q !== gap.topic);

  return (
    <li className="gap">
      <div className="gap-head">
        <span className="gap-rank" aria-hidden>
          {rank}
        </span>
        <div className="gap-main">
          <div className="gap-topic">{gap.topic}</div>
          <div className="gap-meta">
            <span className="gap-count">
              asked {gap.question_count} {gap.question_count === 1 ? "time" : "times"}
            </span>
            <span className="gap-diagnosis">{DIAGNOSIS_LABEL[gap.diagnosis] ?? gap.diagnosis}</span>
          </div>
        </div>
      </div>

      <p className="gap-action">{gap.recommended_action}</p>

      {extras.length > 0 && (
        <>
          <button className="gap-toggle" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
            {open ? "Hide" : `Show ${extras.length} other phrasing${extras.length === 1 ? "" : "s"}`}
          </button>
          {open && (
            <ul className="gap-examples">
              {extras.map((q, i) => (
                <li key={i}>{q}</li>
              ))}
            </ul>
          )}
        </>
      )}
    </li>
  );
}

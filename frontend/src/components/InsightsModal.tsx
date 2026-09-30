import { useEffect, useId, useState } from "react";
import { getKnowledgeGaps } from "../api";
import { useDialog } from "../lib/useDialog";
import type { CaseDiagnosis, KnowledgeGap, KnowledgeGapReport } from "../types";
import { DIAGNOSIS_LABEL } from "../types";
import { Icon } from "./Icon";

interface Props {
  onClose: () => void;
}

type State = { kind: "loading" } | { kind: "ready"; report: KnowledgeGapReport } | { kind: "error" };

// The one screen written for whoever OWNS the documents rather than whoever
// is asking questions: "what are my documents missing, and what did readers
// say was wrong?" Written for someone non-technical — every number is one a
// manager could put in a status update, and every row carries an action
// rather than a diagnosis code.
export function InsightsModal({ onClose }: Props) {
  const [state, setState] = useState<State>({ kind: "loading" });
  const ref = useDialog<HTMLDivElement>(true, onClose);
  const titleId = useId();

  useEffect(() => {
    getKnowledgeGaps()
      .then((report) => setState({ kind: "ready", report }))
      .catch(() => setState({ kind: "error" }));
  }, []);

  return (
    <div className="scrim" onClick={onClose}>
      <div
        ref={ref}
        className="dialog dialog-wide"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={`${titleId}-sub`}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="dialog-header">
          <div>
            {/* Named like the button that opens it. It was "Where your
                documents fall short" behind a button labelled "Coverage". */}
            <h2 id={titleId} className="dialog-title">
              Coverage
            </h2>
            <p id={`${titleId}-sub`} className="dialog-sub">
              Questions your documents couldn&rsquo;t answer, and answers readers flagged as wrong, ranked by
              how often they come up.
            </p>
          </div>
          <button
            type="button"
            className="btn btn-ghost btn-icon dialog-close"
            onClick={onClose}
            aria-label="Close coverage"
          >
            <Icon name="close" size={18} />
          </button>
        </div>

        <div className="dialog-body" aria-busy={state.kind === "loading"}>
          {state.kind === "loading" && (
            <div className="notice" role="status">
              <span className="spinner" aria-hidden="true" />
              Loading coverage…
            </div>
          )}
          {state.kind === "error" && (
            <div className="notice" role="alert">
              <p className="notice-title">Coverage isn&rsquo;t available right now</p>
              <p>Please try again in a moment.</p>
            </div>
          )}
          {state.kind === "ready" && <Report report={state.report} />}
        </div>
      </div>
    </div>
  );
}

function Report({ report }: { report: KnowledgeGapReport }) {
  const pct = Math.round(report.answer_rate * 100);

  if (report.total_questions === 0) {
    // An empty workspace is not a failing one. 0% here would be a scary and
    // meaningless number — there is no denominator yet.
    return (
      <div className="notice">
        <p className="notice-title">No questions asked yet</p>
        <p>Once people start asking, anything your documents can&rsquo;t answer shows up here as a to-do list.</p>
      </div>
    );
  }

  return (
    <>
      <ul className="kpis" aria-label="Summary">
        <Kpi
          label="Questions answered"
          value={`${pct}%`}
          tone={pct >= 90 ? "good" : pct >= 75 ? "warn" : "bad"}
          sub={`${report.answered} of ${report.total_questions}, last ${report.window_days} days`}
        />
        <Kpi
          label="Unanswered"
          value={String(report.unanswered)}
          tone={report.unanswered === 0 ? "good" : "warn"}
          sub="people who had to ask a colleague"
        />
        <Kpi
          label="Flagged by readers"
          value={String(report.marked_wrong)}
          // Neutral, not green, when nobody has rated anything: silence is
          // not evidence that the answers were right.
          tone={report.answers_rated === 0 ? "neutral" : report.marked_wrong === 0 ? "good" : "bad"}
          sub={
            report.answers_rated === 0
              ? "no answers rated yet"
              : `of ${report.answers_rated} rated answer${report.answers_rated === 1 ? "" : "s"}`
          }
        />
        <Kpi
          label="Topics to fix"
          value={String(report.gaps.length)}
          tone={report.gaps.length === 0 ? "good" : "warn"}
          sub="ranked below"
        />
      </ul>

      {report.gaps.length === 0 ? (
        <div className="notice">
          <p className="notice-title">Nothing needs attention</p>
          <p>Every question in this period was answered, and no reader flagged an answer as wrong.</p>
        </div>
      ) : (
        <section aria-labelledby="gaps-heading">
          <h3 id="gaps-heading" className="section-label">
            Topics to fix, most asked first
          </h3>
          <ol className="gap-list">
            {report.gaps.map((gap, i) => (
              <GapRow key={`${gap.diagnosis}-${gap.topic}`} gap={gap} rank={i + 1} />
            ))}
          </ol>
        </section>
      )}
    </>
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
  // Read as one phrase ("79% Questions answered, 19 of 24, last 30 days"),
  // not three unrelated fragments. The coloured edge is decoration; the
  // numbers and words carry the meaning.
  return (
    <li className={`kpi kpi-${tone}`}>
      <span className="kpi-value">{value}</span>
      <span className="kpi-label">{label}</span>
      <span className="kpi-sub">{sub}</span>
    </li>
  );
}

// What readers reported is more urgent than a gap: a gap wastes someone's
// time, a wrong answer gets acted on. The badge colour says which is which.
function badgeTone(d: CaseDiagnosis): string {
  if (d === "USER_REPORTED_INCORRECT" || d === "USER_REPORTED_INCOMPLETE") return "badge-bad";
  if (d === "NO_EVIDENCE_FOUND" || d === "EVIDENCE_OFF_TOPIC") return "badge-warn";
  return "badge-neutral";
}

function GapRow({ gap, rank }: { gap: KnowledgeGap; rank: number }) {
  const [open, setOpen] = useState(false);
  const listId = useId();
  // Only worth expanding when there is more than the topic line to show.
  const extras = gap.example_questions.filter((q) => q !== gap.topic);

  return (
    <li className="gap">
      <span className="gap-rank" aria-hidden="true">
        {rank}
      </span>
      <div>
        <h4 className="gap-topic">{gap.topic}</h4>
        <div className="gap-meta">
          <span>
            Asked {gap.question_count} {gap.question_count === 1 ? "time" : "times"}
          </span>
          <span className={`badge ${badgeTone(gap.diagnosis)}`}>
            {DIAGNOSIS_LABEL[gap.diagnosis] ?? gap.diagnosis}
          </span>
        </div>
        <p className="gap-action">{gap.recommended_action}</p>
        {extras.length > 0 && (
          <>
            <button
              type="button"
              className="link-btn"
              onClick={() => setOpen((o) => !o)}
              aria-expanded={open}
              aria-controls={listId}
            >
              {open ? "Hide other phrasings" : `Show ${extras.length} other phrasing${extras.length === 1 ? "" : "s"}`}
            </button>
            {open && (
              <ul id={listId} className="gap-examples">
                {extras.map((q) => (
                  <li key={q}>{q}</li>
                ))}
              </ul>
            )}
          </>
        )}
      </div>
    </li>
  );
}

import { useEffect, useId, useState } from "react";
import { getKnowledgeGaps } from "../../api";
import type { CaseDiagnosis, KnowledgeGap, KnowledgeGapReport } from "../../types";
import { DIAGNOSIS_LABEL } from "../../types";
import { Icon } from "../Icon";
import { Kpi } from "./Kpi";

type State = { kind: "loading" } | { kind: "ready"; report: KnowledgeGapReport } | { kind: "error" };

// The corpus work list (ADR 0007): what people asked that the documents could
// not answer, and answers readers flagged, clustered by topic and ranked by
// how often they come up. Each row carries the fix: upload the missing
// document, after which the question is asked again to verify it now answers.
export function GapsPanel({ onFix }: { onFix: (gap: KnowledgeGap) => void }) {
  const [state, setState] = useState<State>({ kind: "loading" });

  useEffect(() => {
    getKnowledgeGaps()
      .then((report) => setState({ kind: "ready", report }))
      .catch(() => setState({ kind: "error" }));
  }, []);

  if (state.kind === "loading") {
    return (
      <div className="notice" role="status">
        <span className="spinner" aria-hidden="true" />
        Loading coverage…
      </div>
    );
  }
  if (state.kind === "error") {
    return (
      <div className="notice" role="alert">
        <p className="notice-title">Coverage isn&rsquo;t available right now</p>
      </div>
    );
  }
  const report = state.report;
  if (report.total_questions === 0) {
    // An empty workspace is not a failing one. 0% here would be a scary and
    // meaningless number — there is no denominator yet.
    return (
      <div className="notice">
        <p className="notice-title">No questions asked yet</p>
        <p>Anything your documents can&rsquo;t answer shows up here as a to-do list.</p>
      </div>
    );
  }
  const pct = Math.round(report.answer_rate * 100);

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
          tone={report.answers_rated === 0 ? "neutral" : report.marked_wrong === 0 ? "good" : "bad"}
          sub={report.answers_rated === 0 ? "no answers rated yet" : `of ${report.answers_rated} rated`}
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
          <p>Every question was answered, and no reader flagged an answer as wrong.</p>
        </div>
      ) : (
        <ol className="gap-list" aria-label="Topics to fix, most asked first">
          {report.gaps.map((gap, i) => (
            <GapRow key={`${gap.diagnosis}-${gap.topic}`} gap={gap} rank={i + 1} onFix={onFix} />
          ))}
        </ol>
      )}
    </>
  );
}

function badgeTone(d: CaseDiagnosis): string {
  if (d === "USER_REPORTED_INCORRECT" || d === "USER_REPORTED_INCOMPLETE") return "badge-bad";
  if (d === "NO_EVIDENCE_FOUND" || d === "EVIDENCE_OFF_TOPIC") return "badge-warn";
  return "badge-neutral";
}

function GapRow({ gap, rank, onFix }: { gap: KnowledgeGap; rank: number; onFix: (g: KnowledgeGap) => void }) {
  const [open, setOpen] = useState(false);
  const listId = useId();
  const extras = gap.example_questions.filter((q) => q !== gap.topic);
  const fixable = gap.diagnosis === "NO_EVIDENCE_FOUND" || gap.diagnosis === "EVIDENCE_OFF_TOPIC";

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
          <span className={`badge ${badgeTone(gap.diagnosis)}`}>{DIAGNOSIS_LABEL[gap.diagnosis] ?? gap.diagnosis}</span>
        </div>
        <p className="gap-action">{gap.recommended_action}</p>
        <div className="gap-actions">
          {fixable && (
            <button type="button" className="btn btn-primary btn-sm" onClick={() => onFix(gap)}>
              <Icon name="upload" size={14} />
              Upload the missing document &amp; re-verify
            </button>
          )}
          {extras.length > 0 && (
            <button
              type="button"
              className="link-btn"
              onClick={() => setOpen((o) => !o)}
              aria-expanded={open}
              aria-controls={listId}
            >
              {open ? "Hide other phrasings" : `Show ${extras.length} other phrasing${extras.length === 1 ? "" : "s"}`}
            </button>
          )}
        </div>
        {open && (
          <ul id={listId} className="gap-examples">
            {extras.map((q) => (
              <li key={q}>{q}</li>
            ))}
          </ul>
        )}
      </div>
    </li>
  );
}

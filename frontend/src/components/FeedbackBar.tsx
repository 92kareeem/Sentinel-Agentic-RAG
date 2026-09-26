import { type ReactNode, useId, useState } from "react";
import { ApiError, postFeedback } from "../api";
import { CORRECTION_MAX, feedbackDoneCopy, feedbackErrorCopy } from "../lib/feedback";
import type { FeedbackVerdict } from "../types";

// The one place a reader can tell the system it was wrong.
//
// Every other failure signal comes from the graph noticing itself — a
// refusal, a rejected draft. An answer that passed every check and was still
// wrong is invisible to all of them, and it's the failure a reader actually
// remembers. This control exists for that case.
//
// Two shapes, because the two messages fail differently:
//   answer  → thumbs; "not right" opens a choice of wrong vs. partial
//   refusal → one link, "this should have been answered" — a false refusal,
//             reported by the only person positioned to notice one
//
// The correction box is optional but it's the valuable part: a thumbs-down
// says a question failed, a correction says what passing looks like, and
// only the second can become a regression test.

type Variant = "answer" | "refusal";

type State =
  | { kind: "idle" }
  | { kind: "choosing"; verdict: Exclude<FeedbackVerdict, "HELPFUL"> }
  | { kind: "sending"; verdict: FeedbackVerdict }
  | { kind: "done"; verdict: FeedbackVerdict; recorded: boolean }
  | { kind: "error"; verdict: FeedbackVerdict; message: string };

interface Props {
  traceId: string;
  variant: Variant;
}

export function FeedbackBar({ traceId, variant }: Props) {
  const [state, setState] = useState<State>({ kind: "idle" });
  // Kept outside `state` so an error or a cancel-and-reopen never throws away
  // what the reader typed. Retyping a correction is how corrections stop
  // being written.
  const [correction, setCorrection] = useState("");
  const fieldId = useId();

  async function send(verdict: FeedbackVerdict) {
    setState({ kind: "sending", verdict });
    try {
      // feedbackBody drops the correction for HELPFUL; the rule lives there.
      const resp = await postFeedback(traceId, verdict, correction);
      setState({ kind: "done", verdict, recorded: resp.recorded });
    } catch (err) {
      setState({
        kind: "error",
        verdict,
        message: feedbackErrorCopy(err instanceof ApiError ? err.status : undefined),
      });
    }
  }

  if (state.kind === "done") {
    return (
      <div className="feedback-bar" role="status">
        <span className="feedback-done">{feedbackDoneCopy(state.verdict, state.recorded)}</span>
      </div>
    );
  }

  if (state.kind === "choosing" || state.kind === "sending" || state.kind === "error") {
    const verdict = state.verdict;
    // HELPFUL is sent in one click and never opens the form; an error on it
    // is shown inline in the idle row below instead.
    if (verdict !== "HELPFUL") {
      const sending = state.kind === "sending";
      return (
        <div className="feedback-form">
          {variant === "answer" && (
            <div className="feedback-choice" role="radiogroup" aria-label="What was wrong with it?">
              <Chip
                active={verdict === "INCORRECT"}
                disabled={sending}
                onClick={() => setState({ kind: "choosing", verdict: "INCORRECT" })}
              >
                It&rsquo;s wrong
              </Chip>
              <Chip
                active={verdict === "INCOMPLETE"}
                disabled={sending}
                onClick={() => setState({ kind: "choosing", verdict: "INCOMPLETE" })}
              >
                It&rsquo;s missing something
              </Chip>
            </div>
          )}

          <label className="feedback-label" htmlFor={fieldId}>
            {variant === "refusal"
              ? "What’s the answer, and where is it covered? (optional)"
              : "What should it have said? (optional)"}
          </label>
          <textarea
            id={fieldId}
            className="feedback-text"
            rows={3}
            maxLength={CORRECTION_MAX}
            value={correction}
            disabled={sending}
            onChange={(e) => setCorrection(e.target.value)}
            placeholder={
              variant === "refusal"
                ? "e.g. The refund policy covers this — it's 30 days."
                : "e.g. It's 30 days, not 14."
            }
          />

          {state.kind === "error" && (
            <div className="feedback-error" role="alert">
              {state.message}
            </div>
          )}

          <div className="feedback-actions">
            <button
              className="feedback-send"
              disabled={sending}
              onClick={() => void send(verdict)}
            >
              {sending ? "Sending…" : state.kind === "error" ? "Try again" : "Send"}
            </button>
            <button
              className="feedback-cancel"
              disabled={sending}
              onClick={() => setState({ kind: "idle" })}
            >
              Cancel
            </button>
          </div>
        </div>
      );
    }
  }

  const helpfulPending = state.kind === "sending" && state.verdict === "HELPFUL";
  const helpfulError = state.kind === "error" && state.verdict === "HELPFUL" ? state.message : null;

  if (variant === "refusal") {
    return (
      <div className="feedback-bar">
        <button
          className="feedback-link"
          onClick={() => setState({ kind: "choosing", verdict: "INCORRECT" })}
        >
          This should have been answered
        </button>
      </div>
    );
  }

  return (
    <div className="feedback-bar">
      <span className="feedback-prompt">Was this right?</span>
      <button
        className="feedback-icon"
        aria-label="Yes, this was helpful"
        title="Helpful"
        disabled={helpfulPending}
        onClick={() => void send("HELPFUL")}
      >
        <ThumbIcon />
      </button>
      <button
        className="feedback-icon"
        aria-label="No, something is wrong with this answer"
        title="Not right"
        disabled={helpfulPending}
        onClick={() => setState({ kind: "choosing", verdict: "INCORRECT" })}
      >
        <ThumbIcon down />
      </button>
      {helpfulError && (
        <span className="feedback-error" role="alert">
          {helpfulError}
        </span>
      )}
    </div>
  );
}

function Chip({
  active,
  disabled,
  onClick,
  children,
}: {
  active: boolean;
  disabled: boolean;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <button
      role="radio"
      aria-checked={active}
      className={`feedback-chip${active ? " active" : ""}`}
      disabled={disabled}
      onClick={onClick}
    >
      {children}
    </button>
  );
}

function ThumbIcon({ down = false }: { down?: boolean }) {
  return (
    <svg
      width="15"
      height="15"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
      style={down ? { transform: "rotate(180deg)" } : undefined}
    >
      <path d="M7 10v12" />
      <path d="M15 5.88 14 10h5.83a2 2 0 0 1 1.92 2.56l-2.33 8A2 2 0 0 1 17.5 22H4a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h2.76a2 2 0 0 0 1.79-1.11L12 2a3.13 3.13 0 0 1 3 3.88Z" />
    </svg>
  );
}

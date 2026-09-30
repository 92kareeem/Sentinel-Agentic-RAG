import { useEffect, useId, useRef, useState } from "react";
import { ApiError, postFeedback } from "../api";
import { CORRECTION_MAX, feedbackDoneCopy, feedbackErrorCopy } from "../lib/feedback";
import { prefersReducedMotion } from "../lib/useDialog";
import type { FeedbackVerdict } from "../types";
import { Icon } from "./Icon";

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
type Negative = Exclude<FeedbackVerdict, "HELPFUL">;

type State =
  | { kind: "idle" }
  | { kind: "choosing"; verdict: Negative }
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
  const formRef = useRef<HTMLDivElement>(null);
  const openerRef = useRef<HTMLButtonElement>(null);
  // The form is showing while choosing, and stays showing while a negative
  // verdict is being sent or has failed (so the typed correction stays put).
  // HELPFUL never opens it: it is sent in one click from the thumbs row.
  const formOpen =
    state.kind === "choosing" ||
    ((state.kind === "sending" || state.kind === "error") && state.verdict !== "HELPFUL");
  const wasOpen = useRef(false);

  // Opening the form moves focus into it and brings it on screen — it opens
  // BELOW the answer, which on a long answer is below the fold, and a form
  // that appears off-screen reads as a button that did nothing. Closing it
  // (Cancel) returns focus to the button that opened it.
  useEffect(() => {
    if (formOpen && !wasOpen.current) {
      const form = formRef.current;
      form?.scrollIntoView({ block: "nearest", behavior: prefersReducedMotion() ? "auto" : "smooth" });
      form?.querySelector<HTMLElement>("[data-autofocus]")?.focus({ preventScroll: true });
    } else if (!formOpen && wasOpen.current && state.kind === "idle") {
      openerRef.current?.focus();
    }
    wasOpen.current = formOpen;
  }, [formOpen, state.kind]);

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
      <span className="feedback-done" role="status">
        <Icon name="check" size={14} />
        {feedbackDoneCopy(state.verdict, state.recorded)}
      </span>
    );
  }

  if (formOpen) {
    const verdict = state.verdict as Negative;
    const sending = state.kind === "sending";
    return (
      <div className="feedback-form" ref={formRef}>
        {variant === "answer" && (
          // Two toggle buttons (aria-pressed), not radio buttons: the radio
          // role promises arrow-key navigation, which this never had.
          <div className="choice-group" role="group" aria-label="What was wrong with the answer?">
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              aria-pressed={verdict === "INCORRECT"}
              disabled={sending}
              onClick={() => setState({ kind: "choosing", verdict: "INCORRECT" })}
              data-autofocus
            >
              It&rsquo;s wrong
            </button>
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              aria-pressed={verdict === "INCOMPLETE"}
              disabled={sending}
              onClick={() => setState({ kind: "choosing", verdict: "INCOMPLETE" })}
            >
              It&rsquo;s missing something
            </button>
          </div>
        )}

        <label className="field-label" htmlFor={fieldId}>
          {variant === "refusal"
            ? "What’s the answer, and where is it covered? (optional)"
            : "What should it have said? (optional)"}
        </label>
        <textarea
          id={fieldId}
          className="textarea"
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
          {...(variant === "refusal" ? { "data-autofocus": true } : {})}
        />

        {state.kind === "error" && (
          <p className="form-error" role="alert">
            {state.message}
          </p>
        )}

        <div className="form-actions">
          <button type="button" className="btn btn-primary btn-sm" disabled={sending} onClick={() => void send(verdict)}>
            {sending ? "Sending…" : state.kind === "error" ? "Try again" : "Send"}
          </button>
          <button
            type="button"
            className="btn btn-secondary btn-sm"
            disabled={sending}
            onClick={() => setState({ kind: "idle" })}
          >
            Cancel
          </button>
        </div>
      </div>
    );
  }

  const helpfulPending = state.kind === "sending" && state.verdict === "HELPFUL";
  const helpfulError = state.kind === "error" && state.verdict === "HELPFUL" ? state.message : null;

  if (variant === "refusal") {
    return (
      <button
        ref={openerRef}
        type="button"
        className="link-btn muted"
        onClick={() => setState({ kind: "choosing", verdict: "INCORRECT" })}
      >
        This should have been answered
      </button>
    );
  }

  return (
    <div className="feedback-bar" role="group" aria-label="Rate this answer">
      <span className="feedback-prompt" aria-hidden="true">
        Was this right?
      </span>
      <button
        type="button"
        className="btn btn-ghost btn-icon btn-sm"
        aria-label="Yes, this answer was right"
        title="Helpful"
        disabled={helpfulPending}
        onClick={() => void send("HELPFUL")}
      >
        <Icon name="thumb" />
      </button>
      <button
        ref={openerRef}
        type="button"
        className="btn btn-ghost btn-icon btn-sm"
        aria-label="No, something is wrong with this answer"
        title="Not right"
        disabled={helpfulPending}
        onClick={() => setState({ kind: "choosing", verdict: "INCORRECT" })}
      >
        <Icon name="thumb" flip />
      </button>
      {helpfulError && (
        <span className="form-error" role="alert">
          {helpfulError}
        </span>
      )}
    </div>
  );
}

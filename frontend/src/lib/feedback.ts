import type { FeedbackVerdict } from "../types";

// The rules behind the feedback control, kept out of the component so they
// can be tested without a DOM and so each lives in exactly one place.

export const CORRECTION_MAX = 2000; // matches FeedbackRequest.correction in schemas.py

export interface FeedbackBody {
  trace_id: string;
  verdict: FeedbackVerdict;
  correction?: string;
}

/** The request body for POST /v1/feedback.
 *
 * Two rules, both here so neither can drift:
 *
 * - HELPFUL never carries a correction, even if the reader typed one before
 *   changing their mind. A correction on a helpful verdict would be stored
 *   nowhere (HELPFUL opens no case) and would read as a contradiction if it
 *   ever were.
 * - The question is never sent. The server reads it back from the trace, so
 *   a client can't attach an arbitrary question to a real answer — and since
 *   corrections feed the regression suite, that would let a client plant a
 *   "known correct" answer in the tests the system is held to.
 */
export function feedbackBody(
  traceId: string,
  verdict: FeedbackVerdict,
  correction?: string,
): FeedbackBody {
  const trimmed = verdict === "HELPFUL" ? "" : (correction ?? "").trim();
  return trimmed
    ? { trace_id: traceId, verdict, correction: trimmed }
    : { trace_id: traceId, verdict };
}

/** What the reader sees once the server has answered. */
export function feedbackDoneCopy(verdict: FeedbackVerdict, recorded: boolean): string {
  // Say so plainly when nothing new was recorded. Claiming a second verdict
  // landed would be a small lie, and the counts behind the owner's report
  // depend on there being exactly one.
  if (!recorded) return "You’ve already rated this answer — your first rating stands.";
  if (verdict === "HELPFUL") return "Thanks — noted.";
  // Says who finds out, because that's what makes a reader bother next time:
  // a report that visibly goes somewhere gets written again.
  return "Thanks — whoever owns these documents will see this.";
}

/** What the reader sees when sending failed, from the HTTP status if any. */
export function feedbackErrorCopy(status?: number): string {
  // Traces expire after 30 days, and a local server keeps them in memory, so
  // an answer from before a restart has nothing left to attach a rating to.
  if (status === 404) return "This answer is too old to rate now.";
  if (status === 400) return "That correction couldn’t be sent — is it very long?";
  return "Couldn’t send that. Your text is still here — try again.";
}

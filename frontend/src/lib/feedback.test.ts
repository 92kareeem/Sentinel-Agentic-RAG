import { describe, expect, it } from "vitest";
import { CORRECTION_MAX, feedbackBody, feedbackDoneCopy, feedbackErrorCopy } from "./feedback";

describe("feedbackBody", () => {
  it("never sends a correction with a helpful verdict", () => {
    // The reader typed a correction, then decided the answer was fine after
    // all. HELPFUL opens no case, so the text would be stored nowhere — and
    // if it ever were, it would contradict the verdict it came with.
    const body = feedbackBody("t1", "HELPFUL", "Actually it's 30 days.");
    expect(body).toEqual({ trace_id: "t1", verdict: "HELPFUL" });
  });

  it("sends a correction with a wrong verdict, trimmed", () => {
    const body = feedbackBody("t1", "INCORRECT", "  It's 30 days, not 14.\n");
    expect(body).toEqual({ trace_id: "t1", verdict: "INCORRECT", correction: "It's 30 days, not 14." });
  });

  it("omits a correction that is only whitespace", () => {
    // An empty correction is not a correction. Sending "" would store a case
    // that looks like it has an expected answer and doesn't — exactly the
    // kind of row that later becomes a regression test asserting nothing.
    expect(feedbackBody("t1", "INCOMPLETE", "   \n  ")).toEqual({ trace_id: "t1", verdict: "INCOMPLETE" });
    expect(feedbackBody("t1", "INCORRECT")).toEqual({ trace_id: "t1", verdict: "INCORRECT" });
  });

  it("never sends the question", () => {
    // The server reads the question back from the trace. If the client could
    // send one, it could attach any question to a real answer and plant a
    // "known correct" answer in the regression suite.
    const body = feedbackBody("t1", "INCORRECT", "It's 30 days.");
    expect(Object.keys(body).sort()).toEqual(["correction", "trace_id", "verdict"]);
  });

  it("uses the same length limit the server enforces", () => {
    expect(CORRECTION_MAX).toBe(2000);
  });
});

describe("feedbackDoneCopy", () => {
  it("says plainly when a repeat rating was not recorded", () => {
    // The first verdict stands. Claiming a second one landed would be a
    // small lie, and the owner's counts depend on there being exactly one.
    expect(feedbackDoneCopy("INCORRECT", false)).toMatch(/already rated/);
    expect(feedbackDoneCopy("HELPFUL", false)).toMatch(/first rating stands/);
  });

  it("tells someone who reported a problem who will see it", () => {
    expect(feedbackDoneCopy("INCORRECT", true)).toMatch(/owns these documents/);
    expect(feedbackDoneCopy("INCOMPLETE", true)).toMatch(/owns these documents/);
  });

  it("doesn't promise escalation for a helpful rating", () => {
    expect(feedbackDoneCopy("HELPFUL", true)).not.toMatch(/owns these documents/);
  });
});

describe("feedbackErrorCopy", () => {
  it("explains an expired answer rather than blaming the network", () => {
    expect(feedbackErrorCopy(404)).toMatch(/too old/);
  });

  it("reassures that typed text survived a transient failure", () => {
    expect(feedbackErrorCopy(503)).toMatch(/still here/);
    expect(feedbackErrorCopy(undefined)).toMatch(/still here/);
  });
});

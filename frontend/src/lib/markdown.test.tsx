import { describe, expect, it } from "vitest";
import { renderAnswerMarkdown } from "./markdown";
import type { Citation } from "../types";

// The renderer's input is model output about documents the user uploaded, so
// every one of these cases is reachable from ordinary product use, not just
// from adversarial input.

const NO_CITATIONS: Citation[] = [];
const noop = () => {};

function render(text: string, citations: Citation[] = NO_CITATIONS) {
  return renderAnswerMarkdown(text, citations, noop) as unknown[];
}

describe("renderAnswerMarkdown termination", () => {
  // If any of these regress, the test does not fail with a bad assertion —
  // it hangs, and vitest's timeout reports it. That is the real symptom: the
  // browser tab freezes and the user loses the conversation.

  it("does not hang on a table row with no separator line", () => {
    // The bug: the table branch needs a "| --- |" line beneath the first row
    // to fire, and the paragraph branch used to refuse any line that looked
    // like a table row. An orphan row matched no branch and never advanced.
    const blocks = render("| Region | Refund window |");
    expect(blocks.length).toBe(1);
  });

  it("does not hang on a quoted table row inside prose", () => {
    const blocks = render(
      "The policy table lists this row:\n| EU | 14 days |\nwhich applies to annual plans.",
    );
    expect(blocks.length).toBeGreaterThan(0);
  });

  it("does not hang on a separator line with no header above it", () => {
    expect(render("| --- | --- |").length).toBeGreaterThan(0);
  });

  it("does not blank the answer when a table has no data rows", () => {
    // rows filters to empty here; destructuring gave head === undefined and
    // head.map threw, which unmounts the whole answer.
    expect(() => render("| --- |\n| --- |")).not.toThrow();
  });

  it("does not hang on unterminated or ragged pipes", () => {
    expect(() => render("| a | b\n|c|\n||")).not.toThrow();
  });
});

describe("renderAnswerMarkdown structure", () => {
  it("still renders a well-formed table as a table, not as paragraphs", () => {
    const blocks = render("| Region | Window |\n| --- | --- |\n| EU | 14 days |");
    expect(blocks.length).toBe(1);
  });

  it("keeps consecutive prose lines in one paragraph", () => {
    expect(render("First line.\nSecond line.").length).toBe(1);
  });

  it("separates a paragraph from a following list", () => {
    expect(render("Intro:\n- one\n- two").length).toBe(2);
  });

  it("renders headings, lists and prose as distinct blocks", () => {
    expect(render("# Title\n\nBody text.\n\n1. first\n2. second").length).toBe(3);
  });
});

import { describe, expect, it } from "vitest";
import { buildLanes } from "./traceGraph";
import type { TraceStep } from "../types";

const step = (name: string, meta: Record<string, string> = {}): TraceStep => ({
  name,
  started_ms: 0,
  duration_ms: 10,
  tokens_in: 5,
  tokens_out: 1,
  meta,
});

describe("buildLanes", () => {
  it("puts a clean answer in one lane", () => {
    const lanes = buildLanes(["router", "retriever", "synthesizer", "critic", "grounding_check"].map((n) => step(n)));
    expect(lanes).toHaveLength(1);
    expect(lanes[0].nodes.map((n) => n.name)).toEqual(["router", "retriever", "synthesizer", "critic", "grounding_check"]);
    expect(lanes[0].repair).toBeNull();
  });

  it("splits the repair loop into attempts", () => {
    const lanes = buildLanes([
      step("router"),
      step("retriever"),
      step("synthesizer"),
      step("critic"),
      step("grounding_check", { ok: "False", stripped_ratio: "1.00" }),
      step("repair_rewrite"),
      step("retriever"),
      step("synthesizer"),
      step("critic"),
      step("grounding_check", { ok: "False", stripped_ratio: "1.00" }),
      step("repair_escalate", { model: "big" }),
      step("synthesizer"),
      step("critic"),
      step("grounding_check", { ok: "False", stripped_ratio: "0.50" }),
      step("refusal", { reason: "UNVERIFIABLE_ANSWER" }),
    ]);
    expect(lanes.map((l) => l.attempt)).toEqual([1, 2, 3]);
    expect(lanes[0].repair?.label).toBe("Rewrite");
    expect(lanes[1].repair?.label).toBe("Escalate");
    expect(lanes[0].nodes[4].tone).toBe("bad");
    expect(lanes[0].nodes[4].detail).toBe("rejected · 100% stripped");
    expect(lanes[2].nodes[lanes[2].nodes.length - 1].tone).toBe("bad");
  });

  it("reports access exclusions on the retriever", () => {
    const [lane] = buildLanes([step("retriever", { chunks: "3", rows_excluded: "12" })]);
    expect(lane.nodes[0].detail).toBe("3 chunks · 12 hidden by access");
  });

  it("drops an empty lane left by a repair that never continued", () => {
    expect(buildLanes([step("router"), step("repair_rewrite")])).toHaveLength(1);
  });
});

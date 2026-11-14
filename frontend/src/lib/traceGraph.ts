// Turn a flat list of trace steps into the shape the trace inspector draws:
// one lane per attempt, each lane the nodes that ran in it, and the repair
// that led from one lane to the next.
//
// The graph's repair loop is what makes Sentinel self-healing, and a flat
// list of eleven step names hides it. Lanes make it visible: attempt 1 ran
// the pipeline and failed grounding, a rewrite started attempt 2, and so on.

import type { TraceStep } from "../types";

export const NODE_LABEL: Record<string, string> = {
  cache: "Cache",
  router: "Router",
  retriever: "Retriever",
  synthesizer: "Synthesizer",
  critic: "Critic",
  grounding_check: "Grounding",
  repair_rewrite: "Rewrite",
  repair_escalate: "Escalate",
  refusal: "Refusal",
};

export type NodeTone = "ok" | "bad" | "repair" | "neutral";

export interface LaneNode {
  name: string;
  label: string;
  durationMs: number;
  tokens: number;
  tone: NodeTone;
  detail: string;
}

export interface Lane {
  attempt: number;
  nodes: LaneNode[];
  // The repair node that ENDED this lane and started the next, if any.
  repair: LaneNode | null;
}

function toneFor(step: TraceStep): NodeTone {
  const meta = step.meta ?? {};
  if (step.name === "refusal") return "bad";
  if (step.name.startsWith("repair")) return "repair";
  if (step.name === "grounding_check") return meta.ok === "False" ? "bad" : "ok";
  return "neutral";
}

function detailFor(step: TraceStep): string {
  const m = step.meta ?? {};
  switch (step.name) {
    case "router":
      return [m.label, m.model].filter(Boolean).join(" · ");
    case "retriever":
      return [`${m.chunks ?? 0} chunks`, m.rows_excluded && m.rows_excluded !== "0" ? `${m.rows_excluded} hidden by access` : ""]
        .filter(Boolean)
        .join(" · ");
    case "synthesizer":
      return [m.model, m.downgraded === "True" ? "downgraded for budget" : ""].filter(Boolean).join(" · ");
    case "grounding_check":
      return m.ok === "False"
        ? `rejected · ${Math.round(Number(m.stripped_ratio ?? 0) * 100)}% stripped`
        : `passed${m.repaired_citations && m.repaired_citations !== "0" ? ` · ${m.repaired_citations} citation(s) re-pointed` : ""}`;
    case "repair_escalate":
      return m.model ?? "";
    case "refusal":
      return m.reason ?? "";
    case "cache":
      return "verified answer reused";
    default:
      return "";
  }
}

export function buildLanes(steps: TraceStep[]): Lane[] {
  const lanes: Lane[] = [{ attempt: 1, nodes: [], repair: null }];
  for (const step of steps) {
    const node: LaneNode = {
      name: step.name,
      label: NODE_LABEL[step.name] ?? step.name,
      durationMs: step.duration_ms,
      tokens: (step.tokens_in ?? 0) + (step.tokens_out ?? 0),
      tone: toneFor(step),
      detail: detailFor(step),
    };
    const current = lanes[lanes.length - 1];
    if (step.name.startsWith("repair")) {
      current.repair = node;
      lanes.push({ attempt: current.attempt + 1, nodes: [], repair: null });
    } else {
      current.nodes.push(node);
    }
  }
  // A trailing empty lane appears only if a repair was the last step (the
  // request ended mid-repair, e.g. on a deadline). Drop it.
  if (lanes.length > 1 && lanes[lanes.length - 1].nodes.length === 0) lanes.pop();
  return lanes;
}

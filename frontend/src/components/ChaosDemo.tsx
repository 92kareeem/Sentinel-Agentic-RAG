import { useId, useState } from "react";
import { ApiError, runChaos } from "../api";
import { useDialog } from "../lib/useDialog";
import type { ChaosResult, ChaosScenario } from "../types";
import { CHAOS_SCENARIOS } from "../types";
import { Icon } from "./Icon";
import { TraceGraph } from "./TraceGraph";

const INTRO: Record<ChaosScenario, { title: string; what: string }> = {
  prompt_injection: { title: "Prompt injection", what: "Tries to override the system's instructions." },
  pii_leak: { title: "PII leak", what: "Sends an email, phone number and SSN with a question." },
  hallucination_trap: {
    title: "Hallucination trap",
    what: "Fault injection: the model fabricates a fee and the judge is fooled.",
  },
  out_of_scope: { title: "Out-of-scope question", what: "Asks something no document covers." },
  budget_exceeded: { title: "Budget exhaustion", what: "Runs a costly question on a tiny token budget." },
};

type Slot = { kind: "idle" } | { kind: "running" } | { kind: "done"; r: ChaosResult } | { kind: "error"; msg: string };

// One click fires five attacks through the real pipeline and shows which
// layer stopped each one. Only offered when the server reports demo mode.
export function ChaosDemo({ onClose }: { onClose: () => void }) {
  const [slots, setSlots] = useState<Record<ChaosScenario, Slot>>(
    () => Object.fromEntries(CHAOS_SCENARIOS.map((s) => [s, { kind: "idle" }])) as Record<ChaosScenario, Slot>,
  );
  const [running, setRunning] = useState(false);
  const ref = useDialog<HTMLDivElement>(true, () => !running && onClose());
  const titleId = useId();

  const set = (s: ChaosScenario, slot: Slot) => setSlots((prev) => ({ ...prev, [s]: slot }));

  async function runOne(s: ChaosScenario) {
    set(s, { kind: "running" });
    try {
      set(s, { kind: "done", r: await runChaos(s) });
    } catch (e) {
      set(s, { kind: "error", msg: e instanceof ApiError ? e.detail : "The request failed." });
    }
  }

  async function runAll() {
    setRunning(true);
    // Sequential on purpose: it reads as a story, and it keeps a free-tier
    // provider's per-minute token limit out of the demo.
    for (const s of CHAOS_SCENARIOS) await runOne(s);
    setRunning(false);
  }

  const done = CHAOS_SCENARIOS.filter((s) => slots[s].kind === "done");
  const passed = done.filter((s) => (slots[s] as { r: ChaosResult }).r.passed).length;

  return (
    <div className="scrim" onClick={() => !running && onClose()}>
      <div
        ref={ref}
        className="dialog dialog-wide dialog-tall"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="dialog-header">
          <div>
            <h2 id={titleId} className="dialog-title">
              Adversarial &amp; chaos demo
            </h2>
            <p className="dialog-sub">
              Five attacks through the real guardrails, graph and grounding gate. Watch which layer stops each one.
            </p>
          </div>
          <button type="button" className="btn btn-ghost btn-icon dialog-close" onClick={onClose} disabled={running} aria-label="Close demo">
            <Icon name="close" size={18} />
          </button>
        </div>
        <div className="dialog-body">
          <div className="panel-toolbar">
            <button type="button" className="btn btn-primary" onClick={() => void runAll()} disabled={running} data-autofocus>
              <Icon name="bolt" />
              {running ? "Running…" : done.length ? "Run all again" : "Run all five attacks"}
            </button>
            {done.length > 0 && (
              <span className={`chaos-score ${passed === done.length ? "is-good" : "is-bad"}`} role="status">
                <Icon name={passed === done.length ? "shield" : "alert"} size={14} />
                {passed} of {done.length} handled as designed
              </span>
            )}
          </div>
          <ol className="chaos-list">
            {CHAOS_SCENARIOS.map((s, i) => (
              <ChaosCard key={s} n={i + 1} scenario={s} slot={slots[s]} onRun={() => void runOne(s)} disabled={running} />
            ))}
          </ol>
        </div>
      </div>
    </div>
  );
}

function ChaosCard({
  n,
  scenario,
  slot,
  onRun,
  disabled,
}: {
  n: number;
  scenario: ChaosScenario;
  slot: Slot;
  onRun: () => void;
  disabled: boolean;
}) {
  const intro = INTRO[scenario];
  const r = slot.kind === "done" ? slot.r : null;
  const tone = !r ? "" : r.passed ? " is-good" : " is-bad";

  return (
    <li className={`chaos-card${tone}`}>
      <div className="chaos-head">
        <span className="gap-rank" aria-hidden="true">
          {n}
        </span>
        <div className="chaos-titles">
          <h3 className="gap-topic">{intro.title}</h3>
          <p className="chaos-what">{intro.what}</p>
        </div>
        {slot.kind === "running" ? (
          <span className="spinner" role="status" aria-label="Running" />
        ) : (
          <button type="button" className="btn btn-secondary btn-sm" onClick={onRun} disabled={disabled}>
            {r ? "Run again" : "Run"}
          </button>
        )}
      </div>

      {slot.kind === "error" && (
        <p className="form-error" role="alert">
          {slot.msg}
        </p>
      )}

      {r && (
        <div className="chaos-result">
          <p className="chaos-attack">
            <span className="eyebrow">Attack</span>
            {r.attack}
          </p>
          <p className="chaos-verdict">
            <Icon name={r.passed ? "shield" : "alert"} size={14} />
            <strong>{r.outcome}</strong> by {r.caught_by}
            {r.http_status !== 200 && <span className="badge badge-neutral">HTTP {r.http_status}</span>}
            {r.refusal_reason && <span className="badge badge-neutral">{r.refusal_reason}</span>}
            {r.simulated && (
              <span className="badge badge-neutral" title="A model call was replaced by a deterministic stand-in">
                simulated model
              </span>
            )}
          </p>
          <p className="chaos-detail">{r.detail}</p>
          {Object.keys(r.evidence).length > 0 && (
            <dl className="kv">
              {Object.entries(r.evidence).map(([k, v]) => (
                <div key={k} style={{ display: "contents" }}>
                  <dt>{k.replace(/_/g, " ")}</dt>
                  <dd className={k.includes("sees") || k.includes("draft") ? "mono" : undefined}>{v}</dd>
                </div>
              ))}
            </dl>
          )}
          {r.steps.length > 0 && (
            <details className="details">
              <summary>Path through the graph</summary>
              <TraceGraph steps={r.steps} compact />
            </details>
          )}
          {r.adrs.length > 0 && (
            <ul className="tag-list">
              {r.adrs.map((a) => (
                <li key={a} className="badge badge-neutral">
                  {a}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </li>
  );
}

import { useCallback, useEffect, useId, useState } from "react";
import { ApiError, getRatchet, promoteCase, retireRegression } from "../../api";
import type { RatchetCandidate, RatchetItem, RatchetView } from "../../types";
import { DIAGNOSIS_LABEL } from "../../types";
import { Icon } from "../Icon";
import { Kpi } from "./Kpi";

type State = { kind: "loading" } | { kind: "ready"; view: RatchetView } | { kind: "error" };

// The quality ratchet (ADR 0008). Every failure a person confirms becomes a
// per-question test that, once it passes, can never fail again unnoticed.
//
// Promotion stays a human act: an admin clicks, reviews the reference answer
// (a reader's correction is a claim, not a fact), and the test lands
// PENDING — tracked, not blocking — until an eval run shows it pass and
// locks it. In a deployed environment the suite lives in git, so the panel
// shows the exact command instead of a button.
export function RatchetPanel() {
  const [state, setState] = useState<State>({ kind: "loading" });

  const load = useCallback(() => {
    getRatchet()
      .then((view) => setState({ kind: "ready", view }))
      .catch(() => setState({ kind: "error" }));
  }, []);
  useEffect(load, [load]);

  if (state.kind === "loading") {
    return (
      <div className="notice" role="status">
        <span className="spinner" aria-hidden="true" />
        Loading the regression suite…
      </div>
    );
  }
  if (state.kind === "error") {
    return (
      <div className="notice" role="alert">
        <p className="notice-title">The regression suite isn&rsquo;t available right now</p>
      </div>
    );
  }
  const v = state.view;

  return (
    <>
      <ul className="kpis" aria-label="Regression suite">
        <Kpi label="Enforced" value={String(v.enforced)} tone={v.enforced ? "good" : "neutral"} sub="a failure here fails CI" />
        <Kpi label="Pending" value={String(v.pending)} tone={v.pending ? "warn" : "neutral"} sub="tracked until they pass" />
        <Kpi label="Retired" value={String(v.retired)} tone="neutral" sub="kept on the record, never deleted" />
        <Kpi label="To review" value={String(v.candidates.length)} tone={v.candidates.length ? "warn" : "good"} sub="failures not yet tests" />
      </ul>

      <p className="panel-note">
        After a fix, verify and lock passing tests:{" "}
        <CopyCode text={v.verify_command} />
      </p>

      <section className="panel-section" aria-labelledby="cand-h">
        <h3 id="cand-h" className="section-label">
          Failures waiting for review
        </h3>
        {v.candidates.length === 0 ? (
          <p className="panel-empty">Nothing to review. New refusals and reader reports appear here.</p>
        ) : (
          <ul className="row-list">
            {v.candidates.map((c) => (
              <Candidate key={c.case_id} c={c} writable={v.writable} onDone={load} />
            ))}
          </ul>
        )}
      </section>

      <section className="panel-section" aria-labelledby="suite-h">
        <h3 id="suite-h" className="section-label">
          The suite
        </h3>
        {v.items.length === 0 ? (
          <p className="panel-empty">No regressions yet. Promote a reviewed failure to start the ratchet.</p>
        ) : (
          <ul className="row-list">
            {v.items.map((it) => (
              <SuiteItem key={it.id} it={it} writable={v.writable} onDone={load} />
            ))}
          </ul>
        )}
      </section>
    </>
  );
}

function Candidate({ c, writable, onDone }: { c: RatchetCandidate; writable: boolean; onDone: () => void }) {
  const [open, setOpen] = useState(false);
  const [expect, setExpect] = useState<"answer" | "refuse">("answer");
  const [reference, setReference] = useState(c.correction ?? "");
  const [note, setNote] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const refId = useId();

  async function promote() {
    setBusy(true);
    setError(null);
    try {
      await promoteCase({
        case_id: c.case_id,
        expect,
        reference: expect === "answer" ? reference.trim() || undefined : undefined,
        note: note.trim() || undefined,
      });
      onDone();
    } catch (e) {
      setError(e instanceof ApiError ? e.detail : "Promotion failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <li className="row">
      <div className="row-main">
        <p className="row-title">{c.question}</p>
        <div className="gap-meta">
          <span className="badge badge-warn">{DIAGNOSIS_LABEL[c.diagnosis] ?? c.diagnosis}</span>
          <span>{c.outcome === "REFUSED" ? "refused" : "answered"}</span>
          <span>asked {c.asked_at.slice(0, 10)}</span>
          <span>by {c.owner_id}</span>
        </div>
        {c.correction && <p className="row-quote">Reader says: &ldquo;{c.correction}&rdquo;</p>}
      </div>
      {writable ? (
        !open ? (
          <button type="button" className="btn btn-secondary btn-sm" onClick={() => setOpen(true)}>
            <Icon name="lock" size={14} />
            Make it a test
          </button>
        ) : (
          <div className="feedback-form">
            <div className="choice-group" role="group" aria-label="What must the system do?">
              <button type="button" className="btn btn-secondary btn-sm" aria-pressed={expect === "answer"} onClick={() => setExpect("answer")}>
                Must answer
              </button>
              <button type="button" className="btn btn-secondary btn-sm" aria-pressed={expect === "refuse"} onClick={() => setExpect("refuse")}>
                Must always refuse
              </button>
            </div>
            {expect === "answer" ? (
              <>
                <label className="field-label" htmlFor={refId}>
                  Correct answer (reviewed — a reader&rsquo;s correction is a claim, not a fact)
                </label>
                <textarea id={refId} className="textarea" rows={2} value={reference} onChange={(e) => setReference(e.target.value)} />
              </>
            ) : (
              <>
                <label className="field-label" htmlFor={refId}>
                  Why is this permanently out of scope?
                </label>
                <textarea id={refId} className="textarea" rows={2} value={note} onChange={(e) => setNote(e.target.value)} />
              </>
            )}
            {error && (
              <p className="form-error" role="alert">
                {error}
              </p>
            )}
            <div className="form-actions">
              <button type="button" className="btn btn-primary btn-sm" disabled={busy} onClick={() => void promote()}>
                {busy ? "Promoting…" : "Promote as pending"}
              </button>
              <button type="button" className="btn btn-secondary btn-sm" disabled={busy} onClick={() => setOpen(false)}>
                Cancel
              </button>
            </div>
          </div>
        )
      ) : (
        <CopyCode text={c.promote_command} />
      )}
    </li>
  );
}

function SuiteItem({ it, writable, onDone }: { it: RatchetItem; writable: boolean; onDone: () => void }) {
  const [retiring, setRetiring] = useState(false);
  const [reason, setReason] = useState("");
  const [error, setError] = useState<string | null>(null);
  const reasonId = useId();

  async function retire() {
    setError(null);
    try {
      await retireRegression(it.id, reason.trim());
      onDone();
    } catch (e) {
      setError(e instanceof ApiError ? e.detail : "Could not retire.");
    }
  }

  return (
    <li className="row">
      <div className="row-main">
        <p className="row-title">
          <span className="mono row-id">{it.id}</span> {it.question}
        </p>
        <div className="gap-meta">
          <span className={`badge ${it.status === "enforced" ? "badge-good" : "badge-warn"}`}>{it.status}</span>
          <span>must {it.expect}</span>
          <span>promoted {it.promoted_at.slice(0, 10)}</span>
        </div>
        {it.reference_answer && <p className="row-quote">Must contain: &ldquo;{it.reference_answer}&rdquo;</p>}
        {it.note && <p className="row-quote">{it.note}</p>}
      </div>
      {writable &&
        (retiring ? (
          <div className="feedback-form">
            <label className="field-label" htmlFor={reasonId}>
              Why is this lesson no longer true?
            </label>
            <textarea id={reasonId} className="textarea" rows={2} value={reason} onChange={(e) => setReason(e.target.value)} />
            {error && (
              <p className="form-error" role="alert">
                {error}
              </p>
            )}
            <div className="form-actions">
              <button type="button" className="btn btn-primary btn-sm" disabled={reason.trim().length < 3} onClick={() => void retire()}>
                Retire on the record
              </button>
              <button type="button" className="btn btn-secondary btn-sm" onClick={() => setRetiring(false)}>
                Cancel
              </button>
            </div>
          </div>
        ) : (
          <button type="button" className="btn btn-ghost btn-sm" onClick={() => setRetiring(true)}>
            Retire
          </button>
        ))}
    </li>
  );
}

export function CopyCode({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <span className="copy-code">
      <code>{text}</code>
      <button
        type="button"
        className="btn btn-ghost btn-icon btn-sm"
        aria-label={copied ? "Copied" : "Copy command"}
        title="Copy"
        onClick={() => {
          void navigator.clipboard?.writeText(text).then(() => setCopied(true));
        }}
      >
        <Icon name={copied ? "check" : "copy"} size={14} />
      </button>
    </span>
  );
}

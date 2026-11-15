import { useId, useRef, useState } from "react";
import { ApiError, uploadDocument } from "../api";
import { useDialog } from "../lib/useDialog";
import type { ClassificationLevel } from "../types";
import { CLASSIFICATION_LEVELS, DOCUMENT_ERROR_HELP } from "../types";
import { Icon } from "./Icon";

export interface UploadContext {
  // Set when the upload is fixing a knowledge gap: the topic is shown, and
  // the questions are asked again once indexing finishes.
  topic: string;
  questions: string[];
}

interface Props {
  onClose: () => void;
  onUploaded: (docId: string) => void;
  context?: UploadContext | null;
}

type Status = { kind: "idle" } | { kind: "busy"; stage: string } | { kind: "error"; msg: string };

// What the backend accepts. The hint text and the file picker both read from
// here, so they cannot disagree with each other again.
const ACCEPT = ".pdf,.md,.txt";

const LEVEL_HELP: Record<ClassificationLevel, string> = {
  PUBLIC: "Anyone in your organisation, including contractors",
  INTERNAL: "Employees",
  CONFIDENTIAL: "Confidential clearance and above",
  RESTRICTED: "Restricted clearance only",
};

export function UploadModal({ onClose, onUploaded, context }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [status, setStatus] = useState<Status>({ kind: "idle" });
  const [dragOver, setDragOver] = useState(false);
  const [classification, setClassification] = useState<ClassificationLevel>("INTERNAL");
  const [roles, setRoles] = useState("");
  const busy = status.kind === "busy";
  // A dialog that is mid-upload should not vanish on Escape or a stray click
  // outside: the upload would carry on with nothing on screen to say so.
  const ref = useDialog<HTMLDivElement>(true, () => {
    if (!busy) onClose();
  });
  const titleId = useId();

  const handleFile = async (file: File) => {
    try {
      const result = await uploadDocument(
        file,
        {
          classification,
          roles: roles
            .split(",")
            .map((r) => r.trim().toLowerCase())
            .filter(Boolean),
        },
        (stage) => setStatus({ kind: "busy", stage }),
      );
      onUploaded(result.doc_id);
    } catch (err) {
      let msg = "That document couldn't be uploaded. Please try again.";
      if (err instanceof ApiError) {
        const help = err.errorCode ? DOCUMENT_ERROR_HELP[err.errorCode] : undefined;
        msg = help ?? err.detail ?? msg;
      }
      setStatus({ kind: "error", msg });
    }
  };

  return (
    <div className="scrim" onClick={() => !busy && onClose()}>
      <div
        ref={ref}
        className="dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="dialog-header">
          <div>
            <h2 id={titleId} className="dialog-title">
              {context ? "Fill a knowledge gap" : "Upload a document"}
            </h2>
            {context && (
              <p className="dialog-sub">
                Readers asked: &ldquo;{context.topic}&rdquo;. Once this document is indexed, the question is asked again
                so you can see it answered.
              </p>
            )}
          </div>
          <button
            type="button"
            className="btn btn-ghost btn-icon dialog-close"
            onClick={onClose}
            disabled={busy}
            aria-label="Close"
          >
            <Icon name="close" size={18} />
          </button>
        </div>

        <div className="dialog-body">
          {/* A real button, so it is reachable with Tab and activated with
              Enter or Space. */}
          <button
            type="button"
            className={`dropzone${dragOver ? " is-dragging" : ""}`}
            disabled={busy}
            onClick={() => input.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              const file = e.dataTransfer.files?.[0];
              if (file && !busy) void handleFile(file);
            }}
            aria-describedby={`${titleId}-hint`}
            data-autofocus
          >
            {busy ? (
              <>
                <span className="spinner" aria-hidden="true" />
                <span className="dropzone-title">{status.stage}</span>
              </>
            ) : (
              <>
                <Icon name="upload" size={24} />
                <span className="dropzone-title">Choose a file, or drop it here</span>
                <span id={`${titleId}-hint`} className="dropzone-hint">
                  PDF, Markdown or text · up to 1 MB
                </span>
              </>
            )}
          </button>
          <input
            ref={input}
            type="file"
            accept={ACCEPT}
            hidden
            tabIndex={-1}
            onChange={(e) => {
              const file = e.target.files?.[0];
              e.target.value = "";
              if (file) void handleFile(file);
            }}
          />

          <fieldset className="access-fields" disabled={busy}>
            <legend className="field-label">Who can read it</legend>
            <label className="field">
              <span className="field-label">Classification</span>
              <select
                className="select"
                value={classification}
                onChange={(e) => setClassification(e.target.value as ClassificationLevel)}
              >
                {CLASSIFICATION_LEVELS.map((l) => (
                  <option key={l} value={l}>
                    {l.charAt(0) + l.slice(1).toLowerCase()} — {LEVEL_HELP[l]}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              <span className="field-label">Share with roles (optional, comma separated)</span>
              <input
                className="input"
                value={roles}
                placeholder="e.g. employee, finance — leave empty for only you and admins"
                onChange={(e) => setRoles(e.target.value)}
              />
            </label>
          </fieldset>

          {/* Progress is announced, not just drawn. */}
          <p className="sr-only" aria-live="polite">
            {busy ? status.stage : ""}
          </p>
          {status.kind === "error" && (
            <div className="alert" role="alert">
              <Icon name="alert" />
              <span>{status.msg}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

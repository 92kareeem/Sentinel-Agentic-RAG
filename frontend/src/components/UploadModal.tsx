import { useId, useRef, useState } from "react";
import { ApiError, uploadDocument } from "../api";
import { useDialog } from "../lib/useDialog";
import { DOCUMENT_ERROR_HELP } from "../types";
import { Icon } from "./Icon";

interface Props {
  onClose: () => void;
  onUploaded: (docId: string) => void;
}

type Status = { kind: "idle" } | { kind: "busy"; stage: string } | { kind: "error"; msg: string };

// What the backend accepts. The hint text and the file picker both read from
// here, so they cannot disagree with each other again.
const ACCEPT = ".pdf,.md,.txt";

// Success closes the dialog and hands the new id back to App, which polls
// the document list until processing finishes — the backend has no push
// channel for lifecycle changes.
export function UploadModal({ onClose, onUploaded }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [status, setStatus] = useState<Status>({ kind: "idle" });
  const [dragOver, setDragOver] = useState(false);
  const busy = status.kind === "busy";
  // A dialog that is mid-upload should not vanish on Escape or a stray click
  // outside: the upload would carry on with nothing on screen to say so.
  const ref = useDialog<HTMLDivElement>(true, () => {
    if (!busy) onClose();
  });
  const titleId = useId();

  const handleFile = async (file: File) => {
    try {
      const result = await uploadDocument(file, (stage) => setStatus({ kind: "busy", stage }));
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
          <h2 id={titleId} className="dialog-title">
            Upload a document
          </h2>
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
              Enter or Space. It was a clickable <div>, which made uploading
              impossible without a mouse. */}
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

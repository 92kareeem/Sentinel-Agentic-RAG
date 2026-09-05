import { useRef, useState } from "react";
import { ApiError, uploadDocument } from "../api";
import { DOCUMENT_ERROR_HELP } from "../types";

interface Props {
  onClose: () => void;
  onUploaded: (docId: string) => void;
}

type Status =
  | { kind: "idle" }
  | { kind: "busy"; stage: string }
  | { kind: "error"; msg: string };

// Modal dropzone replacing the old inline "+ Upload a document" button.
// Success closes the modal immediately and hands the new doc_id back to
// App, which polls GET /v1/documents/{id} until status leaves PROCESSING —
// there is no push/streaming channel from the backend for this.
export function UploadModal({ onClose, onUploaded }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [status, setStatus] = useState<Status>({ kind: "idle" });
  const [dragOver, setDragOver] = useState(false);

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

  const busy = status.kind === "busy";

  return (
    <div className="modal-scrim" onClick={onClose}>
      <div className="modal" role="dialog" aria-modal="true" aria-label="Upload a document" onClick={(e) => e.stopPropagation()}>
        <header className="modal-header">
          <h3>Upload a document</h3>
          <button className="drawer-close" onClick={onClose} aria-label="Close">×</button>
        </header>

        <div
          className={`dropzone ${dragOver ? "drag" : ""} ${busy ? "busy" : ""}`}
          onClick={() => !busy && input.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            const file = e.dataTransfer.files?.[0];
            if (file && !busy) handleFile(file);
          }}
        >
          <input
            ref={input}
            type="file"
            accept=".pdf,.md,.txt"
            hidden
            onChange={(e) => {
              const file = e.target.files?.[0];
              e.target.value = "";
              if (file) handleFile(file);
            }}
          />
          {busy ? (
            <>
              <span className="spinner" aria-hidden />
              <div className="dropzone-title">{status.stage}</div>
            </>
          ) : (
            <>
              <div className="dropzone-icon" aria-hidden>↑</div>
              <div className="dropzone-title">Drop a file here or click to browse</div>
              <div className="dropzone-hint">PDF, Markdown, or text · max 1 MB</div>
            </>
          )}
        </div>

        {status.kind === "error" && <div className="modal-error">{status.msg}</div>}
      </div>
    </div>
  );
}

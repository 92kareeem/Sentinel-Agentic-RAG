import type { DocumentSummary } from "../types";

interface Props {
  documents: DocumentSummary[];
  activeDocId: string | null;
  onSelect: (docId: string | null) => void;
  onUploadClick: () => void;
  onDelete: (docId: string) => void;
  open: boolean;
  onCloseMobile: () => void;
}

const STATUS_LABEL: Record<string, string> = {
  UPLOADING: "Uploading…",
  UPLOADED: "Uploaded",
  PROCESSING: "Processing…",
  INDEXED: "Indexed",
  FAILED: "Failed",
  DELETED: "Deleted",
};

function StatusDot({ status }: { status: string }) {
  const cls =
    status === "INDEXED" ? "ok" : status === "FAILED" ? "bad" : status === "DELETED" ? "muted" : "busy";
  return <span className={`doc-status-dot ${cls}`} aria-hidden />;
}

export function Sidebar({ documents, activeDocId, onSelect, onUploadClick, onDelete, open, onCloseMobile }: Props) {
  const visible = documents.filter((d) => d.status !== "DELETED");

  return (
    <>
      {open && <div className="sidebar-scrim" onClick={onCloseMobile} />}
      <nav className={`sidebar ${open ? "open" : ""}`} aria-label="Documents">
        <div className="sidebar-header">
          <span className="sidebar-title">Documents</span>
          <button className="mobile-close" onClick={onCloseMobile} aria-label="Close sidebar">×</button>
        </div>

        <button className="upload-cta" onClick={onUploadClick}>
          + Upload document
        </button>

        <button
          className={`doc-item scope-all ${activeDocId === null ? "active" : ""}`}
          onClick={() => onSelect(null)}
        >
          <span className="doc-icon" aria-hidden>◎</span>
          <span className="doc-info">
            <span className="doc-name">All documents</span>
            <span className="doc-sub">Search everything you've uploaded</span>
          </span>
        </button>

        {visible.length === 0 && (
          <div className="sidebar-empty">No documents yet. Upload one to get started.</div>
        )}

        <div className="doc-list">
          {visible.map((doc) => (
            <div key={doc.document_id} className={`doc-item ${activeDocId === doc.document_id ? "active" : ""}`}>
              <button className="doc-item-main" onClick={() => onSelect(doc.document_id)}>
                <span className="doc-icon" aria-hidden>📄</span>
                <span className="doc-info">
                  <span className="doc-name" title={doc.filename}>{doc.filename}</span>
                  <span className="doc-sub">
                    <StatusDot status={doc.status} />
                    {STATUS_LABEL[doc.status] ?? doc.status}
                    {doc.status === "INDEXED" && doc.page_count ? ` · ${doc.page_count} page${doc.page_count === 1 ? "" : "s"}` : ""}
                  </span>
                  {doc.status === "FAILED" && doc.error_message && (
                    <span className="doc-error">{doc.error_message}</span>
                  )}
                </span>
              </button>
              <button
                className="doc-delete"
                onClick={() => onDelete(doc.document_id)}
                aria-label={`Delete ${doc.filename}`}
                title="Delete document"
              >
                🗑
              </button>
            </div>
          ))}
        </div>
      </nav>
    </>
  );
}

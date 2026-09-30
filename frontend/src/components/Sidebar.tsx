import { COMPACT_QUERY, useDialog, useMediaQuery } from "../lib/useDialog";
import type { DocumentSummary } from "../types";
import { Icon } from "./Icon";

interface Props {
  documents: DocumentSummary[];
  activeDocId: string | null;
  onSelect: (docId: string | null) => void;
  onUploadClick: () => void;
  onDelete: (doc: DocumentSummary) => void;
  open: boolean;
  onCloseMobile: () => void;
}

const STATUS_LABEL: Record<string, string> = {
  UPLOADING: "Uploading…",
  UPLOADED: "Uploaded",
  PROCESSING: "Processing…",
  INDEXED: "Ready",
  FAILED: "Failed",
  DELETED: "Deleted",
};

function statusTone(status: string): string {
  if (status === "INDEXED") return "ok";
  if (status === "FAILED") return "bad";
  return "busy";
}

export function Sidebar({ documents, activeDocId, onSelect, onUploadClick, onDelete, open, onCloseMobile }: Props) {
  const compact = useMediaQuery(COMPACT_QUERY);
  // On a phone the sidebar covers the page, so it has to behave like any
  // other overlay: focus in, Tab stays inside, Escape closes, focus back to
  // the menu button. On a wide screen it is just part of the page.
  const ref = useDialog<HTMLElement>(compact && open, onCloseMobile);
  const visible = documents.filter((d) => d.status !== "DELETED");

  return (
    <>
      {compact && open && <div className="sidebar-scrim" onClick={onCloseMobile} aria-hidden="true" />}
      <nav
        ref={ref}
        id="documents-panel"
        className={`sidebar${open ? " is-open" : ""}`}
        aria-labelledby="documents-heading"
        {...(compact && open ? { role: "dialog", "aria-modal": true } : {})}
      >
        <div className="sidebar-head">
          <h2 id="documents-heading" className="sidebar-title">
            Documents{visible.length > 0 ? ` · ${visible.length}` : ""}
          </h2>
          <button
            type="button"
            className="btn btn-ghost btn-icon btn-sm compact-only"
            onClick={onCloseMobile}
            aria-label="Close documents"
          >
            <Icon name="close" />
          </button>
        </div>

        <button type="button" className="nav-row nav-upload" onClick={onUploadClick}>
          <Icon name="plus" />
          <span>Upload document</span>
        </button>

        <button
          type="button"
          className="nav-row"
          onClick={() => onSelect(null)}
          aria-current={activeDocId === null ? "true" : undefined}
        >
          <Icon name="layers" />
          <span className="nav-text">
            <span className="nav-title">All documents</span>
            <span className="nav-sub">Search everything at once</span>
          </span>
        </button>

        {visible.length === 0 ? (
          <p className="sidebar-note">No documents yet. Upload one to get started.</p>
        ) : (
          <ul className="doc-list">
            {visible.map((doc) => {
              const status = STATUS_LABEL[doc.status] ?? doc.status;
              const pages =
                doc.status === "INDEXED" && doc.page_count
                  ? ` · ${doc.page_count} page${doc.page_count === 1 ? "" : "s"}`
                  : "";
              return (
                <li key={doc.document_id} className="doc-row">
                  <button
                    type="button"
                    className="nav-row"
                    onClick={() => onSelect(doc.document_id)}
                    aria-current={activeDocId === doc.document_id ? "true" : undefined}
                    title={doc.filename}
                  >
                    <Icon name="file" />
                    <span className="nav-text">
                      <span className="nav-title">{doc.filename}</span>
                      {/* Status is spelled out, never carried by the dot's
                          colour alone. */}
                      <span className="nav-sub">
                        <span className={`status-dot ${statusTone(doc.status)}`} aria-hidden="true" />
                        {status}
                        {pages}
                      </span>
                      {doc.status === "FAILED" && doc.error_message && (
                        <span className="nav-error">{doc.error_message}</span>
                      )}
                    </span>
                  </button>
                  <button
                    type="button"
                    className="btn btn-ghost btn-icon btn-sm btn-danger doc-delete"
                    onClick={() => onDelete(doc)}
                    aria-label={`Delete ${doc.filename}`}
                    title="Delete document"
                  >
                    <Icon name="trash" />
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </nav>
    </>
  );
}

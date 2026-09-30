import { useId } from "react";
import { useDialog } from "../lib/useDialog";
import type { Citation } from "../types";
import { Icon } from "./Icon";

interface Props {
  citation: Citation | null;
  onClose: () => void;
}

// The evidence behind one citation. Framed as "evidence" rather than raw
// source metadata; the chunk id is demoted to a labelled technical field.
export function SourceDrawer({ citation, onClose }: Props) {
  const ref = useDialog<HTMLElement>(citation !== null, onClose);
  const titleId = useId();

  if (!citation) return null;

  const docName = citation.source_filename || "Source document";
  const where = [citation.page_number != null ? `Page ${citation.page_number}` : null, citation.section_path]
    .filter(Boolean)
    .join(" · ");

  return (
    <>
      <div className="drawer-scrim" onClick={onClose} aria-hidden="true" />
      <aside ref={ref} className="drawer" role="dialog" aria-modal="true" aria-labelledby={titleId}>
        <div className="dialog-header">
          <div>
            <p className="eyebrow">Source</p>
            <h2 id={titleId} className="dialog-title">
              {docName}
            </h2>
            {where && <p className="drawer-path">{where}</p>}
          </div>
          <button
            type="button"
            className="btn btn-ghost btn-icon dialog-close"
            onClick={onClose}
            aria-label="Close source"
          >
            <Icon name="close" size={18} />
          </button>
        </div>
        <div className="dialog-body">
          <blockquote className="quote">{citation.quote}</blockquote>
          <p className="quote-note">
            <Icon name="check" size={14} />
            This is the passage the answer relies on
          </p>
          <details className="details">
            <summary>Technical details</summary>
            <dl className="kv">
              <dt>Chunk ID</dt>
              <dd className="mono">{citation.chunk_id}</dd>
              <dt>Document ID</dt>
              <dd className="mono">{citation.document_id}</dd>
            </dl>
          </details>
        </div>
      </aside>
    </>
  );
}

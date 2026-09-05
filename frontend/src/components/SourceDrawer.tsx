import { useEffect, useRef } from "react";
import type { Citation } from "../types";

interface Props {
  citation: Citation | null;
  onClose: () => void;
}

// Replaces CitationDrawer. Same data (page/filename come from the chunk's own
// metadata now, not a parsed chunk_id — see backend/app/models/schemas.py
// Citation), but framed as "evidence" rather than raw source metadata, and
// the chunk id is demoted to a labelled technical field instead of the
// footer of every card.
export function SourceDrawer({ citation, onClose }: Props) {
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!citation) return;
    closeRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [citation, onClose]);

  if (!citation) return null;

  const docName = citation.source_filename || "Source document";
  const page = citation.page_number != null ? `Page ${citation.page_number}` : null;

  return (
    <>
      <div className="drawer-scrim" onClick={onClose} />
      <aside className="drawer" role="dialog" aria-modal="true" aria-label="Source evidence">
        <header>
          <div>
            <div className="drawer-eyebrow">Source</div>
            <div className="drawer-doc">{docName}</div>
            <div className="drawer-path">
              {[page, citation.section_path].filter(Boolean).join(" · ")}
            </div>
          </div>
          <button ref={closeRef} className="drawer-close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </header>
        <blockquote className="drawer-quote">{citation.quote}</blockquote>
        <div className="drawer-verified">
          <span className="check-dot" aria-hidden />
          Supports this statement
        </div>
        <details className="drawer-advanced">
          <summary>Advanced</summary>
          <dl>
            <dt>Chunk ID</dt>
            <dd>{citation.chunk_id}</dd>
            <dt>Document ID</dt>
            <dd>{citation.document_id}</dd>
          </dl>
        </details>
      </aside>
    </>
  );
}

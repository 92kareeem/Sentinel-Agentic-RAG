import type { Citation } from "../types";

interface Props {
  citation: Citation | null;
  onClose: () => void;
}

export function CitationDrawer({ citation, onClose }: Props) {
  if (!citation) return null;

  // Show real provenance. This previously derived a "document name" from the
  // chunk_id prefix, which is now a server-generated uuid and means nothing to
  // a reader — the filename and page come from the chunk's own metadata.
  const docName = citation.source_filename || citation.document_id || "Source document";
  const page = citation.page_number != null ? `Page ${citation.page_number}` : null;

  return (
    <aside className="drawer" role="complementary" aria-label="Citation source">
      <header>
        <div>
          <div className="drawer-doc">{docName}</div>
          <div className="drawer-path">
            {citation.section_path}
            {page && <span className="drawer-page"> · {page}</span>}
          </div>
        </div>
        <button className="drawer-close" onClick={onClose} aria-label="Close citation">
          ×
        </button>
      </header>
      <div className="drawer-summary">Source excerpt</div>
      <pre className="drawer-quote">{citation.quote}</pre>
      <footer className="drawer-id">Chunk ID: {citation.chunk_id}</footer>
    </aside>
  );
}

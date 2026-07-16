import type { Citation } from "../types";

interface Props {
  citation: Citation | null;
  onClose: () => void;
}

export function CitationDrawer({ citation, onClose }: Props) {
  if (!citation) return null;
  const docName = citation.chunk_id.split("_")[0];
  return (
    <aside className="drawer">
      <header>
        <div>
          <div className="drawer-doc">{docName}</div>
          <div className="drawer-path">{citation.section_path}</div>
        </div>
        <button className="drawer-close" onClick={onClose}>
          ×
        </button>
      </header>
      <pre className="drawer-quote">{citation.quote}</pre>
      <footer className="drawer-id">{citation.chunk_id}</footer>
    </aside>
  );
}

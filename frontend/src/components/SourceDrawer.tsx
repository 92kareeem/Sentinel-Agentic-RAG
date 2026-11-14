import { useEffect, useId, useState } from "react";
import { getChunk } from "../api";
import { type Verification, verifyChunk } from "../lib/provenance";
import { useDialog } from "../lib/useDialog";
import type { ChunkDetail, Citation } from "../types";
import { Icon } from "./Icon";

interface Props {
  citation: Citation | null;
  onClose: () => void;
}

type Proof =
  | { kind: "loading" }
  | { kind: "ready"; chunk: ChunkDetail; result: Verification }
  | { kind: "unavailable"; reason: string };

// The evidence behind one citation — and proof that it is what it claims.
//
// The drawer fetches the full passage, recomputes its provenance hash in the
// browser with the recipe the server publishes, and says plainly whether the
// quote is byte-for-byte the span of the file that was indexed. "Trust us"
// becomes "check it".
export function SourceDrawer({ citation, onClose }: Props) {
  const ref = useDialog<HTMLElement>(citation !== null, onClose);
  const titleId = useId();
  const [proof, setProof] = useState<Proof>({ kind: "loading" });

  useEffect(() => {
    if (!citation) return;
    let live = true;
    setProof({ kind: "loading" });
    getChunk(citation.document_id, citation.chunk_id)
      .then(async (chunk) => {
        const result = await verifyChunk(chunk);
        if (live) setProof({ kind: "ready", chunk, result });
      })
      .catch(() => live && setProof({ kind: "unavailable", reason: "The full passage could not be loaded." }));
    return () => {
      live = false;
    };
  }, [citation]);

  if (!citation) return null;

  const docName = citation.source_filename || "Source document";
  const where = [citation.page_number != null ? `Page ${citation.page_number}` : null, citation.section_path]
    .filter(Boolean)
    .join(" · ");
  const passage = proof.kind === "ready" ? proof.chunk.text : citation.quote;

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
          <button type="button" className="btn btn-ghost btn-icon dialog-close" onClick={onClose} aria-label="Close source">
            <Icon name="close" size={18} />
          </button>
        </div>
        <div className="dialog-body">
          <blockquote className="quote">{passage}</blockquote>
          <ProofBadge proof={proof} />
          <details className="details">
            <summary>Provenance details</summary>
            <dl className="kv">
              <dt>Chunk ID</dt>
              <dd className="mono">{citation.chunk_id}</dd>
              <dt>Document ID</dt>
              <dd className="mono">{citation.document_id}</dd>
              {proof.kind === "ready" && (
                <>
                  <dt>Characters</dt>
                  <dd>
                    {proof.chunk.char_start}–{proof.chunk.char_end}
                  </dd>
                  <dt>Classification</dt>
                  <dd>{proof.chunk.classification}</dd>
                  <dt>File SHA-256</dt>
                  <dd className="mono">{proof.chunk.document_checksum_sha256 || "—"}</dd>
                  <dt>Passage hash</dt>
                  <dd className="mono">{proof.chunk.provenance_hash || "—"}</dd>
                  <dt>Recipe</dt>
                  <dd className="mono">{proof.chunk.hash_recipe}</dd>
                </>
              )}
            </dl>
          </details>
        </div>
      </aside>
    </>
  );
}

function ProofBadge({ proof }: { proof: Proof }) {
  if (proof.kind === "loading") {
    return (
      <p className="quote-note muted" role="status">
        <span className="spinner" aria-hidden="true" /> Verifying provenance…
      </p>
    );
  }
  if (proof.kind === "unavailable") {
    return (
      <p className="quote-note muted">
        <Icon name="alert" size={14} />
        {proof.reason}
      </p>
    );
  }
  const r = proof.result;
  if (r.state === "verified") {
    return (
      <p className="quote-note" role="status">
        <Icon name="shield" size={14} />
        Verified in your browser: this is the exact text at characters {proof.chunk.char_start}–{proof.chunk.char_end} of
        the indexed file.
      </p>
    );
  }
  if (r.state === "mismatch") {
    return (
      <p className="quote-note bad" role="alert">
        <Icon name="alert" size={14} />
        Provenance mismatch: this passage does not hash to what was indexed. Do not rely on it.
      </p>
    );
  }
  return (
    <p className="quote-note muted">
      <Icon name="alert" size={14} />
      {r.reason}
    </p>
  );
}

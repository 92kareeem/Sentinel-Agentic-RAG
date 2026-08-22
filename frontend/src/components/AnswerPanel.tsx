import type { Citation } from "../types";

interface Props {
  answer: string;
  citations: Citation[];
  onChipClick: (citation: Citation) => void;
  isStreaming?: boolean;
}

// Renders answer text with [chunk:id] tags replaced by numbered chips.
export function AnswerPanel({ answer, citations, onChipClick, isStreaming = false }: Props) {
  const order = new Map<string, number>();
  citations.forEach((c, i) => order.set(c.chunk_id, i + 1));
  const byId = new Map(citations.map((c) => [c.chunk_id, c]));

  const parts = answer.split(/\[chunk:([\w-]+)\]/g);
  const previewText = answer.trim() || (isStreaming ? "Generating answer…" : "No answer available yet.");

  return (
    <section className="answer">
      <div className="answer-header">
        <strong>Answer</strong>
        <span className={`answer-status ${isStreaming ? "streaming" : "ready"}`}>
          {isStreaming ? "Streaming response…" : "Cited answer"}
        </span>
      </div>
      <div className="answer-text">
        {previewText === "Generating answer…" ? (
          <span className="answer-placeholder">{previewText}</span>
        ) : (
          parts.map((part, i) => {
            if (i % 2 === 0) return <span key={i}>{part}</span>;
            const citation = byId.get(part);
            if (!citation) return null;
            return (
              <button key={i} className="chip" onClick={() => onChipClick(citation)}>
                {order.get(part) ?? "?"}
              </button>
            );
          })
        )}
      </div>
      {citations.length > 0 && (
        <div className="answer-footer">
          <span>Click any citation chip to inspect the supporting excerpt.</span>
        </div>
      )}
    </section>
  );
}

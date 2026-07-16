import type { Citation } from "../types";

interface Props {
  answer: string;
  citations: Citation[];
  onChipClick: (citation: Citation) => void;
}

// Renders answer text with [chunk:id] tags replaced by numbered chips.
export function AnswerPanel({ answer, citations, onChipClick }: Props) {
  const order = new Map<string, number>();
  citations.forEach((c, i) => order.set(c.chunk_id, i + 1));
  const byId = new Map(citations.map((c) => [c.chunk_id, c]));

  const parts = answer.split(/\[chunk:([\w-]+)\]/g);
  // parts alternate: text, chunk_id, text, chunk_id, ...
  return (
    <section className="answer">
      {parts.map((part, i) => {
        if (i % 2 === 0) return <span key={i}>{part}</span>;
        const citation = byId.get(part);
        if (!citation) return null;
        return (
          <button key={i} className="chip" onClick={() => onChipClick(citation)}>
            {order.get(part) ?? "?"}
          </button>
        );
      })}
    </section>
  );
}

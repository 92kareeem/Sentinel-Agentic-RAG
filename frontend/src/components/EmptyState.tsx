interface Props {
  hasDocuments: boolean;
  onExample: (q: string) => void;
}

// Generic examples only — the brief explicitly warns against fabricating
// example questions tied to document content the backend hasn't told us
// about (section 15). These are safe regardless of what's uploaded.
const EXAMPLES = [
  "What is this document about?",
  "Summarize the key points",
  "What are the main policies or requirements described here?",
];

const STEPS = [
  { n: "1", title: "Upload", body: "Add a PDF, Word doc, or text file." },
  { n: "2", title: "Ask", body: "Ask a question in plain language." },
  { n: "3", title: "Verify", body: "Get an answer with page-level citations." },
];

const TRUST_BADGES = ["Every sentence cited", "Grounded in your documents", "Refuses to guess"];

export function EmptyState({ hasDocuments, onExample }: Props) {
  return (
    <div className="empty-state">
      <div className="empty-mark">S</div>
      <h2>Ask questions about your documents</h2>
      <p className="empty-sub">
        Sentinel reads your documents and answers only from what's actually written in them —
        every claim traced back to a page, so you can trust the answer without re-reading the source.
      </p>

      <div className="trust-row">
        {TRUST_BADGES.map((b) => (
          <span key={b} className="trust-badge">
            <span className="check-dot" aria-hidden /> {b}
          </span>
        ))}
      </div>

      <div className="steps-row">
        {STEPS.map((s) => (
          <div key={s.n} className="step-card">
            <div className="step-num">{s.n}</div>
            <div className="step-title">{s.title}</div>
            <div className="step-body">{s.body}</div>
          </div>
        ))}
      </div>

      {!hasDocuments && (
        <div className="empty-hint">Upload a document from the sidebar to get started.</div>
      )}

      {hasDocuments && (
        <div className="empty-examples">
          <div className="empty-examples-label">Try asking</div>
          {EXAMPLES.map((q) => (
            <button key={q} className="example-chip" onClick={() => onExample(q)}>
              {q}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

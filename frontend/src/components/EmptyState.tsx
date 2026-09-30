import { Icon } from "./Icon";

interface Props {
  hasDocuments: boolean;
  onExample: (q: string) => void;
}

// Generic on purpose: example questions must not assume content the backend
// has not told us about. These are true of any uploaded document.
const EXAMPLES = [
  "What is this document about?",
  "Summarize the key points",
  "What are the main policies or requirements described here?",
];

// Must match what UploadModal actually accepts (.pdf, .md, .txt). This used
// to promise "Word doc", which the uploader rejects.
const STEPS = [
  { title: "Upload", body: "Add a PDF, Markdown or text file." },
  { title: "Ask", body: "Ask a question in plain language." },
  { title: "Verify", body: "Every answer cites the passage it came from." },
];

const TRUST = ["Every sentence cited", "Grounded in your documents", "Refuses to guess"];

export function EmptyState({ hasDocuments, onExample }: Props) {
  return (
    <div className="empty">
      <div className="empty-mark" aria-hidden="true">
        S
      </div>
      <h2 className="empty-title">Ask questions about your documents</h2>
      <p className="empty-lead">
        Sentinel answers only from what is actually written in your documents, and traces every
        claim back to its source — so you can trust the answer without re-reading the document.
      </p>

      <ul className="pill-row" aria-label="How answers are checked">
        {TRUST.map((t) => (
          <li key={t} className="pill">
            <Icon name="check" size={14} />
            {t}
          </li>
        ))}
      </ul>

      <ol className="steps" aria-label="How it works">
        {STEPS.map((s, i) => (
          <li key={s.title} className="step">
            <span className="step-num" aria-hidden="true">
              {i + 1}
            </span>
            <h3 className="step-title">{s.title}</h3>
            <p className="step-body">{s.body}</p>
          </li>
        ))}
      </ol>

      {hasDocuments ? (
        <section className="examples" aria-labelledby="examples-heading">
          <h3 id="examples-heading" className="section-label">
            Try asking
          </h3>
          <ul className="example-list">
            {EXAMPLES.map((q) => (
              <li key={q}>
                <button type="button" className="example" onClick={() => onExample(q)}>
                  {q}
                </button>
              </li>
            ))}
          </ul>
        </section>
      ) : (
        <p className="empty-hint">Upload a document from the documents panel to get started.</p>
      )}
    </div>
  );
}

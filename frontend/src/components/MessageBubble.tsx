import { renderAnswerMarkdown } from "../lib/markdown";
import type { ChatMessage, Citation, TraceRecord } from "../types";
import { REFUSAL_COPY } from "../types";
import { FeedbackBar } from "./FeedbackBar";
import { Icon } from "./Icon";
import { ResponseDetails } from "./ResponseDetails";

interface Props {
  message: ChatMessage;
  onCiteClick: (citation: Citation) => void;
  trace: TraceRecord | null;
}

function Verified({ faithfulness, sourceCount }: { faithfulness: number; sourceCount: number }) {
  // Plain language rather than a raw score. 0.7 is the same threshold the
  // old score dial coloured at — the one real constant, not a new scale.
  const grounded = faithfulness >= 0.7;
  return (
    <span className={`verified${grounded ? "" : " low"}`}>
      <Icon name={grounded ? "check" : "alert"} size={14} />
      {grounded
        ? `Grounded in ${sourceCount} source${sourceCount === 1 ? "" : "s"}`
        : "Low confidence — check the sources"}
    </span>
  );
}

export function MessageBubble({ message, onCiteClick, trace }: Props) {
  if (message.role === "user") {
    return (
      <div className="msg-user">
        <span className="sr-only">You asked: </span>
        {message.text}
      </div>
    );
  }

  if (message.kind === "pending") {
    return (
      <div className="card card-pending" role="status">
        <span className="typing-dot" aria-hidden="true" />
        <span className="typing-dot" aria-hidden="true" />
        <span className="typing-dot" aria-hidden="true" />
        <span className="sr-only">Sentinel is checking your documents…</span>
      </div>
    );
  }

  if (message.kind === "error") {
    return (
      <div className="card card-error" role="alert">
        <Icon name="alert" />
        <p>{message.message}</p>
      </div>
    );
  }

  if (message.kind === "refusal") {
    // Driven by the structured reason code, never by matching prose: each
    // reason asks the reader to do something different.
    const copy = REFUSAL_COPY[message.reasonCode] ?? REFUSAL_COPY.INSUFFICIENT_EVIDENCE;
    return (
      <article className="card card-refusal" aria-label="Sentinel could not answer">
        <h3 className="refusal-title">
          <Icon name="alert" />
          {copy.title}
        </h3>
        <p>{copy.body}</p>
        <p className="refusal-hint">{copy.hint}</p>
        <div className="answer-footer">
          <FeedbackBar traceId={message.traceId} variant="refusal" />
        </div>
      </article>
    );
  }

  const byId = new Map(message.citations.map((c) => [c.chunk_id, c]));

  return (
    <article className="card" aria-label="Sentinel's answer">
      <div className="answer-body">
        {renderAnswerMarkdown(message.text, message.citations, (id) => {
          const c = byId.get(id);
          if (c) onCiteClick(c);
        })}
      </div>

      {message.citations.length > 0 && (
        <section className="answer-section" aria-label="Sources">
          <h4 className="section-label" aria-hidden="true">
            Sources
          </h4>
          <ol className="source-list">
            {message.citations.map((c, i) => (
              <li key={c.chunk_id}>
                <button type="button" className="source-card" onClick={() => onCiteClick(c)}>
                  <span className="source-index" aria-hidden="true">
                    {i + 1}
                  </span>
                  <span className="source-meta">
                    <span className="source-name">{c.source_filename || "Document"}</span>
                    <span className="source-sub">
                      {[c.page_number != null ? `Page ${c.page_number}` : null, c.section_path]
                        .filter(Boolean)
                        .join(" · ")}
                    </span>
                  </span>
                  <span className="sr-only">, open source {i + 1}</span>
                </button>
              </li>
            ))}
          </ol>
        </section>
      )}

      <div className="answer-footer">
        <Verified faithfulness={message.critic.faithfulness} sourceCount={message.citations.length} />
        <FeedbackBar traceId={message.traceId} variant="answer" />
      </div>

      <ResponseDetails
        traceId={message.traceId}
        critic={message.critic}
        repairCount={message.repairCount}
        model={message.model}
        latencyMs={message.latencyMs}
        sourceCount={message.citations.length}
        trace={trace}
      />
    </article>
  );
}

import { renderAnswerMarkdown } from "../lib/markdown";
import type { ChatMessage, Citation, TraceRecord } from "../types";
import { REFUSAL_COPY } from "../types";
import { ResponseDetails } from "./ResponseDetails";

interface Props {
  message: ChatMessage;
  onCiteClick: (citation: Citation) => void;
  trace: TraceRecord | null;
}

function VerifiedBadge({ faithfulness, sourceCount }: { faithfulness: number; sourceCount: number }) {
  // Human language, not raw scores — see brief section 8. 0.7 mirrors the
  // threshold ScoreBadge used to color its dial; kept as the one real
  // constant rather than inventing a new confidence scale.
  const grounded = faithfulness >= 0.7;
  return (
    <div className={`verified-badge ${grounded ? "ok" : "low"}`}>
      <span className="check-dot" aria-hidden />
      {grounded
        ? `Grounded in ${sourceCount} source${sourceCount === 1 ? "" : "s"}`
        : "Low-confidence answer — verify against the sources"}
    </div>
  );
}

export function MessageBubble({ message, onCiteClick, trace }: Props) {
  if (message.role === "user") {
    return (
      <div className="msg msg-user">
        <div className="bubble bubble-user">{message.text}</div>
      </div>
    );
  }

  if (message.kind === "pending") {
    return (
      <div className="msg msg-assistant">
        <div className="bubble bubble-assistant bubble-pending">
          <span className="typing-dot" />
          <span className="typing-dot" />
          <span className="typing-dot" />
        </div>
      </div>
    );
  }

  if (message.kind === "error") {
    return (
      <div className="msg msg-assistant">
        <div className="bubble bubble-assistant bubble-error">{message.message}</div>
      </div>
    );
  }

  if (message.kind === "refusal") {
    // Driven by the backend's structured reason_code rather than by matching
    // on prose. The three cases need different user actions: rephrase/upload,
    // ask more specifically, or simply retry.
    const copy = REFUSAL_COPY[message.reasonCode] ?? REFUSAL_COPY.INSUFFICIENT_EVIDENCE;
    return (
      <div className="msg msg-assistant">
        <div className="bubble bubble-assistant refusal-card">
          <div className="refusal-title">
            <span aria-hidden>✦</span> {copy.title}
          </div>
          <p>{copy.body}</p>
          <p className="refusal-hint">{copy.hint}</p>
        </div>
      </div>
    );
  }

  const byId = new Map(message.citations.map((c) => [c.chunk_id, c]));

  return (
    <div className="msg msg-assistant">
      <div className="bubble bubble-assistant">
        <div className="answer-body">
          {renderAnswerMarkdown(message.text, message.citations, (id) => {
            const c = byId.get(id);
            if (c) onCiteClick(c);
          })}
        </div>

        {message.citations.length > 0 && (
          <div className="sources-block">
            <div className="sources-title">Sources</div>
            <div className="sources-list">
              {message.citations.map((c, i) => (
                <button key={c.chunk_id} className="source-card" onClick={() => onCiteClick(c)}>
                  <span className="source-index">{i + 1}</span>
                  <span className="source-meta">
                    <span className="source-name">{c.source_filename || "Document"}</span>
                    <span className="source-sub">
                      {[c.page_number != null ? `Page ${c.page_number}` : null, c.section_path]
                        .filter(Boolean)
                        .join(" · ")}
                    </span>
                  </span>
                </button>
              ))}
            </div>
          </div>
        )}

        <VerifiedBadge faithfulness={message.critic.faithfulness} sourceCount={message.citations.length} />

        <ResponseDetails
          traceId={message.traceId}
          critic={message.critic}
          repairCount={message.repairCount}
          model={message.model}
          latencyMs={message.latencyMs}
          sourceCount={message.citations.length}
          trace={trace}
        />
      </div>
    </div>
  );
}

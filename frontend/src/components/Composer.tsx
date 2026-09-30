import { useRef, useState } from "react";
import { Icon } from "./Icon";

const MAX_LEN = 1000;
// The count only appears once it is worth reading. "0 / 1000" under an empty
// box was noise on every screen.
const SHOW_COUNT_AT = 0.8 * MAX_LEN;

interface Props {
  loading: boolean;
  onSubmit: (query: string) => void;
  scopedTo: string | null; // a filename, or null for all documents
  onClearScope: () => void;
}

export const COMPOSER_ID = "composer-input";

// Enter sends and Shift+Enter adds a line: the convention every chat product
// already teaches. It is also spelled out for screen readers (see the hint),
// since nothing on screen says it.
export function Composer({ loading, onSubmit, scopedTo, onClearScope }: Props) {
  const [text, setText] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);

  const submit = () => {
    const q = text.trim();
    if (!q || loading) return;
    onSubmit(q);
    setText("");
    if (ref.current) ref.current.style.height = "auto";
  };

  const autosize = (el: HTMLTextAreaElement) => {
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  };

  const nearLimit = text.length >= SHOW_COUNT_AT;

  return (
    <div className="composer">
      <div className="composer-meta">
        {scopedTo ? (
          <span className="scope is-scoped">
            <Icon name="file" size={14} />
            <span className="scope-text">Answering from {scopedTo}</span>
            <button
              type="button"
              className="scope-clear"
              onClick={onClearScope}
              aria-label={`Stop limiting answers to ${scopedTo}`}
              title="Search all documents"
            >
              <Icon name="close" size={12} />
            </button>
          </span>
        ) : (
          <span className="scope">
            <Icon name="layers" size={14} />
            <span className="scope-text">Searching all documents</span>
          </span>
        )}
        {nearLimit && (
          <span className={`char-count${text.length >= MAX_LEN ? " near-limit" : ""}`} aria-live="polite">
            {MAX_LEN - text.length} characters left
          </span>
        )}
      </div>

      <div className="composer-box">
        <label htmlFor={COMPOSER_ID} className="sr-only">
          Ask a question about your documents
        </label>
        {/* Deliberately NOT disabled while an answer is loading: disabling it
            threw keyboard focus out of the box after every question. Typing
            ahead is fine; only sending waits. */}
        <textarea
          id={COMPOSER_ID}
          ref={ref}
          className="composer-input"
          value={text}
          onChange={(e) => {
            setText(e.target.value);
            autosize(e.target);
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              submit();
            }
          }}
          placeholder="Ask about your documents…"
          rows={1}
          maxLength={MAX_LEN}
          aria-describedby="composer-hint"
        />
        <span id="composer-hint" className="sr-only">
          Press Enter to send, Shift and Enter for a new line.
        </span>
        <button
          type="button"
          className="btn btn-primary btn-icon"
          onClick={submit}
          disabled={loading || !text.trim()}
          aria-label={loading ? "Waiting for the answer" : "Send question"}
        >
          {loading ? <span className="spinner" aria-hidden="true" /> : <Icon name="send" size={18} />}
        </button>
      </div>
    </div>
  );
}

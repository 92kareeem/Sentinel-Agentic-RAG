import { useRef, useState } from "react";

const MAX_LEN = 1000;

interface Props {
  loading: boolean;
  disabled?: boolean;
  onSubmit: (query: string) => void;
  scopeLabel: string;
}

// Primary input. Enter submits, Shift+Enter inserts a newline — the
// convention every chat product (ChatGPT, Gemini, Claude) already trains
// users on, so no on-screen instructions are needed for it.
export function Composer({ loading, disabled, onSubmit, scopeLabel }: Props) {
  const [text, setText] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);

  const submit = () => {
    const q = text.trim();
    if (!q || loading || disabled) return;
    onSubmit(q);
    setText("");
    if (ref.current) ref.current.style.height = "auto";
  };

  const autosize = (el: HTMLTextAreaElement) => {
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  };

  return (
    <div className="composer">
      <div className="composer-scope">{scopeLabel}</div>
      <div className="composer-box">
        <textarea
          ref={ref}
          value={text}
          onChange={(e) => {
            setText(e.target.value);
            autosize(e.target);
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
          placeholder="Ask Sentinel about your documents…"
          rows={1}
          maxLength={MAX_LEN}
          disabled={loading || disabled}
          aria-label="Ask a question"
        />
        <button
          className="composer-send"
          onClick={submit}
          disabled={loading || disabled || !text.trim()}
          aria-label="Send"
        >
          {loading ? <span className="spinner" aria-hidden /> : "↑"}
        </button>
      </div>
      <div className="composer-footer">
        <span>{text.length} / {MAX_LEN}</span>
      </div>
    </div>
  );
}

import { useState } from "react";

interface Props {
  loading: boolean;
  onSubmit: (query: string) => void;
  error: string | null;
}

export function QueryBox({ loading, onSubmit, error }: Props) {
  const [text, setText] = useState("");

  const submit = () => {
    const q = text.trim();
    if (q && !loading) onSubmit(q);
  };

  return (
    <section className="querybox">
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            submit();
          }
        }}
        placeholder="Ask the documents… e.g. What restocking fee applies in India?"
        rows={3}
        maxLength={1000}
        disabled={loading}
      />
      <div className="querybox-row">
        <span className="hint">{text.length}/1000 · Enter to ask</span>
        <button onClick={submit} disabled={loading || !text.trim()}>
          {loading ? "Thinking…" : "Ask Sentinel"}
        </button>
      </div>
      {error && <div className="error-banner">{error}</div>}
    </section>
  );
}

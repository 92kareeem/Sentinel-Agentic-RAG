import { useRef, useState } from "react";
import { ApiError, uploadDocument } from "../api";

interface Props {
  onIndexed: (filename: string, chunks: number) => void;
}

type Status =
  | { kind: "idle" }
  | { kind: "busy"; stage: string }
  | { kind: "done"; msg: string }
  | { kind: "error"; msg: string };

export function UploadBar({ onIndexed }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [status, setStatus] = useState<Status>({ kind: "idle" });

  const pick = () => input.current?.click();

  const onFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = ""; // allow re-selecting the same file
    if (!file) return;

    try {
      const result = await uploadDocument(file, (stage) => setStatus({ kind: "busy", stage }));
      setStatus({ kind: "done", msg: `Indexed ${result.chunks_indexed} chunks from ${file.name}` });
      onIndexed(file.name, result.chunks_indexed);
    } catch (err) {
      const msg =
        err instanceof ApiError ? `${err.status}: ${err.detail}` : "Upload failed — try again";
      setStatus({ kind: "error", msg });
    }
  };

  const busy = status.kind === "busy";
  return (
    <div className="uploadbar">
      <input
        ref={input}
        type="file"
        accept=".pdf,.md,.txt"
        onChange={onFile}
        hidden
      />
      <button className="upload-btn" onClick={pick} disabled={busy}>
        {busy ? status.stage : "＋ Upload a document"}
      </button>
      {status.kind === "done" && <span className="upload-done">✓ {status.msg}</span>}
      {status.kind === "error" && <span className="upload-err">{status.msg}</span>}
      <span className="upload-hint">.pdf / .md / .txt · max 1 MB</span>
    </div>
  );
}

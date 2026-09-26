import type { ReactNode } from "react";
import type { Citation } from "../types";

// Minimal markdown renderer for LLM answers: headings, bold/italic, bullet
// and numbered lists, pipe tables, paragraphs — plus inline [chunk:id]
// markers rewritten into numbered citation chips. No dependency: the answer
// space here is narrow enough (headings/lists/tables/bold/citations) that a
// full markdown library would be more surface area than the feature needs.

const CHUNK_RE = /\[chunk:([\w-]+)\]/g;

function renderInline(text: string, order: Map<string, number>, onCite: (id: string) => void): ReactNode[] {
  const boldSplit = text.split(/(\*\*[^*]+\*\*)/g);
  const nodes: ReactNode[] = [];
  boldSplit.forEach((seg, i) => {
    if (seg.startsWith("**") && seg.endsWith("**") && seg.length > 4) {
      nodes.push(<strong key={`b${i}`}>{seg.slice(2, -2)}</strong>);
      return;
    }
    const parts = seg.split(CHUNK_RE);
    parts.forEach((part, j) => {
      if (j % 2 === 0) {
        if (part) nodes.push(<span key={`t${i}-${j}`}>{part}</span>);
      } else {
        const n = order.get(part);
        if (n) {
          nodes.push(
            <button
              key={`c${i}-${j}`}
              type="button"
              className="citechip"
              onClick={() => onCite(part)}
              aria-label={`View source ${n}`}
            >
              {n}
            </button>,
          );
        }
      }
    });
  });
  return nodes;
}

function renderTable(lines: string[], key: number): ReactNode {
  const rows = lines
    .filter((l) => !/^\s*\|?\s*[-: |]+\s*\|?\s*$/.test(l))
    .map((l) => l.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim()));
  const [head, ...body] = rows;
  // Separator-only input ("| --- |" twice) filters down to nothing, and
  // head.map would then throw and blank the whole answer. Degrade to the raw
  // lines instead: an ugly table beats a lost answer.
  if (!head) {
    return <p key={key}>{lines.join(" ")}</p>;
  }
  return (
    <div className="md-table-wrap" key={key}>
      <table className="md-table">
        <thead>
          <tr>{head.map((c, i) => <th key={i}>{c}</th>)}</tr>
        </thead>
        <tbody>
          {body.map((r, i) => (
            <tr key={i}>{r.map((c, j) => <td key={j}>{c}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const TABLE_ROW_RE = /^\s*\|.*\|\s*$/;
const TABLE_SEP_RE = /^\s*\|?\s*[-: |]+\s*\|?\s*$/;
const BULLET_RE = /^\s*[-*]\s+/;
const ORDERED_RE = /^\s*\d+\.\s+/;
const HEADING_RE = /^(#{1,3})\s+(.*)$/;

// Does this line start a block that is NOT a paragraph? Used only to decide
// where a paragraph ENDS, never whether one begins — see the paragraph branch.
function startsBlock(line: string): boolean {
  return (
    TABLE_ROW_RE.test(line) ||
    BULLET_RE.test(line) ||
    ORDERED_RE.test(line) ||
    HEADING_RE.test(line)
  );
}

export function renderAnswerMarkdown(
  text: string,
  citations: Citation[],
  onCite: (id: string) => void,
): ReactNode {
  const order = new Map<string, number>();
  citations.forEach((c, i) => order.set(c.chunk_id, i + 1));

  const lines = text.split("\n");
  const blocks: ReactNode[] = [];
  let i = 0;
  let key = 0;

  while (i < lines.length) {
    const line = lines[i];

    if (TABLE_ROW_RE.test(line) && lines[i + 1] && TABLE_SEP_RE.test(lines[i + 1])) {
      const tableLines: string[] = [];
      while (i < lines.length && TABLE_ROW_RE.test(lines[i])) {
        tableLines.push(lines[i]);
        i += 1;
      }
      blocks.push(renderTable(tableLines, key++));
      continue;
    }

    const heading = HEADING_RE.exec(line);
    if (heading) {
      const level = heading[1].length;
      const content = renderInline(heading[2], order, onCite);
      if (level === 1) blocks.push(<h3 key={key++}>{content}</h3>);
      else if (level === 2) blocks.push(<h4 key={key++}>{content}</h4>);
      else blocks.push(<h5 key={key++}>{content}</h5>);
      i += 1;
      continue;
    }

    if (BULLET_RE.test(line)) {
      const items: string[] = [];
      while (i < lines.length && BULLET_RE.test(lines[i])) {
        items.push(lines[i].replace(BULLET_RE, ""));
        i += 1;
      }
      blocks.push(
        <ul key={key++}>
          {items.map((it, idx) => <li key={idx}>{renderInline(it, order, onCite)}</li>)}
        </ul>,
      );
      continue;
    }

    if (ORDERED_RE.test(line)) {
      const items: string[] = [];
      while (i < lines.length && ORDERED_RE.test(lines[i])) {
        items.push(lines[i].replace(ORDERED_RE, ""));
        i += 1;
      }
      blocks.push(
        <ol key={key++}>
          {items.map((it, idx) => <li key={idx}>{renderInline(it, order, onCite)}</li>)}
        </ol>,
      );
      continue;
    }

    if (line.trim() === "") {
      i += 1;
      continue;
    }

    // Paragraph: the fallback branch, and the only one that guarantees the
    // outer loop advances. It consumes the current line unconditionally
    // BEFORE testing the continuation guard.
    //
    // Taking the current line only if it passed the guard was an infinite
    // loop: the guard excludes table rows, but the table branch above needs a
    // separator row beneath the first line to fire. An orphan row — a single
    // "| Region | Refund window |" quoted out of a document table, which is
    // exactly what this product's answers contain — matched no branch, left
    // `i` unchanged, and froze the tab. A renderer fed model output about
    // user-supplied documents must not be able to hang on its input, so
    // progress is structural here rather than a property of the guards.
    const paraLines: string[] = [lines[i]];
    i += 1;
    while (i < lines.length && lines[i].trim() !== "" && !startsBlock(lines[i])) {
      paraLines.push(lines[i]);
      i += 1;
    }
    blocks.push(<p key={key++}>{renderInline(paraLines.join(" "), order, onCite)}</p>);
  }

  return blocks;
}

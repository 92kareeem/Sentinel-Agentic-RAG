import { useState } from "react";

// Two small single-series charts for the telemetry tab. Single series, so one
// hue (the accent), no legend — the heading names the measure — and every
// value is also available in a table for anyone who cannot use the chart.

export interface Bar {
  key: string;
  label: string; // axis label, e.g. "09:00"
  value: number;
  tooltip: string; // full sentence shown on hover
}

export function ColumnChart({ bars, title, unit }: { bars: Bar[]; title: string; unit: string }) {
  const [hover, setHover] = useState<Bar | null>(null);
  const max = Math.max(1, ...bars.map((b) => b.value));
  // Label every Nth bar so 24 hours never collide on a phone.
  const every = Math.max(1, Math.ceil(bars.length / 8));

  return (
    <figure className="chart">
      <figcaption className="chart-head">
        <span className="chart-title">{title}</span>
        <span className="chart-hover" aria-live="polite">
          {hover ? hover.tooltip : `max ${max.toLocaleString()} ${unit}`}
        </span>
      </figcaption>
      <div className="columns" onMouseLeave={() => setHover(null)}>
        {bars.map((b) => (
          <div
            key={b.key}
            className="column-hit"
            onMouseEnter={() => setHover(b)}
            aria-hidden="true"
          >
            <div
              className={`column${b.value === 0 ? " column-zero" : ""}${hover?.key === b.key ? " is-hover" : ""}`}
              style={{ height: `${(b.value / max) * 100}%` }}
            />
          </div>
        ))}
      </div>
      <div className="columns-axis" aria-hidden="true">
        {bars.map((b, i) => (
          <span key={b.key}>{i % every === 0 ? b.label : ""}</span>
        ))}
      </div>
      <details className="details">
        <summary>Show as table</summary>
        <table className="data-table">
          <thead>
            <tr>
              <th scope="col">Period</th>
              <th scope="col">{unit}</th>
            </tr>
          </thead>
          <tbody>
            {bars.map((b) => (
              <tr key={b.key}>
                <td>{b.label}</td>
                <td>{b.value.toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </figure>
  );
}

export interface Row {
  key: string;
  label: string;
  value: number;
  display: string; // formatted value shown at the end of the row
  sub?: string;
}

// Horizontal bars: names are long, values compared by length.
export function BarList({ rows, title }: { rows: Row[]; title: string }) {
  const max = Math.max(1, ...rows.map((r) => r.value));
  return (
    <figure className="chart">
      <figcaption className="chart-head">
        <span className="chart-title">{title}</span>
      </figcaption>
      <ul className="barlist">
        {rows.map((r) => (
          <li key={r.key}>
            <span className="barlist-label">
              {r.label}
              {r.sub && <span className="barlist-sub">{r.sub}</span>}
            </span>
            <span className="barlist-track" aria-hidden="true">
              <span className="barlist-fill" style={{ width: `${Math.max(2, (r.value / max) * 100)}%` }} />
            </span>
            <span className="barlist-value">{r.display}</span>
          </li>
        ))}
      </ul>
    </figure>
  );
}

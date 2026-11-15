// One headline number. Read as one phrase ("79% Questions answered, 19 of
// 24"), not three fragments. The coloured edge is decoration; the words and
// numbers carry the meaning.
export function Kpi({
  label,
  value,
  sub,
  tone,
}: {
  label: string;
  value: string;
  sub: string;
  tone: "good" | "warn" | "bad" | "neutral";
}) {
  return (
    <li className={`kpi kpi-${tone}`}>
      <span className="kpi-value">{value}</span>
      <span className="kpi-label">{label}</span>
      <span className="kpi-sub">{sub}</span>
    </li>
  );
}

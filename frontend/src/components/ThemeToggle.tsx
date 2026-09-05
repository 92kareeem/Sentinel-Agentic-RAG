import { useEffect, useState } from "react";

type Theme = "light" | "dark";

function getInitialTheme(): Theme {
  try {
    const saved = localStorage.getItem("sentinel-theme");
    if (saved === "light" || saved === "dark") return saved;
  } catch {
    // localStorage unavailable (private mode, etc.) — fall through to default.
  }
  return "dark";
}

// Manual light/dark toggle rather than only following prefers-color-scheme:
// this is a product decision (brief explicitly asked for a switch), and
// persists per-browser so returning stakeholders keep their choice.
export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(getInitialTheme);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    try {
      localStorage.setItem("sentinel-theme", theme);
    } catch {
      // Non-fatal — theme just won't persist across reloads.
    }
  }, [theme]);

  return (
    <button
      className="theme-toggle"
      onClick={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
      aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
      title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
    >
      {theme === "dark" ? "☀" : "☾"}
    </button>
  );
}

export function applyStoredTheme(): void {
  document.documentElement.setAttribute("data-theme", getInitialTheme());
}

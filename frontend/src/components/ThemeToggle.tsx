import { useEffect, useState } from "react";
import { Icon } from "./Icon";

type Theme = "light" | "dark";

function getInitialTheme(): Theme {
  try {
    const saved = localStorage.getItem("sentinel-theme");
    if (saved === "light" || saved === "dark") return saved;
  } catch {
    // localStorage unavailable (private mode, etc.) — fall through.
  }
  // No saved choice: follow the operating system rather than forcing dark on
  // someone whose whole machine is set to light.
  return typeof window !== "undefined" && window.matchMedia?.("(prefers-color-scheme: light)").matches
    ? "light"
    : "dark";
}

// Manual light/dark switch (a product decision — the brief asked for one),
// persisted per browser so returning users keep their choice.
export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(getInitialTheme);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    try {
      localStorage.setItem("sentinel-theme", theme);
    } catch {
      // Non-fatal — the theme just won't persist across reloads.
    }
  }, [theme]);

  const next = theme === "dark" ? "light" : "dark";
  return (
    <button
      type="button"
      className="btn btn-secondary btn-icon"
      onClick={() => setTheme(next)}
      aria-label={`Switch to ${next} theme`}
      title={`Switch to ${next} theme`}
    >
      <Icon name={theme === "dark" ? "sun" : "moon"} />
    </button>
  );
}

export function applyStoredTheme(): void {
  document.documentElement.setAttribute("data-theme", getInitialTheme());
}

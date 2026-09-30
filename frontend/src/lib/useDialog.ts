import { useEffect, useRef, useState } from "react";

// Keyboard and focus behaviour for anything that covers the page: the
// Coverage and Upload dialogs, the source drawer, and the sidebar on a phone.
//
// Each of those used to implement some of this and none implemented all of
// it — only the source drawer closed on Escape, none kept focus inside, and
// none put focus back where it came from — so a keyboard user could Tab
// straight out of a modal into the page dimmed behind it, and lose their
// place in the conversation every time they closed one. One hook, used by
// all four, so they cannot drift apart again.
//
// While open:
//   * focus moves in — to [data-autofocus] if present, else the first control
//   * Tab and Shift+Tab cycle inside it (the WAI-ARIA dialog pattern)
//   * Escape closes it
//   * the page behind does not scroll
// On close, focus returns to whatever opened it.

const FOCUSABLE = [
  "a[href]",
  "button:not([disabled])",
  "textarea:not([disabled])",
  "input:not([disabled]):not([type=hidden])",
  "select:not([disabled])",
  "summary",
  "[tabindex]:not([tabindex='-1'])",
].join(",");

function focusables(root: HTMLElement): HTMLElement[] {
  return [...root.querySelectorAll<HTMLElement>(FOCUSABLE)].filter(
    // Skip anything not rendered — a hidden file input, a collapsed <details>
    // body — or Tab would land on an element the user cannot see.
    (el) => el.getClientRects().length > 0,
  );
}

export function useDialog<T extends HTMLElement>(open: boolean, onClose: () => void) {
  const ref = useRef<T>(null);
  // Read through a ref, so a parent that passes a fresh arrow function on
  // every render does not re-run the effect — which would yank focus back to
  // the first control on each keystroke inside the dialog.
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;

  useEffect(() => {
    if (!open) return;
    const root = ref.current;
    if (!root) return;

    const opener = document.activeElement as HTMLElement | null;
    const first = root.querySelector<HTMLElement>("[data-autofocus]") ?? focusables(root)[0] ?? root;
    first.focus();

    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        onCloseRef.current();
        return;
      }
      if (e.key !== "Tab") return;
      const items = focusables(root);
      if (items.length === 0) {
        e.preventDefault();
        return;
      }
      const head = items[0];
      const tail = items[items.length - 1];
      const active = document.activeElement;
      if (e.shiftKey && (active === head || !root.contains(active))) {
        e.preventDefault();
        tail.focus();
      } else if (!e.shiftKey && (active === tail || !root.contains(active))) {
        e.preventDefault();
        head.focus();
      }
    };
    document.addEventListener("keydown", onKey);

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = previousOverflow;
      // Only if the opener is still on screen. When the phone sidebar closes
      // itself to open the upload dialog, the button that opened the sidebar
      // is gone from view, and focusing it would scroll the page to nowhere.
      if (opener && opener.isConnected && opener.getClientRects().length > 0) {
        opener.focus();
      }
    };
  }, [open]);

  return ref;
}

// The layout's phone/tablet form, where the sidebar becomes an overlay that
// has to behave like a dialog. Must match the 860px breakpoint in styles.css;
// a CSS custom property cannot be used inside a media query, so the number
// lives in both places.
export const COMPACT_QUERY = "(max-width: 860px)";

export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(
    () => typeof window !== "undefined" && window.matchMedia(query).matches,
  );
  useEffect(() => {
    const mql = window.matchMedia(query);
    const update = () => setMatches(mql.matches);
    update();
    mql.addEventListener("change", update);
    return () => mql.removeEventListener("change", update);
  }, [query]);
  return matches;
}

export function prefersReducedMotion(): boolean {
  return typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
}

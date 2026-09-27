"use client";

import { useCallback, useEffect, useState } from "react";

/**
 * Tracks an element's content width so SVG charts render at real pixel size (crisp text, no letterboxing).
 * Returns a callback ref, so it also works for elements that mount after the first render.
 */
export function useElementWidth<T extends HTMLElement>(fallback = 800) {
  const [el, setEl] = useState<T | null>(null);
  const [width, setWidth] = useState(fallback);
  const ref = useCallback((node: T | null) => setEl(node), []);

  useEffect(() => {
    if (!el) return;
    if (el.clientWidth) setWidth(el.clientWidth);
    if (typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver((entries) => {
      const w = Math.round(entries[0].contentRect.width);
      if (w > 0) setWidth(w);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [el]);

  return [ref, width] as const;
}

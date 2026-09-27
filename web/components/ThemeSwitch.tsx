"use client";

import { useEffect, useState } from "react";
import { effectiveTheme, setTheme, storedTheme, type Theme } from "@/lib/theme";
import Icon from "./Icon";

export default function ThemeSwitch() {
  // Unknown until mounted (the server can't know the OS theme); render the dark position meanwhile.
  const [theme, setThemeState] = useState<Theme | null>(null);

  useEffect(() => {
    setThemeState(effectiveTheme());
    // Keep following the OS until the user makes an explicit choice.
    const mq = window.matchMedia("(prefers-color-scheme: light)");
    const onChange = () => {
      if (!storedTheme()) setThemeState(mq.matches ? "light" : "dark");
    };
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);

  const dark = theme !== "light";
  const toggle = () => {
    const next: Theme = dark ? "light" : "dark";
    setTheme(next);
    setThemeState(next);
  };

  return (
    <button
      type="button"
      role="switch"
      aria-checked={dark}
      aria-label="Dark mode"
      title={dark ? "Switch to light mode" : "Switch to dark mode"}
      className="theme-switch"
      onClick={toggle}
      suppressHydrationWarning
    >
      <span className="theme-switch-thumb" aria-hidden>
        <Icon name={dark ? "moon" : "sun"} size={14} strokeWidth={2} />
      </span>
    </button>
  );
}

// Light/dark theme: follows the OS until the user flips the header switch; the choice is remembered.

export type Theme = "light" | "dark";

export const THEME_KEY = "proofvoice.theme";

/** Runs in <head> before first paint so a saved theme never flashes the other one. */
export const THEME_BOOT_SCRIPT = `(function(){try{var t=localStorage.getItem("${THEME_KEY}");if(t==="light"||t==="dark")document.documentElement.setAttribute("data-theme",t)}catch(e){}})()`;

export function systemTheme(): Theme {
  return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

export function storedTheme(): Theme | null {
  try {
    const t = window.localStorage.getItem(THEME_KEY);
    return t === "light" || t === "dark" ? t : null;
  } catch {
    return null;
  }
}

export function effectiveTheme(): Theme {
  return storedTheme() ?? systemTheme();
}

export function setTheme(t: Theme) {
  const root = document.documentElement;
  // Ease the brightness change instead of snapping (collapsed under reduced motion by the global rule).
  root.classList.add("theme-changing");
  root.setAttribute("data-theme", t);
  try {
    window.localStorage.setItem(THEME_KEY, t);
  } catch {
    /* private mode: the choice lasts for this page only */
  }
  window.setTimeout(() => root.classList.remove("theme-changing"), 320);
}

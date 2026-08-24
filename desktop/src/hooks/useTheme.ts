import { useState, useEffect, useCallback } from "react";

type Theme = "light" | "dark" | "auto";

/** Toggle .dark on <html> and persist to localStorage. CSS already exists. */
export function useTheme() {
  const [theme, setTheme] = useState<Theme>(() => {
    const saved = localStorage.getItem("huginn:theme");
    if (saved === "dark" || saved === "light" || saved === "auto") return saved;
    return "auto";
  });

  // Resolve the effective theme (auto → system preference)
  const resolvedTheme = (): "light" | "dark" => {
    if (theme === "auto") {
      return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    }
    return theme;
  };

  useEffect(() => {
    const effective = resolvedTheme();
    document.documentElement.classList.toggle("dark", effective === "dark");
    localStorage.setItem("huginn:theme", theme);
  }, [theme]);

  // Re-apply the persisted accent/skin. Setting inline style in SettingsPanel
  // is enough while the panel is open, but on reload useTheme() must restore
  // it too -- otherwise the chosen skin silently resets to the default blue.
  useEffect(() => {
    const savedRgb = localStorage.getItem("accent-color");
    if (savedRgb && /^\d+( \d+ \d+)$/.test(savedRgb.trim())) {
      document.documentElement.style.setProperty("--accent-rgb", savedRgb.trim());
      document.documentElement.style.setProperty("--seed-primary", `rgb(${savedRgb.trim()})`);
    }
  }, []);

  // Listen to system theme changes when in auto mode
  useEffect(() => {
    if (theme !== "auto") return;
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const handler = () => {
      document.documentElement.classList.toggle("dark", mq.matches);
    };
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, [theme]);

  const toggleTheme = useCallback(() => {
    setTheme((t) => {
      if (t === "light") return "dark";
      if (t === "dark") return "auto";
      return "light"; // auto → light
    });
  }, []);

  return { theme, toggleTheme, setTheme, isDark: resolvedTheme() === "dark" };
}

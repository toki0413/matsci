import { useState, useEffect, useCallback } from "react";

type Theme = "light" | "dark";

/** Toggle .dark on <html> and persist to localStorage. CSS already exists. */
export function useTheme() {
  const [theme, setTheme] = useState<Theme>(() => {
    const saved = localStorage.getItem("huginn:theme");
    if (saved === "dark" || saved === "light") return saved;
    // respect system preference on first run
    return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  });

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
    localStorage.setItem("huginn:theme", theme);
  }, [theme]);

  const toggleTheme = useCallback(() => {
    setTheme((t) => (t === "dark" ? "light" : "dark"));
  }, []);

  return { theme, toggleTheme, setTheme };
}

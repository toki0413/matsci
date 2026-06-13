/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: {
          primary: "#0b0f19",
          secondary: "#151b2b",
          tertiary: "#1e2538",
          hover: "#252d44",
        },
        accent: {
          DEFAULT: "#3b82f6",
          hover: "#2563eb",
          glow: "rgba(59, 130, 246, 0.25)",
        },
        success: "#22c55e",
        warning: "#f59e0b",
        error: "#ef4444",
        border: "#2a324a",
        text: {
          primary: "#f8fafc",
          secondary: "#94a3b8",
          muted: "#64748b",
        },
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "Roboto", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "Menlo", "Monaco", "Consolas", "monospace"],
      },
      boxShadow: {
        glow: "0 0 20px rgba(59, 130, 246, 0.15)",
      },
    },
  },
  plugins: [require("@tailwindcss/forms")],
};

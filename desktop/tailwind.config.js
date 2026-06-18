/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: {
          primary: "#181614",
          secondary: "#1e1b18",
          tertiary: "#282521",
          hover: "#352f29",
        },
        accent: {
          DEFAULT: "#d4884a",
          hover: "#dd9a62",
          glow: "rgba(212, 136, 74, 0.22)",
        },
        success: "#6b9e8a",
        warning: "#e0a84e",
        error: "#d4645a",
        border: "#262320",
        text: {
          primary: "#faf6f1",
          secondary: "#a19b94",
          muted: "#706b64",
        },
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "Roboto", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "Menlo", "Monaco", "Consolas", "monospace"],
      },
      boxShadow: {
        glow: "0 0 20px rgba(212, 136, 74, 0.15)",
      },
    },
  },
  plugins: [require("@tailwindcss/forms")],
};

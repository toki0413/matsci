/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: {
          primary: "#f8f4ef",
          secondary: "#f0ebe4",
          tertiary: "#e8e2d9",
          hover: "#ddd6cb",
        },
        accent: {
          DEFAULT: "#d4884a",
          hover: "#c07840",
          glow: "rgba(212, 136, 74, 0.15)",
        },
        success: "#6b9e8a",
        warning: "#c08830",
        error: "#d4645a",
        border: "#d5cfc6",
        text: {
          primary: "#2a2520",
          secondary: "#6b665f",
          muted: "#9a9590",
        },
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "Roboto", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "Menlo", "Monaco", "Consolas", "monospace"],
      },
      boxShadow: {
        glow: "0 0 20px rgba(212, 136, 74, 0.10)",
      },
    },
  },
  plugins: [require("@tailwindcss/forms")],
};

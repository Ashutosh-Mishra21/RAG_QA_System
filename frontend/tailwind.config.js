/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}", "./public/index.html"],
  theme: {
    extend: {
      colors: {
        bg: "#0B0F19",
        card: "#121826",
        primary: "#6366F1",
        accent: "#22D3EE",
        text: "#E5E7EB",
        muted: "#9CA3AF",
        "bubble-ai": "#1F2937",
        "border-soft": "rgba(255,255,255,0.06)",
      },
      fontFamily: {
        display: ['"Bricolage Grotesque"', "sans-serif"],
        sans: ['"Geist"', '"Inter"', "system-ui", "sans-serif"],
        mono: ['"JetBrains Mono"', "ui-monospace", "monospace"],
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(99,102,241,0.25), 0 10px 40px -10px rgba(99,102,241,0.35)",
        card: "0 1px 0 rgba(255,255,255,0.04) inset, 0 10px 30px -15px rgba(0,0,0,0.6)",
      },
      backgroundImage: {
        "grad-primary": "linear-gradient(135deg, #6366F1 0%, #22D3EE 100%)",
        "grad-soft": "linear-gradient(180deg, rgba(99,102,241,0.08) 0%, rgba(34,211,238,0.02) 100%)",
      },
      animation: {
        "fade-in": "fadeIn 0.4s ease-out both",
        "slide-up": "slideUp 0.5s cubic-bezier(0.22,1,0.36,1) both",
        "blink": "blink 1.2s infinite",
        "shimmer": "shimmer 2.5s linear infinite",
      },
      keyframes: {
        fadeIn: {
          "0%": { opacity: 0 },
          "100%": { opacity: 1 },
        },
        slideUp: {
          "0%": { opacity: 0, transform: "translateY(12px)" },
          "100%": { opacity: 1, transform: "translateY(0)" },
        },
        blink: {
          "0%,100%": { opacity: 0.2 },
          "50%": { opacity: 1 },
        },
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
      },
    },
  },
  plugins: [],
};

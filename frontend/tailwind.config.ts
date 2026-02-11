import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        background: 'var(--background)',
        surface: 'var(--surface)',
        'surface-hover': 'var(--surface-hover)',
        border: 'var(--border)',
        primary: 'var(--primary)',
        'primary-hover': 'var(--primary-hover)',
        text: 'var(--foreground)',
        'text-secondary': 'var(--text-secondary)',
      },
    },
  },
  darkMode: 'class',
  plugins: [],
}
export default config


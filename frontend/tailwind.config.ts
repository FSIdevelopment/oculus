import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        background: '#0a0a0a',
        surface: '#141414',
        'surface-hover': '#1a1a1a',
        border: '#262626',
        primary: '#3b82f6',
        'primary-hover': '#2563eb',
        text: '#ededed',
        'text-secondary': '#a3a3a3',
      },
    },
  },
  plugins: [],
}
export default config


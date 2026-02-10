import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Oculus Strategy Platform',
  description: 'AI-powered strategy design and optimization platform',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}


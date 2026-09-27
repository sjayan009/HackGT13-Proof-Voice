import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: {
    default: "ProofVoice — synthetic speech forensics",
    template: "%s · ProofVoice",
  },
  description:
    "ProofVoice detects, localizes and explains evidence of synthetic speech in audio files and live streams, with explicit uncertainty.",
  applicationName: "ProofVoice",
  robots: { index: false, follow: false },
  openGraph: {
    title: "ProofVoice",
    description: "A real-time evidence layer for synthetic speech.",
    type: "website",
  },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
  themeColor: [
    { media: "(prefers-color-scheme: dark)", color: "#07090d" },
    { media: "(prefers-color-scheme: light)", color: "#f5f6f8" },
  ],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <a className="skip-link" href="#main">
          Skip to content
        </a>
        {children}
      </body>
    </html>
  );
}

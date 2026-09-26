import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ProofVoice",
  description: "A real-time evidence layer for synthetic speech.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

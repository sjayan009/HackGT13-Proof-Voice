import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Check a voice message",
  description: "Check a voice message for signs of an AI-cloned voice. Audio is analysed and deleted, never stored.",
};

export default function CheckLayout({ children }: { children: React.ReactNode }) {
  return children;
}

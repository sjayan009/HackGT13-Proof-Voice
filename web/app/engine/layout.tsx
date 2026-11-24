import type { Metadata } from "next";

export const metadata: Metadata = { title: "Inference Control Center" };

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}

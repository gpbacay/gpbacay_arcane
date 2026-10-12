import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import "katex/dist/katex.min.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-gilroy",
  display: "swap",
  weight: ["300", "400", "500", "600", "700", "800"],
});

export const metadata: Metadata = {
  title: "ARCANE: Adaptive System 1 AI",
  description: "The library for building adaptive System 1 models with grounded decisions, neural memory, inference-time adaptation, and biologically inspired components.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        suppressHydrationWarning
        className={`${inter.variable} font-sans antialiased flex flex-col min-h-screen`}
      >
        {children}
      </body>
    </html>
  );
}

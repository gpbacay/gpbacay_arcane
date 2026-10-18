import type { Metadata } from "next";
import { Inter } from "next/font/google";
import { ArcaneChat } from "@/components/ArcaneChat";
import "./globals.css";
import "katex/dist/katex.min.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-gilroy",
  display: "swap",
  weight: ["300", "400", "500", "600", "700", "800"],
});

export const metadata: Metadata = {
  title: "ARCANE 5.0: System 1 AI",
  description: "The first library for making true System 1 AI models through adaptive inference, grounded decisions, neural memory, and biologically inspired components.",
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
        <ArcaneChat />
      </body>
    </html>
  );
}

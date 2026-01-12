import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import "katex/dist/katex.min.css";
import { SiteFooter } from "@/components/footer";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-gilroy",
  display: "swap",
  weight: ["300", "400", "500", "600", "700", "800"],
});

export const metadata: Metadata = {
  title: "A.R.C.A.N.E. Project Docs",
  description: "Augmented Reconstruction of Consciousness through Artificial Neural Evolution - Documentation for the A.R.C.A.N.E. Neural Architecture Project",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${inter.variable} font-sans antialiased flex flex-col min-h-screen`}
      >
        <div className="flex-1 flex flex-col">
          {children}
        </div>
        <SiteFooter />
      </body>
    </html>
  );
}

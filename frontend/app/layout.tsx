import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "WildEcho | AI-Powered Wildlife Audio Identification",
  description: "Identify wildlife species from bioacoustic recordings using our novel TaxaMoE (Taxa-aware Mixture of Experts) architecture. Immersive wildlife monitoring in the Pantanal wetlands.",
  keywords: ["bioacoustics", "wildlife monitoring", "TaxaMoE", "machine learning", "audio classification", "Pantanal", "birdclef"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        {children}
      </body>
    </html>
  );
}

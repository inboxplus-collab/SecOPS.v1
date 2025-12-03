import type { ReactNode } from "react";
import Link from "next/link";

export default function MarketingLayout({ children }: { children: ReactNode }) {
  return (
    <div className="bg-black text-white min-h-screen font-sans">
      <header className="p-6 flex justify-between items-center border-b border-gray-800">
        <Link href="/" className="text-3xl font-bold">
          SecOpsAI
        </Link>
        <nav className="space-x-6 text-lg flex items-center">
          <Link href="/marketing/features">Features</Link>
          <Link href="/marketing/pricing">Pricing</Link>
          <Link href="/marketing/enterprise">Enterprise</Link>
          <Link href="/marketing/docs">Docs</Link>
          <Link href="/marketing/contact">Contact</Link>
          <Link href="/console" className="px-4 py-2 bg-blue-600 rounded-lg ml-4 font-semibold">
            Launch Console →
          </Link>
        </nav>
      </header>
      <main>{children}</main>
      <footer className="text-center py-10 border-t border-gray-800 text-gray-400">
        © 2025 SecOpsAI. All rights reserved.
      </footer>
    </div>
  );
}

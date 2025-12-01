"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";

type TopNavProps = {
  /**
   * Optional: show console links (Overview / Issues / Checks / Settings)
   * If false, only brand + minimal links.
   */
  showConsoleLinks?: boolean;
};

const consoleNavItems = [
  { href: "/console", label: "Console" },
  { href: "/console/analysis", label: "Analysis" },
  { href: "/console/issues", label: "Issues" },
  { href: "/console/checks", label: "Checks" },
  { href: "/console/settings", label: "Settings" },
];

export default function TopNav({ showConsoleLinks = true }: TopNavProps) {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);

  const isActive = (href: string) => {
    if (href === "/") return pathname === "/";
    return pathname === href || pathname?.startsWith(href + "/");
  };

  const links = showConsoleLinks ? consoleNavItems : [];

  return (
    <header className="w-full border-b bg-white/70 backdrop-blur-md sticky top-0 z-40">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-3 flex items-center justify-between gap-4">
        {/* Brand */}
        <div className="flex items-center gap-2">
          <Link href="/" className="flex items-center gap-2">
            <div className="h-7 w-7 rounded-lg bg-slate-900 flex items-center justify-center text-[11px] font-semibold text-white">
              SA
            </div>
            <div className="flex flex-col leading-tight">
              <span className="text-sm font-semibold tracking-tight text-slate-900">
                SecOpsAI
              </span>
              <span className="text-[10px] text-slate-500">
                Autonomous DevSecOps
              </span>
            </div>
          </Link>
        </div>

        {/* Desktop nav */}
        {links.length > 0 && (
          <nav className="hidden md:flex items-center gap-4 text-xs">
            {links.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className={`rounded-md px-3 py-1.5 font-medium transition ${
                  isActive(item.href)
                    ? "bg-slate-900 text-white shadow-sm"
                    : "text-slate-700 hover:bg-slate-100"
                }`}
              >
                {item.label}
              </Link>
            ))}
          </nav>
        )}

        {/* Right side (placeholder for profile / auth later) */}
        <div className="hidden md:flex items-center gap-2 text-[11px] text-slate-500">
          <span className="rounded-full border border-slate-200 bg-white px-2.5 py-1">
            MVP Preview
          </span>
        </div>

        {/* Mobile menu toggle */}
        {links.length > 0 && (
          <button
            onClick={() => setOpen((v) => !v)}
            className="md:hidden inline-flex items-center justify-center rounded-md border border-slate-200 bg-white px-2 py-1.5 text-slate-700 shadow-sm"
          >
            <svg
              className="h-4 w-4"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <path d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
        )}
      </div>

      {/* Mobile dropdown */}
      {links.length > 0 && open && (
        <div className="md:hidden border-t bg-white">
          <nav className="max-w-7xl mx-auto px-4 py-2 flex flex-col gap-1 text-sm">
            {links.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                onClick={() => setOpen(false)}
                className={`rounded-md px-3 py-1.5 font-medium transition ${
                  isActive(item.href)
                    ? "bg-slate-900 text-white shadow-sm"
                    : "text-slate-700 hover:bg-slate-100"
                }`}
              >
                {item.label}
              </Link>
            ))}
          </nav>
        </div>
      )}
    </header>
  );
}

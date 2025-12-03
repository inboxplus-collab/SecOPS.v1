import Link from "next/link";

const cards = [
  { title: "Users", href: "/admin/users", description: "Seat usage, roles, and MFA status." },
  { title: "Audit Logs", href: "/admin/audit", description: "SOC2-ready event history and changes." },
  { title: "System Health", href: "/admin/system", description: "Service uptime, billing status, license usage." },
];

export default function AdminHome() {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-50 px-6 py-16">
      <div className="max-w-5xl mx-auto space-y-10">
        <div>
          <p className="text-sm text-emerald-300 uppercase tracking-widest">Admin</p>
          <h1 className="text-4xl font-bold">Admin Dashboard</h1>
          <p className="text-slate-300 mt-3 max-w-3xl">
            Monitor users, compliance, autonomous agent activity, and system health. All data is API-driven and ready for SOC2 audits.
          </p>
        </div>

        <div className="grid gap-6 md:grid-cols-3">
          {cards.map((card) => (
            <Link
              key={card.title}
              href={card.href}
              className="rounded-2xl border border-slate-800 bg-slate-900/80 p-6 shadow-lg hover:border-emerald-500/60"
            >
              <h2 className="text-xl font-semibold">{card.title}</h2>
              <p className="text-slate-300 text-sm mt-2">{card.description}</p>
              <span className="mt-4 inline-flex items-center gap-2 text-emerald-300 text-sm">
                Open <span aria-hidden>→</span>
              </span>
            </Link>
          ))}
        </div>
      </div>
    </div>
  );
}

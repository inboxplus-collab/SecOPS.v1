const services = [
  { name: "API", status: "Healthy" },
  { name: "RAG Engine", status: "Healthy" },
  { name: "Scheduler", status: "Degraded" },
];

export default function SystemStatusPage() {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-50 px-6 py-16">
      <div className="max-w-5xl mx-auto space-y-8">
        <div>
          <h1 className="text-3xl font-bold">System Health</h1>
          <p className="text-slate-300 mt-2 max-w-3xl">
            Track uptime, license usage, and billing status across the SecOpsAI control plane.
          </p>
        </div>

        <div className="grid gap-4 md:grid-cols-3">
          {services.map((service) => (
            <div key={service.name} className="rounded-xl border border-slate-800 bg-slate-900 p-4">
              <div className="flex items-center justify-between">
                <p className="font-semibold">{service.name}</p>
                <span className={`text-sm ${service.status === "Healthy" ? "text-emerald-300" : "text-amber-300"}`}>
                  {service.status}
                </span>
              </div>
              <p className="text-slate-400 text-sm mt-2">Continuous probes + admin API.</p>
            </div>
          ))}
        </div>

        <div className="rounded-xl border border-slate-800 bg-slate-900 p-6 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-slate-400">License usage</p>
              <p className="text-2xl font-semibold">18 / 25 seats</p>
            </div>
            <div>
              <p className="text-sm text-slate-400">Billing tier</p>
              <p className="text-lg font-semibold text-emerald-300">Enterprise</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <div className="h-2 flex-1 rounded-full bg-slate-800">
              <div className="h-2 rounded-full bg-emerald-400" style={{ width: "72%" }} />
            </div>
            <span className="text-sm text-slate-300">72% utilized</span>
          </div>
        </div>
      </div>
    </div>
  );
}

const assurances = [
  "Private cloud and on-prem deployment options",
  "SOC2-ready audit logging and immutable evidence",
  "SSO with Okta, Azure AD, or Google Workspace",
  "Dedicated customer success and 24/7 security response",
];

export default function Enterprise() {
  return (
    <div className="bg-slate-950 text-white min-h-screen p-20">
      <h1 className="text-5xl font-extrabold">Built for regulated enterprises</h1>
      <p className="text-lg text-slate-300 mt-4 max-w-3xl">
        Deploy SecOpsAI into isolated environments with full control over data, identity, and operational
        governance. Pair AI autonomy with the auditability and compliance posture your board expects.
      </p>

      <div className="mt-10 grid md:grid-cols-2 gap-8">
        <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-8">
          <h2 className="text-2xl font-bold text-cyan-300">Enterprise assurance</h2>
          <ul className="mt-4 space-y-3 text-slate-200">
            {assurances.map((item) => (
              <li key={item} className="flex gap-2 items-start">
                <span className="text-cyan-400">◆</span>
                <span>{item}</span>
              </li>
            ))}
          </ul>
        </div>
        <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-8">
          <h2 className="text-2xl font-bold text-cyan-300">Security operations AI</h2>
          <p className="text-slate-200 leading-relaxed mt-3">
            Leverage autonomous agents for patch orchestration, Kubernetes repair, CI/CD diagnostics, and
            live runbooks backed by RAG-driven reasoning. Every action is traceable and reversible.
          </p>
        </div>
      </div>

      <div className="mt-12">
        <a
          href="/contact"
          className="inline-block bg-blue-600 hover:bg-blue-700 px-7 py-3 rounded-lg font-semibold"
        >
          Schedule a security review
        </a>
      </div>
    </div>
  );
}

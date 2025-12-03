import EnterpriseBlock from "../components/EnterpriseBlock";

export default function Enterprise() {
  return (
    <div className="px-20 py-20">
      <h1 className="text-6xl font-bold">Enterprise</h1>
      <p className="mt-10 text-2xl text-gray-300 max-w-3xl">
        SecOpsAI is built for global-scale companies requiring high availability,
        custom compliance, on-premise deployment, and private LLM execution.
      </p>

      <div className="mt-16 grid grid-cols-1 md:grid-cols-2 gap-12">
        <EnterpriseBlock title="Private Cloud / On-Prem" desc="Deploy inside your firewalls." />
        <EnterpriseBlock
          title="Dedicated LLM Compute"
          desc="Run custom tuned models isolated per client."
        />
        <EnterpriseBlock
          title="SOC2, GDPR, HIPAA readiness"
          desc="Fully compliant auditing and privacy controls."
        />
        <EnterpriseBlock title="24/7 Autonomous Agent Monitoring" desc="AI agents that never sleep." />
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

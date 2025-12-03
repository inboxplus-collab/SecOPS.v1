const features = [
  {
    title: "Autonomous Remediation",
    description:
      "AI agents triage vulnerabilities, open pull requests, and verify fixes across cloud, CI/CD, and Kubernetes.",
  },
  {
    title: "Full-Stack Observability",
    description:
      "Prometheus, Grafana, and deep traces keep security and reliability signals unified for on-call teams.",
  },
  {
    title: "Enterprise-Grade Guardrails",
    description:
      "Role-based access, secrets isolation, and zero-trust patterns are baked into every workflow.",
  },
  {
    title: "Continuous Posture Scanning",
    description:
      "Detect drift, misconfigurations, and compliance gaps with automated evidence collection.",
  },
];

export default function Features() {
  return (
    <div className="bg-slate-950 text-white min-h-screen p-20">
      <h1 className="text-5xl font-extrabold">Capabilities built for security-led teams</h1>
      <p className="text-lg text-slate-300 mt-4 max-w-3xl">
        SecOpsAI pairs multi-model LLM reasoning with production telemetry to predict, prevent, and repair
        incidents before they impact customers.
      </p>

      <div className="grid md:grid-cols-2 gap-10 mt-12">
        {features.map((feature) => (
          <div key={feature.title} className="rounded-2xl border border-slate-800 bg-slate-900/60 p-8 shadow-xl">
            <h2 className="text-2xl font-bold text-cyan-300">{feature.title}</h2>
            <p className="text-slate-200 mt-3 leading-relaxed">{feature.description}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

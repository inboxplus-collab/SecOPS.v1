import FeatureCard from "./components/FeatureCard";
import Hero from "./components/Hero";

export default function MarketingHome() {
  return (
    <div>
      <Hero />

      <section className="px-20 py-20">
        <h2 className="text-4xl font-bold mb-8">Why SecOpsAI?</h2>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-12">
          <FeatureCard
            title="Autonomous DevSecOps"
            desc="AI that continuously scans, fixes, and optimizes your entire stack."
          />
          <FeatureCard
            title="Self-Healing Infrastructure"
            desc="Detects misconfigurations and auto-applies patches safely."
          />
          <FeatureCard
            title="Vulnerability Intelligence"
            desc="OSV, NVD, Semgrep — all unified into one AI-driven brain."
          />
        </div>
      </section>
    </div>
  );
}

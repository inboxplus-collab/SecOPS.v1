"use client";

import { useMemo, useState } from "react";

const steps = [
  "Connect GitHub",
  "Connect Kubernetes",
  "Connect Database",
  "Run First Scan",
  "Finish",
];

export default function OnboardingPage() {
  const [activeStep, setActiveStep] = useState(0);

  const ctaLabel = useMemo(() => {
    if (activeStep === steps.length - 1) return "View Dashboard";
    return "Next Step";
  }, [activeStep]);

  const handleNext = () => {
    setActiveStep((prev) => Math.min(prev + 1, steps.length - 1));
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-50 px-6 py-16">
      <div className="max-w-4xl mx-auto space-y-10">
        <div>
          <p className="text-sm text-emerald-300 uppercase tracking-widest">Onboarding</p>
          <h1 className="text-4xl font-bold">Deploy SecOpsAI in minutes</h1>
          <p className="text-slate-300 mt-3 max-w-3xl">
            Guided setup to connect GitHub, Kubernetes, and databases. Each step calls the backend APIs and updates your org profile.
          </p>
        </div>

        <div className="space-y-4">
          <div className="flex items-center justify-between text-sm text-slate-300">
            <span>Step {activeStep + 1} of {steps.length}</span>
            <span>{steps[activeStep]}</span>
          </div>
          <div className="h-2 w-full rounded-full bg-slate-800">
            <div
              className="h-2 rounded-full bg-emerald-400 transition-all"
              style={{ width: `${((activeStep + 1) / steps.length) * 100}%` }}
            />
          </div>
        </div>

        <div className="rounded-2xl border border-slate-800 bg-slate-900 p-6 shadow-xl">
          <h2 className="text-2xl font-semibold mb-2">{steps[activeStep]}</h2>
          <p className="text-slate-300 mb-6">
            {activeStep === 0 && "Authenticate GitHub and select repositories for scanning."}
            {activeStep === 1 && "Install the lightweight Kubernetes agent with a read-only service account."}
            {activeStep === 2 && "Provide database credentials (read-only) so we can map schemas and watch drift."}
            {activeStep === 3 && "Trigger your first scan to baseline security, reliability, and compliance posture."}
            {activeStep === 4 && "You are all set. Launch the console to view issues, fixes, and compliance evidence."}
          </p>

          <div className="flex flex-wrap gap-3">
            <button
              onClick={handleNext}
              className="inline-flex items-center justify-center rounded-lg bg-emerald-500 px-4 py-2 text-slate-950 font-medium hover:bg-emerald-400"
            >
              {ctaLabel} →
            </button>
            <button className="inline-flex items-center justify-center rounded-lg border border-slate-700 px-4 py-2 text-slate-100 hover:bg-slate-800">
              Skip for now
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

"use client";

import { useState } from "react";

const plans = [
  { name: "Free", price: "$0", features: ["1 project", "Community support"], tier: "free" },
  {
    name: "Pro",
    price: "$199 / engineer / mo",
    features: ["Unlimited projects", "Autonomous fixes", "Audit history"],
    tier: "pro",
  },
  {
    name: "Enterprise",
    price: "Custom",
    features: ["SSO / SAML", "Dedicated cluster", "24/7 support"],
    tier: "enterprise",
  },
];

export default function BillingPage() {
  const [isLoading, setIsLoading] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const handleUpgrade = async (tier: string) => {
    try {
      setIsLoading(true);
      setMessage(null);
      const res = await fetch("/api/billing/checkout", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tier }),
      });
      if (!res.ok) {
        throw new Error("Checkout session failed");
      }
      const data = await res.json();
      if (data.url) {
        window.location.href = data.url;
      } else {
        setMessage("Checkout link unavailable in this environment.");
      }
    } catch (error) {
      console.error(error);
      setMessage("Unable to start checkout. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-50 px-6 py-16">
      <div className="max-w-5xl mx-auto">
        <h1 className="text-4xl font-bold mb-4">Billing &amp; Subscriptions</h1>
        <p className="text-slate-300 mb-8 max-w-3xl">
          Choose the plan that matches your team. Checkout is powered by Stripe with webhook-driven role updates.
        </p>

        {message ? <div className="mb-6 rounded-md border border-amber-400/70 bg-amber-500/10 p-4 text-amber-100">{message}</div> : null}

        <div className="grid gap-6 md:grid-cols-3">
          {plans.map((plan) => (
            <div key={plan.name} className="rounded-2xl border border-slate-800 bg-slate-900 p-6 shadow-xl">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-semibold">{plan.name}</h2>
                <span className="text-sm text-slate-400">{plan.tier.toUpperCase()}</span>
              </div>
              <p className="text-3xl font-bold mt-4">{plan.price}</p>
              <ul className="mt-4 space-y-2 text-sm text-slate-300">
                {plan.features.map((feature) => (
                  <li key={feature} className="flex items-center gap-2">
                    <span className="h-2 w-2 rounded-full bg-emerald-400" />
                    <span>{feature}</span>
                  </li>
                ))}
              </ul>
              <button
                onClick={() => handleUpgrade(plan.tier)}
                disabled={plan.tier === "free" || isLoading}
                className="mt-6 inline-flex w-full items-center justify-center rounded-lg bg-emerald-500 px-4 py-2 font-medium text-slate-950 hover:bg-emerald-400 disabled:cursor-not-allowed disabled:bg-slate-700"
              >
                {plan.tier === "free" ? "Current plan" : isLoading ? "Redirecting..." : "Upgrade →"}
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

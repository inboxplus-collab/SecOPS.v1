const plans = [
  {
    name: "Starter",
    price: "$29/mo",
    description: "For small teams bootstrapping security automation.",
    features: [
      "Up to 5 projects",
      "Automated CI/CD scanning",
      "Weekly security reports",
    ],
  },
  {
    name: "Growth",
    price: "$99/mo",
    description: "For teams scaling reliability and compliance.",
    features: [
      "Unlimited projects",
      "Runtime & cloud posture monitoring",
      "Continuous vulnerability intelligence",
      "Slack / Teams alerts",
    ],
    highlight: true,
  },
  {
    name: "Enterprise",
    price: "Custom",
    description: "For global organizations with advanced needs.",
    features: [
      "Private cloud / on-prem deployment",
      "Dedicated LLM control plane",
      "SOC 2 / HIPAA controls",
      "24/7 autonomous guardrails",
    ],
  },
];

export default function PricingTable() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-10">
      {plans.map((plan) => (
        <div
          key={plan.name}
          className={`p-8 rounded-2xl border ${
            plan.highlight ? "border-blue-500 shadow-lg shadow-blue-900/30" : "border-gray-800"
          } bg-gray-900`}
        >
          <div className="flex items-center justify-between">
            <h2 className="text-3xl font-bold">{plan.name}</h2>
            <span className="text-xl text-blue-400">{plan.price}</span>
          </div>
          <p className="mt-4 text-gray-400">{plan.description}</p>
          <ul className="mt-8 space-y-3 text-gray-300">
            {plan.features.map((feature) => (
              <li key={feature} className="flex items-start space-x-3">
                <span className="text-blue-500">•</span>
                <span>{feature}</span>
              </li>
            ))}
          </ul>
          <a
            href="/marketing/contact"
            className={`mt-8 inline-block w-full text-center px-4 py-3 rounded-xl font-semibold ${
              plan.highlight
                ? "bg-blue-600 hover:bg-blue-700"
                : "bg-gray-800 hover:bg-gray-700 text-gray-200"
            }`}
          >
            Talk to sales
          </a>
        </div>
      ))}
    </div>
  );
}

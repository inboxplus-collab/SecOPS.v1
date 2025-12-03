"use client";

import { useState } from "react";
import { api } from "@/lib/api-client";

export default function HealButton({ issueId }: { issueId: string }) {
  const [loading, setLoading] = useState(false);

  async function heal() {
    setLoading(true);
    await api.k8s.heal(issueId);
    setLoading(false);
  }

  return (
    <button onClick={heal} className="px-3 py-1 bg-green-600 text-white rounded">
      {loading ? "Healing…" : "Auto-Heal"}
    </button>
  );
}

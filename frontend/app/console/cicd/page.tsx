"use client";

import { useState, useEffect } from "react";
import { api } from "@/lib/api-client";

export default function CICDPage() {
  const [runs, setRuns] = useState([]);

  useEffect(() => {
    api.cicd.github.list().then(setRuns);
  }, []);

  return (
    <div className="p-6 space-y-6">
      <h1 className="text-xl font-semibold">CI/CD Pipeline</h1>

      <div className="space-y-3">
        {runs.map((run: any) => (
          <div key={run.id} className="border p-3 rounded bg-white shadow">
            <div className="font-semibold">{run.name}</div>
            <div>Status: {run.conclusion}</div>
            <button className="mt-2 px-3 py-1 bg-blue-600 text-white rounded">
              Analyze & Auto-Fix
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

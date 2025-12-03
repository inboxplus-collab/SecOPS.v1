"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api-client";
import type { ActivityItem } from "@/lib/types";

export default function ActivityPage() {
  const [items, setItems] = useState<ActivityItem[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.activity
      .list()
      .then(setItems)
      .catch(() => setError("Unable to load activity right now."));
  }, []);

  return (
    <div className="p-6 space-y-6">
      <h1 className="text-xl font-semibold">Activity Timeline</h1>

      {error ? (
        <div className="text-sm text-red-600">{error}</div>
      ) : null}

      <div className="space-y-4">
        {items.length === 0 ? (
          <div className="rounded border border-dashed border-neutral-300 bg-white p-4 text-sm text-neutral-500">
            No activity yet. Events from scans, agents, and automation will
            appear here.
          </div>
        ) : (
          items.map((e, i) => (
            <div
              key={e.id || i}
              className="border-l-4 border-blue-600 bg-white pl-4 pr-3 py-2 rounded shadow"
            >
              <div className="font-semibold">{e.title}</div>
              <div className="text-xs text-gray-500">{e.timestamp}</div>
              <div className="text-gray-700 mt-1 whitespace-pre-line">
                {e.description}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

"use client";

import { useState } from "react";
import { api } from "@/lib/api-client";

export default function RagViewerPage() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<any>(null);

  async function runSearch() {
    const res = await api.rag.search({ query });
    setResult(res);
  }

  return (
    <div className="p-6 space-y-4">
      <h1 className="text-xl font-semibold">RAG Explorer</h1>

      <div className="flex gap-2">
        <input
          className="border p-2 rounded w-full"
          placeholder="Ask anything…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button onClick={runSearch} className="px-4 py-2 bg-blue-600 text-white rounded">
          Search
        </button>
      </div>

      {result && (
        <div className="space-y-4">
          <h2 className="font-semibold text-lg">AI Answer</h2>
          <p className="bg-white p-3 rounded shadow">{result.answer}</p>

          <h2 className="font-semibold text-lg">Citations</h2>
          <ul className="space-y-1">
            {result.citations.map((c: any, i: number) => (
              <li key={i} className="text-sm text-gray-600">
                <span className="font-semibold">{c.source}</span>: {c.content.slice(0, 200)}…
              </li>
            ))}
          </ul>

          <h2 className="font-semibold text-lg">Retrieved Docs</h2>
          <pre className="bg-gray-900 text-gray-100 p-4 text-sm rounded">
            {JSON.stringify(result.debug, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}

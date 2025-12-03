"use client";

import { FormEvent, useState } from "react";

export default function DocsChatPage() {
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState<string>("");
  const [citations, setCitations] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setIsLoading(true);
    setAnswer("");
    setCitations([]);

    try {
      const res = await fetch("/api/docs/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: query }),
      });
      const data = await res.json();
      setAnswer(data.answer ?? "No answer generated.");
      setCitations(data.citations ?? []);
    } catch (error) {
      console.error(error);
      setAnswer("Docs chat failed. Please retry.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-50 px-6 py-16">
      <div className="max-w-4xl mx-auto space-y-8">
        <div>
          <p className="text-sm text-emerald-300 uppercase tracking-widest">DocsGPT</p>
          <h1 className="text-4xl font-bold">Documentation chat</h1>
          <p className="text-slate-300 mt-3 max-w-3xl">
            Ask anything about SecOpsAI. The backend RAG service retrieves doc chunks and the LLM responds with citations.
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="rounded-xl border border-slate-800 bg-slate-900 p-4 shadow-lg">
            <textarea
              className="w-full rounded-lg border border-slate-800 bg-slate-950 p-3 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500"
              rows={3}
              placeholder="How do I connect GitHub?"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
            <div className="mt-3 flex items-center justify-between">
              <span className="text-xs text-slate-400">RAG + LLM with citations</span>
              <button
                type="submit"
                disabled={isLoading}
                className="inline-flex items-center justify-center rounded-lg bg-emerald-500 px-4 py-2 text-slate-950 font-medium hover:bg-emerald-400 disabled:opacity-60"
              >
                {isLoading ? "Thinking..." : "Ask"}
              </button>
            </div>
          </div>
        </form>

        {answer && (
          <div className="rounded-xl border border-slate-800 bg-slate-900 p-6 space-y-3">
            <p className="text-lg font-semibold">Answer</p>
            <p className="text-slate-200 whitespace-pre-line">{answer}</p>
            {citations.length > 0 && (
              <div className="pt-2">
                <p className="text-sm text-slate-400 mb-2">Citations</p>
                <ul className="list-disc list-inside text-sm text-slate-300 space-y-1">
                  {citations.map((cite) => (
                    <li key={cite}>{cite}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

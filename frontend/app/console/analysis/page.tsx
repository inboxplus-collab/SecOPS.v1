"use client";

import { FormEvent, useState } from "react";
import { Button } from "@/components/shared/Button";
import { Card } from "@/components/shared/Card";
import { Input } from "@/components/shared/Input";
import { runAnalysis, RunAnalysisPayload, RunAnalysisResponse } from "@/lib/api-client";

export default function AnalysisPage() {
  const [repository, setRepository] = useState("");
  const [databaseUrl, setDatabaseUrl] = useState("");
  const [codePath, setCodePath] = useState("");
  const [result, setResult] = useState<RunAnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    const payload: RunAnalysisPayload = {
      repository: repository.trim(),
      database_url: databaseUrl.trim() || undefined,
      code_path: codePath.trim() || undefined,
    };

    setError(null);
    setResult(null);
    setLoading(true);

    try {
      const response = await runAnalysis(payload);
      setResult(response);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to run analysis.";
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="flex min-h-screen flex-col bg-neutral-950 text-neutral-50">
      <div className="mx-auto flex w-full max-w-5xl flex-1 flex-col gap-4 px-4 py-6">
        <header className="space-y-1">
          <h1 className="text-2xl font-semibold">Repository analysis</h1>
          <p className="text-sm text-neutral-400">
            Provide inputs for repository, database, and code path to trigger an analysis run.
          </p>
        </header>

        <Card className="space-y-4">
          <form onSubmit={handleSubmit} className="space-y-4">
            <Input
              label="Repository"
              placeholder="owner/repo"
              value={repository}
              onChange={(event) => setRepository(event.target.value)}
              required
            />
            <Input
              label="Database URL"
              placeholder="postgresql://user:pass@host:5432/dbname"
              helperText="Optional. Used for database-aware analysis."
              value={databaseUrl}
              onChange={(event) => setDatabaseUrl(event.target.value)}
            />
            <Input
              label="Code path"
              placeholder="services/api"
              helperText="Optional relative path within the repository."
              value={codePath}
              onChange={(event) => setCodePath(event.target.value)}
            />
            <div className="flex gap-2">
              <Button type="submit" loading={loading} disabled={!repository.trim()}>
                {loading ? "Running analysis…" : "Run analysis"}
              </Button>
              <Button type="button" variant="secondary" onClick={() => setResult(null)} disabled={loading}>
                Clear results
              </Button>
            </div>
          </form>

          {error && (
            <div className="rounded-md border border-red-900/70 bg-red-900/30 px-3 py-2 text-sm text-red-100">
              {error}
            </div>
          )}

          {result && (
            <div className="space-y-2">
              <h2 className="text-sm font-medium text-neutral-200">Analysis result</h2>
              <pre className="overflow-auto rounded-md border border-neutral-800 bg-neutral-950 p-3 text-xs text-neutral-200">
                {JSON.stringify(result, null, 2)}
              </pre>
            </div>
          )}
        </Card>
      </div>
    </main>
  );
}

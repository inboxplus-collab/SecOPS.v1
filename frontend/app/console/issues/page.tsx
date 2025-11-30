import { Suspense } from "react";
import { IssueList } from "@/components/issues/IssueList";
import { Card } from "@/components/shared/Card";

export default function IssuesPage() {
  return (
    <main className="flex min-h-screen flex-col bg-neutral-950 text-neutral-50">
      <div className="mx-auto flex w-full max-w-5xl flex-1 flex-col gap-4 px-4 py-6">
        <header>
          <h1 className="text-2xl font-semibold">Issues</h1>
          <p className="text-sm text-neutral-400">
            Findings from GitHub, Kubernetes, CI pipelines, and scanners.
          </p>
        </header>

        <Card>
          <Suspense fallback={<div className="text-xs text-neutral-500">Loading issues…</div>}>
            <IssueList />
          </Suspense>
        </Card>
      </div>
    </main>
  );
}

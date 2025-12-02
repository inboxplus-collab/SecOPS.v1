"use client";

import { useEffect, useMemo, useState } from "react";
import { IssueDetailHeader } from "@/components/issues/IssueDetailHeader";
import IssueDetailTabs from "@/components/issues/IssueDetailTabs";
import { Card } from "@/components/shared/Card";
import { fetchIssueById } from "@/lib/api-client";
import type { IssueDetail } from "@/lib/types";

interface IssueDetailPageProps {
  params: { id: string };
}

export default function IssueDetailPage({ params }: IssueDetailPageProps) {
  const { id } = params;
  const [issue, setIssue] = useState<IssueDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState("overview");

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        setLoading(true);
        setError(null);
        const data = await fetchIssueById(id);
        if (!cancelled) {
          setIssue(data);
        }
      } catch (err) {
        if (!cancelled) {
          setError("Unable to fetch issue details.");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, [id]);

  const tabs = useMemo(
    () => [
      { id: "overview", label: "Overview" },
      {
        id: "metadata",
        label: "Metadata",
        badgeCount: issue ? Object.keys(issue.extra ?? issue.metadata ?? {}).length : 0,
      },
      {
        id: "references",
        label: "References",
        badgeCount: issue?.references?.length ?? 0,
        disabled: !issue || (issue.references?.length ?? 0) === 0,
      },
    ],
    [issue]
  );

  return (
    <main className="flex min-h-screen flex-col bg-neutral-950 text-neutral-50">
      <div className="mx-auto flex w-full max-w-5xl flex-1 flex-col gap-4 px-4 py-6">
        <IssueDetailHeader issue={issue ?? undefined} issueId={id} />
        <Card className="space-y-4">
          <IssueDetailTabs tabs={tabs} activeId={activeTab} onChange={setActiveTab} />
          <div className="space-y-3 text-sm text-neutral-300">
            {loading && <div className="text-xs text-neutral-500">Loading details…</div>}
            {error && (
              <div className="rounded-md border border-red-500/50 bg-red-950/40 p-3 text-xs text-red-100">{error}</div>
            )}
            {!loading && !error && issue && activeTab === "overview" && (
              <div className="space-y-2">
                <p className="text-base text-neutral-50">{issue.description ?? "No description provided."}</p>
                {issue.root_cause && (
                  <div>
                    <h3 className="text-xs font-semibold uppercase tracking-wide text-neutral-400">Root Cause</h3>
                    <p>{issue.root_cause}</p>
                  </div>
                )}
                {issue.impact && (
                  <div>
                    <h3 className="text-xs font-semibold uppercase tracking-wide text-neutral-400">Impact</h3>
                    <p>{issue.impact}</p>
                  </div>
                )}
                {issue.proposed_fix && (
                  <div>
                    <h3 className="text-xs font-semibold uppercase tracking-wide text-neutral-400">Proposed Fix</h3>
                    <p>{issue.proposed_fix}</p>
                  </div>
                )}
              </div>
            )}
            {!loading && !error && issue && activeTab === "metadata" && (
              <div className="rounded-md border border-neutral-800 bg-neutral-900/60 p-3">
                <pre className="whitespace-pre-wrap text-xs text-neutral-200">
                  {JSON.stringify(issue.extra ?? issue.metadata ?? {}, null, 2)}
                </pre>
              </div>
            )}
            {!loading && !error && issue && activeTab === "references" && (
              <ul className="list-disc space-y-1 pl-4 text-neutral-200">
                {(issue.references ?? []).map((ref) => (
                  <li key={ref}>{ref}</li>
                ))}
              </ul>
            )}
          </div>
        </Card>
      </div>
    </main>
  );
}

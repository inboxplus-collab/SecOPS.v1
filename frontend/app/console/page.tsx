"use client";

import Link from "next/link";
import { Suspense, useEffect, useMemo, useState } from "react";
import { Card } from "@/components/shared/Card";
import { Button } from "@/components/shared/Button";
import DashboardCharts from "@/components/issues/DashboardCharts";
import {
  fetchCostBreakdown,
  fetchIssues,
  fetchPlatformSummary,
} from "@/lib/api-client";

export default function ConsolePage() {
  const [issuesOpen, setIssuesOpen] = useState<number | null>(null);
  const [checksFailing, setChecksFailing] = useState<number | null>(null);
  const [monthlyCost, setMonthlyCost] = useState<number | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const [summary, costs, issues] = await Promise.all([
          fetchPlatformSummary(),
          fetchCostBreakdown(),
          fetchIssues({ page: 1, pageSize: 1 }),
        ]);

        if (cancelled) return;

        setIssuesOpen(summary.issues_open ?? issues.total ?? 0);
        setChecksFailing(summary.checks_failing ?? 0);
        setMonthlyCost(
          costs?.reduce((acc, item) => acc + (item.amount ?? 0), 0) ?? null
        );
      } catch (e) {
        if (!cancelled) {
          setIssuesOpen(null);
          setChecksFailing(null);
          setMonthlyCost(null);
        }
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  const formattedCost = useMemo(() => {
    if (monthlyCost == null) return "$ –";
    return `$ ${monthlyCost.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
  }, [monthlyCost]);

  return (
    <main className="flex min-h-screen flex-col bg-neutral-950 text-neutral-50">
      <div className="mx-auto flex w-full max-w-6xl flex-1 flex-col gap-6 px-4 py-6">
        <header className="flex flex-col justify-between gap-4 md:flex-row md:items-center">
          <div>
            <h1 className="text-2xl font-semibold">SecOpsAI Console</h1>
            <p className="text-sm text-neutral-400">
              Autonomous insights across issues, checks, and infrastructure.
            </p>
          </div>
          <div className="flex gap-2">
            <Link href="/console/issues">
              <Button variant="secondary">View Issues</Button>
            </Link>
            <Link href="/console/checks">
              <Button>View Checks</Button>
            </Link>
          </div>
        </header>

        <section className="grid gap-4 md:grid-cols-3">
          <Card>
            <h2 className="text-sm font-medium text-neutral-200">Issues</h2>
            <p className="mt-1 text-2xl font-semibold">
              {issuesOpen == null ? "–" : issuesOpen}
            </p>
            <p className="mt-1 text-xs text-neutral-500">
              Total open issues across all integrations.
            </p>
          </Card>
          <Card>
            <h2 className="text-sm font-medium text-neutral-200">Checks failing</h2>
            <p className="mt-1 text-2xl font-semibold">
              {checksFailing == null ? "–" : checksFailing}
            </p>
            <p className="mt-1 text-xs text-neutral-500">
              Critical checks that need attention.
            </p>
          </Card>
          <Card>
            <h2 className="text-sm font-medium text-neutral-200">
              Monthly cost estimate
            </h2>
            <p className="mt-1 text-2xl font-semibold">{formattedCost}</p>
            <p className="mt-1 text-xs text-neutral-500">
              Rough across compute, storage, network, and LLM usage.
            </p>
          </Card>
        </section>

        <section>
          <Card className="space-y-4">
            <div className="flex flex-col justify-between gap-2 md:flex-row md:items-center">
              <div>
                <h2 className="text-sm font-medium text-neutral-200">
                  Risk & activity overview
                </h2>
                <p className="text-xs text-neutral-500">
                  Issues trend, check status, and cost breakdown.
                </p>
              </div>
              <Link href="/console/issues">
                <Button variant="ghost" className="text-xs">
                  Explore issues →
                </Button>
              </Link>
            </div>
            <Suspense fallback={<div className="text-xs text-neutral-500">Loading charts…</div>}>
              <DashboardCharts />
            </Suspense>
          </Card>
        </section>
      </div>
    </main>
  );
}

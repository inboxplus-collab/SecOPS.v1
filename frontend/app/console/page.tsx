import Link from "next/link";
import { Suspense } from "react";
import { Card } from "@/components/shared/Card";
import { Button } from "@/components/shared/Button";
import { DashboardCharts } from "@/components/issues/DashboardCharts";

export default function ConsolePage() {
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
            <p className="mt-1 text-2xl font-semibold">–</p>
            <p className="mt-1 text-xs text-neutral-500">
              Total open issues across all integrations.
            </p>
          </Card>
          <Card>
            <h2 className="text-sm font-medium text-neutral-200">Checks failing</h2>
            <p className="mt-1 text-2xl font-semibold">–</p>
            <p className="mt-1 text-xs text-neutral-500">
              Critical checks that need attention.
            </p>
          </Card>
          <Card>
            <h2 className="text-sm font-medium text-neutral-200">
              Monthly cost estimate
            </h2>
            <p className="mt-1 text-2xl font-semibold">$ –</p>
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

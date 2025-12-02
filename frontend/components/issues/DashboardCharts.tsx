"use client";

import React, { useEffect, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Bar,
  BarChart,
  CartesianGrid,
} from "recharts";
import { Card } from "../shared/Card";
import Loader from "../shared/Loader";
import EmptyState from "../shared/EmptyState";
import { fetchIssues, fetchIssuesTrend } from "@/lib/api-client";
import { Issue } from "@/lib/types";

interface SeverityDistribution {
  severity: Issue["severity"];
  count: number;
}

const DashboardCharts: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [issueTrend, setIssueTrend] = useState<{ date: string; count: number }[]>([]);
  const [severityStats, setSeverityStats] = useState<SeverityDistribution[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        setLoading(true);
        setError(null);

        const [trend, issuesResponse] = await Promise.all([
          fetchIssuesTrend(),
          fetchIssues({ page: 1, pageSize: 100 }),
        ]);

        if (cancelled) return;

        const severityCounts = (issuesResponse.items ?? []).reduce<
          Record<Issue["severity"], number>
        >((acc, issue) => {
          acc[issue.severity] = (acc[issue.severity] ?? 0) + 1;
          return acc;
        }, {
          info: 0,
          low: 0,
          medium: 0,
          high: 0,
          critical: 0,
        });

        setIssueTrend(trend);
        setSeverityStats(
          Object.entries(severityCounts).map(([severity, count]) => ({
            severity: severity as Issue["severity"],
            count,
          }))
        );
      } catch (e) {
        if (!cancelled) setError("Failed to load dashboard data.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  if (loading) {
    return (
      <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
        <Card>
          <Loader />
        </Card>
        <Card>
          <Loader />
        </Card>
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-md border border-red-500/50 bg-red-950/40 p-4 text-sm text-red-100">
        {error}
      </div>
    );
  }

  if (
    (!issueTrend || issueTrend.length === 0) &&
    (!severityStats || severityStats.length === 0)
  ) {
    return (
      <EmptyState
        title="No Data Yet"
        description="Run checks to populate dashboard charts."
      />
    );
  }

  return (
    <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
      <Card title="Issues Over Time">
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={issueTrend}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2f2f2f" />
              <XAxis dataKey="date" stroke="#888" />
              <YAxis stroke="#888" />
              <Tooltip
                contentStyle={{ backgroundColor: "#111", border: "1px solid #333" }}
                labelStyle={{ color: "#fff" }}
              />
              <Line
                type="monotone"
                dataKey="count"
                stroke="#4f9fff"
                strokeWidth={2}
                dot={{ r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </Card>

      <Card title="Severity Distribution">
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={severityStats}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2f2f2f" />
              <XAxis dataKey="severity" stroke="#888" />
              <YAxis stroke="#888" />
              <Tooltip
                contentStyle={{ backgroundColor: "#111", border: "1px solid #333" }}
                labelStyle={{ color: "#fff" }}
              />
              <Bar dataKey="count" fill="#ff5757" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Card>
    </div>
  );
};

export default DashboardCharts;

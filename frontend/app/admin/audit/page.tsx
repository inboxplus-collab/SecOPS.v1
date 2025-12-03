const auditLogs = Array.from({ length: 12 }).map((_, idx) => ({
  id: `audit-${idx + 1}`,
  actor: idx % 2 === 0 ? "system" : "founder@secops.ai",
  action: idx % 2 === 0 ? "policy.update" : "scan.run",
  target: `service-${idx + 1}`,
  createdAt: new Date(Date.now() - idx * 5 * 60 * 1000).toISOString(),
}));

export default function AuditLogsPage() {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-50 px-6 py-16">
      <div className="max-w-5xl mx-auto">
        <h1 className="text-3xl font-bold">Audit Logs</h1>
        <p className="text-slate-300 mt-2 mb-6 max-w-3xl">
          SOC2-friendly audit history showing every admin action, agent remediation, and policy update.
        </p>

        <div className="rounded-xl border border-slate-800 bg-slate-900 divide-y divide-slate-800">
          {auditLogs.map((entry) => (
            <div key={entry.id} className="flex items-center justify-between px-4 py-3">
              <div>
                <p className="font-medium">{entry.action}</p>
                <p className="text-slate-400 text-sm">
                  {entry.actor} on {entry.target}
                </p>
              </div>
              <span className="text-xs text-slate-400">{new Date(entry.createdAt).toLocaleString()}</span>
            </div>
          ))}
        </div>
      </div>
"use client";
import { useEffect, useState } from "react";
import { api } from "@/lib/api-client";
import type { AuditLogEntry } from "@/lib/types";

export default function AdminAudit() {
  const [logs, setLogs] = useState<AuditLogEntry[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.admin
      .audit()
      .then(setLogs)
      .catch(() => setError("Unable to load audit logs."));
  }, []);

  return (
    <div className="p-6">
      <h1 className="text-xl font-semibold mb-4">Audit Logs</h1>

      {error ? (
        <div className="mb-4 text-sm text-red-600">{error}</div>
      ) : null}

      <table className="w-full text-left bg-white shadow rounded">
        <thead>
          <tr className="border-b">
            <th className="p-3">Actor</th>
            <th className="p-3">Action</th>
            <th className="p-3">Target</th>
            <th className="p-3">Timestamp</th>
          </tr>
        </thead>

        <tbody>
          {logs.map((log) => (
            <tr key={log.id} className="border-b last:border-none">
              <td className="p-3">{log.actor}</td>
              <td className="p-3">{log.action}</td>
              <td className="p-3">{log.target || "—"}</td>
              <td className="p-3">{log.timestamp}</td>
            </tr>
          ))}
          {logs.length === 0 ? (
            <tr>
              <td className="p-3 text-sm text-neutral-500" colSpan={4}>
                No audit activity yet.
              </td>
            </tr>
          ) : null}
        </tbody>
      </table>
    </div>
  );
}

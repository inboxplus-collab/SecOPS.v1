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

const users = [
  { email: "founder@secops.ai", role: "admin", tier: "enterprise", status: "Active", mfa: true },
  { email: "engineer@secops.ai", role: "member", tier: "pro", status: "Active", mfa: false },
  { email: "guest@secops.ai", role: "viewer", tier: "free", status: "Invited", mfa: false },
];

export default function AdminUsersPage() {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-50 px-6 py-16">
      <div className="max-w-5xl mx-auto">
        <h1 className="text-3xl font-bold">User Directory</h1>
        <p className="text-slate-300 mt-2 mb-6 max-w-3xl">
          Track seats, MFA adoption, and subscription tier in one place. Data syncs with the billing webhook and identity provider.
        </p>

        <div className="overflow-hidden rounded-xl border border-slate-800 bg-slate-900">
          <table className="min-w-full divide-y divide-slate-800 text-sm">
            <thead className="bg-slate-900/80">
              <tr>
                <th className="px-4 py-3 text-left font-semibold">Email</th>
                <th className="px-4 py-3 text-left font-semibold">Role</th>
                <th className="px-4 py-3 text-left font-semibold">Tier</th>
                <th className="px-4 py-3 text-left font-semibold">Status</th>
                <th className="px-4 py-3 text-left font-semibold">MFA</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800">
              {users.map((user) => (
                <tr key={user.email} className="hover:bg-slate-800/40">
                  <td className="px-4 py-3">{user.email}</td>
                  <td className="px-4 py-3 text-slate-300">{user.role}</td>
                  <td className="px-4 py-3 text-emerald-300">{user.tier}</td>
                  <td className="px-4 py-3">{user.status}</td>
                  <td className="px-4 py-3">{user.mfa ? "Enabled" : "Missing"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
"use client";
import { useEffect, useState } from "react";
import { api } from "@/lib/api-client";
import type { AdminUser } from "@/lib/types";

export default function AdminUsers() {
  const [users, setUsers] = useState<AdminUser[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.admin
      .users()
      .then(setUsers)
      .catch(() => setError("Unable to load users."));
  }, []);

  return (
    <div className="p-6">
      <h1 className="text-xl font-semibold mb-4">Users</h1>

      {error ? (
        <div className="mb-4 text-sm text-red-600">{error}</div>
      ) : null}

      <table className="w-full text-left bg-white shadow rounded">
        <thead>
          <tr className="border-b">
            <th className="p-3">Email</th>
            <th className="p-3">Role</th>
            <th className="p-3">Status</th>
          </tr>
        </thead>

        <tbody>
          {users.map((u) => (
            <tr key={u.id} className="border-b last:border-none">
              <td className="p-3">{u.email}</td>
              <td className="p-3">{u.role}</td>
              <td className="p-3">{u.active ? "Active" : "Disabled"}</td>
            </tr>
          ))}
          {users.length === 0 ? (
            <tr>
              <td className="p-3 text-sm text-neutral-500" colSpan={3}>
                No users found.
              </td>
            </tr>
          ) : null}
        </tbody>
      </table>
    </div>
  );
}

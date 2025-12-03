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

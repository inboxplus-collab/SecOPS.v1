"use client";
import { useEffect, useState } from "react";
import { api } from "@/lib/api-client";
import type { AdminTeam } from "@/lib/types";

export default function AdminTeams() {
  const [teams, setTeams] = useState<AdminTeam[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.admin
      .teams()
      .then(setTeams)
      .catch(() => setError("Unable to load teams."));
  }, []);

  return (
    <div className="p-6">
      <h1 className="text-xl font-semibold mb-4">Teams</h1>

      {error ? (
        <div className="mb-4 text-sm text-red-600">{error}</div>
      ) : null}

      <table className="w-full text-left bg-white shadow rounded">
        <thead>
          <tr className="border-b">
            <th className="p-3">Name</th>
            <th className="p-3">Members</th>
            <th className="p-3">Lead</th>
            <th className="p-3">Status</th>
          </tr>
        </thead>

        <tbody>
          {teams.map((team) => (
            <tr key={team.id} className="border-b last:border-none">
              <td className="p-3">{team.name}</td>
              <td className="p-3">{team.members}</td>
              <td className="p-3">{team.lead || "—"}</td>
              <td className="p-3">{team.active ? "Active" : "Inactive"}</td>
            </tr>
          ))}
          {teams.length === 0 ? (
            <tr>
              <td className="p-3 text-sm text-neutral-500" colSpan={4}>
                No teams found.
              </td>
            </tr>
          ) : null}
        </tbody>
      </table>
    </div>
  );
}

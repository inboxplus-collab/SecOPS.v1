"use client";

import React from "react";
import { ChevronRight } from "lucide-react";
import Link from "next/link";
import clsx from "clsx";

export interface IssueCardProps {
  id: string;
  title: string;
  severity: "info" | "low" | "medium" | "high" | "critical";
  status: "open" | "acknowledged" | "resolved" | "ignored";
  source: string;                // github, k8s, ci, scanner, rag, manual
  target?: string;               // repo name, cluster, service, etc.
  created_at?: string;
  metadata?: Record<string, any>;
}

/* -------------------------------------------------------------------------- */
/*                                UI Helpers                                  */
/* -------------------------------------------------------------------------- */

const severityColors: Record<IssueCardProps["severity"], string> = {
  info: "bg-blue-100 text-blue-700 border-blue-300",
  low: "bg-green-100 text-green-700 border-green-300",
  medium: "bg-yellow-100 text-yellow-800 border-yellow-400",
  high: "bg-orange-100 text-orange-800 border-orange-400",
  critical: "bg-red-100 text-red-800 border-red-400",
};

const statusColors: Record<IssueCardProps["status"], string> = {
  open: "bg-red-100 text-red-700 border-red-300",
  acknowledged: "bg-yellow-100 text-yellow-800 border-yellow-300",
  resolved: "bg-green-100 text-green-700 border-green-300",
  ignored: "bg-gray-100 text-gray-600 border-gray-300",
};

/* -------------------------------------------------------------------------- */
/*                                 Component                                  */
/* -------------------------------------------------------------------------- */

export const IssueCard: React.FC<IssueCardProps> = ({
  id,
  title,
  severity,
  status,
  source,
  target,
  created_at,
  metadata,
}) => {
  return (
    <Link href={`/console/issues/${id}`} className="block w-full">
      <div
        className={clsx(
          "p-4 rounded-xl border bg-white shadow-sm hover:shadow-md transition-all duration-150 cursor-pointer",
          "border-neutral-200"
        )}
      >
        {/* HEADER ROW */}
        <div className="flex items-start justify-between">
          <div className="flex flex-col gap-1">
            <h3 className="text-lg font-semibold text-neutral-900">{title}</h3>

            <div className="flex flex-wrap items-center gap-2 mt-1">
              {/* Severity Tag */}
              <span
                className={clsx(
                  "px-2 py-0.5 text-xs font-medium rounded border",
                  severityColors[severity]
                )}
              >
                {severity.toUpperCase()}
              </span>

              {/* Status Tag */}
              <span
                className={clsx(
                  "px-2 py-0.5 text-xs font-medium rounded border",
                  statusColors[status]
                )}
              >
                {status.toUpperCase()}
              </span>

              {/* Source */}
              <span className="px-2 py-0.5 text-xs font-medium rounded bg-neutral-100 text-neutral-600 border border-neutral-300">
                {source}
              </span>

              {/* Target (optional) */}
              {target && (
                <span className="px-2 py-0.5 text-xs font-medium rounded bg-neutral-100 text-neutral-600 border border-neutral-300">
                  {target}
                </span>
              )}
            </div>
          </div>

          {/* Icon */}
          <ChevronRight className="w-5 h-5 text-neutral-400" />
        </div>

        {/* FOOTER INFO */}
        <div className="flex justify-between items-center mt-4 text-xs text-neutral-500">
          <div>
            Created:{" "}
            {created_at
              ? new Date(created_at).toLocaleString()
              : "Unknown"}
          </div>

          {metadata && (
            <div className="text-neutral-400 italic">
              {Object.keys(metadata).length} metadata fields
            </div>
          )}
        </div>
      </div>
    </Link>
  );
};

export default IssueCard;

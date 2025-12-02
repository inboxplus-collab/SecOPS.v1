"use client";

import React, { useEffect, useState } from "react";
import clsx from "clsx";
import { fetchIssueById } from "@/lib/api-client";
import { Issue, IssueDetail } from "@/lib/types";
import IssueSeverityIndicator from "./IssueSeverityIndicator";
import IssueStatusIndicator from "./IssueStatusIndicator";
import Button from "../shared/Button";

interface IssueDetailHeaderProps {
  issue?: Issue | IssueDetail;
  issueId?: string;
  onResolve?: (issueId: string) => void;
  className?: string;
}

const IssueDetailHeader: React.FC<IssueDetailHeaderProps> = ({
  issue,
  issueId,
  onResolve,
  className,
}) => {
  const [resolvedIssue, setResolvedIssue] = useState<Issue | IssueDetail | null>(issue ?? null);
  const [loading, setLoading] = useState<boolean>(!!issueId && !issue);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!issueId || issue) return;

    let cancelled = false;
    async function load() {
      try {
        setLoading(true);
        const data = await fetchIssueById(issueId);
        if (!cancelled) {
          setResolvedIssue(data);
        }
      } catch (e) {
        if (!cancelled) setError("Unable to load issue details.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, [issueId, issue]);

  if (error) {
    return (
      <div className="rounded-md border border-red-500/50 bg-red-950/40 p-4 text-sm text-red-100">
        {error}
      </div>
    );
  }

  if (loading || !resolvedIssue) {
    return <div className="text-xs text-neutral-500">Loading issue…</div>;
  }

  return (
    <div className={clsx("border-b pb-4 flex flex-col gap-3", className)}>
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold text-neutral-50">{resolvedIssue.title}</h1>

          <div className="mt-1 flex gap-4 items-center">
            <IssueSeverityIndicator severity={resolvedIssue.severity} size="sm" />
            <IssueStatusIndicator status={resolvedIssue.status} size="sm" />
          </div>
        </div>

        {onResolve && resolvedIssue.status !== "resolved" && (
          <Button
            variant="primary"
            onClick={() => onResolve(resolvedIssue.id)}
            className="w-fit"
          >
            Mark as Resolved
          </Button>
        )}
      </div>

      <div className="text-sm text-neutral-400 flex flex-wrap gap-4 mt-1">
        <span>
          <span className="font-medium text-neutral-300">Issue ID:</span>{" "}
          {resolvedIssue.id}
        </span>

        {resolvedIssue.created_at && (
          <span>
            <span className="font-medium text-neutral-300">Created:</span>{" "}
            {new Date(resolvedIssue.created_at).toLocaleString()}
          </span>
        )}

        {resolvedIssue.source && (
          <span>
            <span className="font-medium text-neutral-300">Source:</span>{" "}
            {resolvedIssue.source}
          </span>
        )}
      </div>
    </div>
  );
};

export { IssueDetailHeader };
export default IssueDetailHeader;

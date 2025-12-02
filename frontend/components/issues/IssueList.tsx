"use client";

import React, { useEffect, useState } from "react";
import IssueCard from "./IssueCard";
import Loader from "../shared/Loader";
import EmptyState from "../shared/EmptyState";
import { fetchIssues } from "@/lib/api-client";
import { Issue } from "@/lib/types";

interface IssueListProps {
  loading?: boolean;
  issues?: Issue[];
  onSelect?: (issue: Issue) => void; // optional click handler
}

const IssueList: React.FC<IssueListProps> = ({ loading = false, issues, onSelect }) => {
  const [items, setItems] = useState<Issue[]>(issues ?? []);
  const [isLoading, setIsLoading] = useState<boolean>(loading && !issues);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (issues) {
      setItems(issues);
      setIsLoading(loading);
      return;
    }

    let cancelled = false;

    async function load() {
      try {
        setIsLoading(true);
        const { items } = await fetchIssues({ page: 1, pageSize: 50 });
        if (!cancelled) {
          setItems(items);
        }
      } catch (e) {
        if (!cancelled) {
          setError("Failed to load issues from the API.");
        }
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, [issues, loading]);

  if (error) {
    return (
      <div className="rounded-md border border-red-500/50 bg-red-950/40 p-4 text-sm text-red-100">
        {error}
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="flex justify-center py-12">
        <Loader />
      </div>
    );
  }

  if (!items || items.length === 0) {
    return (
      <EmptyState
        title="No Issues Found"
        description="Run security checks or sync integrations to populate issues."
      />
    );
  }

  return (
    <div className="grid grid-cols-1 gap-4">
      {items.map((issue) => (
        <IssueCard
          key={issue.id}
          issue={issue}
          onClick={() => onSelect?.(issue)}
        />
      ))}
    </div>
  );
};

export { IssueList };
export default IssueList;

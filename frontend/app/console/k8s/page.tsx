"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api-client";
import K8sIssueCard from "@/components/k8s/K8sIssueCard";

export default function K8sPage() {
  const [issues, setIssues] = useState([]);

  useEffect(() => {
    api.k8s.list().then(setIssues);
  }, []);

  return (
    <div className="p-6 space-y-4">
      <h1 className="text-xl font-semibold">Kubernetes Issues</h1>

      <div className="grid gap-4">
        {issues.map((issue: any) => (
          <K8sIssueCard key={issue.id} issue={issue} />
        ))}
      </div>
    </div>
  );
}

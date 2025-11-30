import { Suspense } from "react";
import { IssueDetailHeader } from "@/components/issues/IssueDetailHeader";
import { IssueDetailTabs } from "@/components/issues/IssueDetailTabs";
import { Card } from "@/components/shared/Card";

interface IssueDetailPageProps {
  params: { id: string };
}

export default function IssueDetailPage({ params }: IssueDetailPageProps) {
  const { id } = params;

  return (
    <main className="flex min-h-screen flex-col bg-neutral-950 text-neutral-50">
      <div className="mx-auto flex w-full max-w-5xl flex-1 flex-col gap-4 px-4 py-6">
        <Suspense fallback={<div className="text-xs text-neutral-500">Loading issue…</div>}>
          <IssueDetailHeader issueId={id} />
        </Suspense>
        <Card>
          <Suspense fallback={<div className="text-xs text-neutral-500">Loading details…</div>}>
            <IssueDetailTabs issueId={id} />
          </Suspense>
        </Card>
      </div>
    </main>
  );
}

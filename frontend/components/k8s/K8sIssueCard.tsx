import HealButton from "./HealButton";

export default function K8sIssueCard({ issue }: { issue: any }) {
  return (
    <div className="border rounded p-4 space-y-2 bg-white shadow">
      <div className="font-semibold">{issue.title}</div>
      <div className="text-sm text-gray-600">{issue.description}</div>
      <HealButton issueId={issue.id} />
    </div>
  );
}

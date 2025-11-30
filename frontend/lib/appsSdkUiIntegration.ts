/**
 * Lightweight compatibility helpers for the OpenAI Apps SDK UI
 * (https://github.com/openai/apps-sdk-ui).
 *
 * The library ships React components for conversations, run logs, and
 * streaming token updates. This module keeps the SecOps console agnostic:
 *  - describes the components we expect to consume,
 *  - provides mapping helpers from SecOps issue/check data to UI-friendly
 *    structures,
 *  - documents how to progressively enhance the console without creating a
 *    hard dependency on the SDK yet.
 */

export type AppsSdkUiComponent = {
  name: string;
  description: string;
  requiredProps: string[];
  optionalProps?: string[];
};

export const recommendedAppsSdkComponents: AppsSdkUiComponent[] = [
  {
    name: "ChatPanel",
    description:
      "Streaming conversation surface for operator <> agent collaboration, ideal for SecOps runbooks.",
    requiredProps: ["messages", "onSend"],
    optionalProps: ["status", "actions", "avatar"],
  },
  {
    name: "RunSummary",
    description: "Compact summary of an app run, perfect for check history or remediation rollups.",
    requiredProps: ["title", "metadata"],
    optionalProps: ["footnotes", "actions"],
  },
  {
    name: "TokenStream",
    description: "Live token stream visualizer; can be wired to the backend's SSE endpoints.",
    requiredProps: ["chunks"],
    optionalProps: ["onChunk", "className"],
  },
];

export type AppsSdkUiBlueprint = {
  packageName: string;
  versionHint: string;
  components: AppsSdkUiComponent[];
  note: string;
};

export function buildAppsSdkBlueprint(): AppsSdkUiBlueprint {
  return {
    packageName: "@openai/apps-sdk-ui",
    versionHint: "^0.2.x",
    components: recommendedAppsSdkComponents,
    note:
      "Install the package, wrap SecOps data into the props above, and render components on the console pages.",
  };
}

export type IssueLike = {
  id: string;
  title: string;
  severity: string;
  summary: string;
  guidance: string;
};

export function mapIssueToChatMessage(issue: IssueLike) {
  return {
    role: "assistant",
    content: `${issue.title} (severity: ${issue.severity}) – ${issue.summary}. Guidance: ${issue.guidance}`,
    metadata: { issueId: issue.id, severity: issue.severity },
  };
}

export function mapIssuesToRunSummary(issues: IssueLike[]) {
  return {
    title: "Latest SecOps Findings",
    metadata: issues.map((issue) => ({
      label: issue.title,
      value: issue.severity,
    })),
    footnotes: [
      "Components mirror the Apps SDK UI contract. Swap in the real library to enable interactive shells.",
      "Data mapping is isolated here so console pages can stay framework-neutral.",
    ],
  };
}

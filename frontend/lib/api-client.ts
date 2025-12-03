// frontend/lib/api-client.ts

import type {
  Issue,
  IssueDetail,
  IssuesTrendPoint,
  CheckStatusSlice,
  CostBreakdownItem,
  ActivityItem,
  AdminUser,
  AdminTeam,
  AuditLogEntry,
} from "./types";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/+$/, "") || "http://localhost:8000";

export class ApiError extends Error {
  status: number;
  url: string;
  body: unknown;

  constructor(message: string, status: number, url: string, body: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.url = url;
    this.body = body;
  }
}

export interface RequestOptions {
  accessToken?: string | null;
  init?: RequestInit;
}

async function request<T>(
  path: string,
  { accessToken, init }: RequestOptions = {}
): Promise<T> {
  const url = `${API_BASE_URL}${path}`;

  const headers: HeadersInit = {
    "Content-Type": "application/json",
    ...(init?.headers || {}),
  };

  if (accessToken) {
    headers["Authorization"] = `Bearer ${accessToken}`;
  }

  const response = await fetch(url, {
    ...init,
    headers,
  });

  const text = await response.text();
  let data: unknown = null;

  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = text;
    }
  }

  if (!response.ok) {
    throw new ApiError(
      `Request to ${url} failed with status ${response.status}`,
      response.status,
      url,
      data
    );
  }

  return data as T;
}

export interface FetchIssuesParams {
  page?: number;
  pageSize?: number;
  status?: string;
  severity?: string;
}

export async function fetchPlatformSummary(options?: RequestOptions) {
  return request<{ status: string; issues_open: number; checks_failing: number }>(
    "/api/platform/summary",
    options
  );
}

export async function fetchIssues(
  params: FetchIssuesParams = {},
  options?: RequestOptions
): Promise<{ items: Issue[]; total: number }> {
  const query = new URLSearchParams();
  if (params.page) query.set("page", String(params.page));
  if (params.pageSize) query.set("page_size", String(params.pageSize));
  if (params.status) query.set("status", params.status);
  if (params.severity) query.set("severity", params.severity);

  const qs = query.toString();
  const path = qs ? `/api/issues?${qs}` : "/api/issues";

  return request<{ items: Issue[]; total: number }>(path, options);
}

export async function fetchIssueById(
  id: string,
  options?: RequestOptions
): Promise<IssueDetail> {
  return request<IssueDetail>(`/api/issues/${encodeURIComponent(id)}`, options);
}

export async function fetchIssuesTrend(
  options?: RequestOptions
): Promise<IssuesTrendPoint[]> {
  return request<IssuesTrendPoint[]>("/api/platform/issues/trend", options);
}

export async function fetchChecksStatus(
  options?: RequestOptions
): Promise<CheckStatusSlice[]> {
  return request<CheckStatusSlice[]>("/api/platform/checks/status", options);
}

export async function fetchCostBreakdown(
  options?: RequestOptions
): Promise<CostBreakdownItem[]> {
  return request<CostBreakdownItem[]>("/api/platform/costs/breakdown", options);
}

export interface RagQueryPayload {
  question: string;
  intent?: string;
  context?: Record<string, unknown>;
  debug?: boolean;
}

export interface RagQueryResponse {
  answer: string;
  intent: string;
  mode: string;
  citations: Array<{
    id?: string;
    title?: string;
    url?: string;
    snippet?: string;
    source_type?: string;
  }>;
  latency_ms?: number;
  status: string;
  error_message?: string | null;
}

export async function queryRag(
  payload: RagQueryPayload,
  options?: RequestOptions
): Promise<RagQueryResponse> {
  return request<RagQueryResponse>("/api/rag/query", {
    ...options,
    init: {
      method: "POST",
      body: JSON.stringify(payload),
      ...(options?.init || {}),
    },
  });
}

export interface RunAnalysisPayload {
  repository: string;
  database_url?: string;
  code_path?: string;
}

export type RunAnalysisResponse = Record<string, unknown>;

export async function runAnalysis(
  payload: RunAnalysisPayload,
  options?: RequestOptions
): Promise<RunAnalysisResponse> {
  return request<RunAnalysisResponse>("/analysis/run", {
    ...options,
    init: {
      method: "POST",
      body: JSON.stringify(payload),
      ...(options?.init || {}),
    },
  });
}

export async function fetchActivity(options?: RequestOptions) {
  return request<ActivityItem[]>("/api/activity", options);
}

export async function fetchAdminUsers(options?: RequestOptions) {
  return request<AdminUser[]>("/api/admin/users", options);
}

export async function fetchAdminTeams(options?: RequestOptions) {
  return request<AdminTeam[]>("/api/admin/teams", options);
}

export async function fetchAuditLogs(options?: RequestOptions) {
  return request<AuditLogEntry[]>("/api/admin/audit", options);
}

export const api = {
  activity: {
    list: fetchActivity,
  },
  admin: {
    users: fetchAdminUsers,
    teams: fetchAdminTeams,
    audit: fetchAuditLogs,
  },
};

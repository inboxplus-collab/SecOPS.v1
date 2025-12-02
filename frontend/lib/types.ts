// frontend/lib/types.ts

/* ============================================================================
 *  Shared primitives / aliases
 * ========================================================================== */

export type ISODateString = string;

/* ============================================================================
 *  Auth
 * ========================================================================== */

export interface AuthLoginRequest {
  email: string;
  password: string;
}

export interface AuthUser {
  id: string;
  org_id: string;
  email: string;
  full_name?: string | null;
  is_superuser: boolean;
}

export interface AuthLoginResponse {
  access_token: string;
  token_type: "bearer";
  user: AuthUser;
}

/* ============================================================================
 *  Issues
 * ========================================================================== */

export type IssueSeverity = "info" | "low" | "medium" | "high" | "critical";
export type IssueStatus =
  | "open"
  | "acknowledged"
  | "resolved"
  | "ignored"
  | "in_progress"
  | "suppressed";

export type IssueResolutionState = "resolved" | "suppressed" | "acknowledged";

export interface IssueLocation {
  kind?: string | null;
  repo?: string | null;
  file_path?: string | null;
  line?: number | null;
  cluster?: string | null;
  namespace?: string | null;
  resource_kind?: string | null;
  resource_name?: string | null;
  environment?: string | null;
  metadata?: Record<string, any> | null;
}

export interface Issue {
  id: string;
  org_id: string;

  title: string;
  description?: string | null;

  severity: IssueSeverity;
  status: IssueStatus;

  source?: string | null; // github | k8s | ci | scanner | rag | manual
  check_name?: string | null;
  code?: string | null; // rule / check id

  target?: string | null; // repo, cluster, service, env, etc.
  external_url?: string | null;
  tags?: string[];
  location?: IssueLocation | null;
  metadata?: Record<string, any> | null;

  created_at: ISODateString;
  updated_at: ISODateString;
  first_seen_at?: ISODateString | null;
  last_seen_at?: ISODateString | null;
  resolved_at?: ISODateString | null;
}

export interface IssueDetail extends Issue {
  root_cause?: string | null;
  impact?: string | null;
  proposed_fix?: string | null;
  precautions?: string | null;
  references?: string[];
  extra?: Record<string, any>;
  resolved_by?: string | null;
  resolution_state?: IssueResolutionState | null;
  resolution_note?: string | null;
}

export interface IssuesTrendPoint {
  date: string;
  count: number;
}

export interface CheckStatusSlice {
  status: string;
  count: number;
}

export interface CostBreakdownItem {
  category: string;
  amount: number;
}

/* ============================================================================
 *  RAG / AI query
 * ========================================================================== */

export interface RAGQueryRequest {
  query: string;
  top_k?: number;
  org_id?: string;
  project_id?: string;
  use_live_search?: boolean;
}

export interface RAGSource {
  id: string;
  title?: string;
  url?: string;
  snippet?: string;
  score?: number;
  metadata?: Record<string, any>;
}

export interface RAGTokenUsage {
  prompt: number;
  completion: number;
  total: number;
}

export interface RAGQueryMeta {
  model?: string;
  latency_ms?: number;
  token_usage?: RAGTokenUsage;
}

export interface RAGQueryResponse {
  answer: string;
  reasoning?: string;
  sources: RAGSource[];
  meta?: RAGQueryMeta;
}

/* ============================================================================
 *  Platform / health
 * ========================================================================== */

export type HealthStatus = "ok" | "degraded" | "error";

export interface PlatformHealthCheck {
  name: string;                // e.g. "database", "github", "rag"
  status: HealthStatus;        // ok | degraded | error
  details?: string;
}

export interface PlatformHealthResponse {
  status: HealthStatus;
  checks: PlatformHealthCheck[];
  timestamp: ISODateString;
  version?: string;
}

/* ============================================================================
 *  Generic API helpers (optional but handy)
 * ========================================================================== */

export interface ApiErrorShape {
  detail?: string | string[] | Record<string, any>;
}

/**
 * Shape for simple "success" style responses from mutation endpoints.
 */
export interface SimpleSuccessResponse {
  success: boolean;
  message?: string;
}

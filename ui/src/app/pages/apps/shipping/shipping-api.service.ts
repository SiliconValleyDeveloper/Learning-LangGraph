import { Injectable } from "@angular/core";
import { HttpClient } from "@angular/common/http";
import { Observable } from "rxjs";

import { environment } from "../../../../environments/environment";

export interface ShippingHealth {
  status: string;
  postgres_ok: boolean;
  project?: string;
  human_approval?: boolean;
  customers?: number;
  ports?: number;
  error?: string;
}

export interface ShippingGraphNode {
  id: string;
  label: string;
  kind: string;
}

export interface ShippingGraphEdge {
  source: string;
  target: string;
  label: string;
}

export interface ShippingGraph {
  nodes: ShippingGraphNode[];
  edges: ShippingGraphEdge[];
  mermaid: string;
  agents: string[];
  mcp_tools: string[];
  human_approval_before: string[];
  lanes?: string[];
  max_retrieval_retries?: number;
  max_fix_attempts?: number;
}

export interface ShippingTraceStep {
  agent: string;
  summary: string;
  [key: string]: unknown;
}

export interface ShippingApproval {
  approval_id: string;
  thread_id?: string;
  status: string;
  action: string;
  proposal: Record<string, unknown>;
  risk_review: Record<string, unknown>;
  reviewer?: string;
  note?: string;
}

export interface ShippingRunState {
  action: string;
  plan: string[];
  parameters: Record<string, unknown>;
  proposal: Record<string, unknown>;
  risk_review: {
    risk_level?: string;
    warnings?: string[];
    hard_blocks?: string[];
    human_approval_required?: boolean;
  };
  status: string;
  errors: string[];
  lane?: string;
  route_reason?: string;
  rewritten_query?: string;
  evidence_grade?: string;
  evidence_score?: number;
  rerank_backend?: string;
  retrieval_attempts?: number;
  verified?: boolean;
  verification_issues?: string[];
  fix_attempts?: number;
}

export interface ShippingEvidence {
  citation?: string;
  source_type?: string;
  source_id?: string;
  title?: string;
  score?: number;
  content?: string;
}

export interface ShippingChoice {
  kind: "customer" | "sailing" | "route";
  label: string;
  value: string;
}

export interface ShippingResponse {
  thread_id: string;
  action: string;
  status: string;
  answer: string;
  data: unknown;
  errors: string[];
  approval?: ShippingApproval | null;
  verified?: boolean | null;
  evidence_grade?: string;
  citations?: Array<Record<string, unknown>>;
}

export interface ShippingRunResult {
  thread_id: string;
  interrupted: boolean;
  assistant_message: string;
  pending: ShippingApproval | null;
  response: ShippingResponse | null;
  state: ShippingRunState;
  trace: ShippingTraceStep[];
  graph: ShippingGraph;
  evidence?: ShippingEvidence[];
  choices?: ShippingChoice[];
}

@Injectable({ providedIn: "root" })
export class ShippingApiService {
  private readonly base = `${environment.shippingApiUrl}/api/shipping`;

  constructor(private readonly http: HttpClient) {}

  health(): Observable<ShippingHealth> {
    return this.http.get<ShippingHealth>(`${this.base}/health`);
  }

  graph(): Observable<ShippingGraph> {
    return this.http.get<ShippingGraph>(`${this.base}/graph`);
  }

  run(prompt: string): Observable<ShippingRunResult> {
    return this.http.post<ShippingRunResult>(`${this.base}/run`, { prompt });
  }

  decide(
    threadId: string,
    approve: boolean,
    reviewer: string,
    note: string,
  ): Observable<ShippingRunResult> {
    return this.http.post<ShippingRunResult>(`${this.base}/approve`, {
      thread_id: threadId,
      approve,
      reviewer,
      note,
    });
  }
}

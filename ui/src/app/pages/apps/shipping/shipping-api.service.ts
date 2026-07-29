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

export type ShippingChoiceKind =
  | "customer"
  | "sailing"
  | "port"
  | "route"
  | "booking_ref"
  | "quote_ref"
  | "container_type"
  | "entity"
  | "status"
  | "field_value"
  | "dismiss";

export interface ShippingChoice {
  kind: ShippingChoiceKind | string;
  field?: string;
  label: string;
  value: string;
  reason?: string;
}

export interface ShippingRecoveryGroup {
  field: string;
  title: string;
  choices: ShippingChoice[];
}

export interface ShippingRecovery {
  active: boolean;
  action: string;
  filled: Record<string, unknown>;
  missing_fields: string[];
  invalid_fields: string[];
  errors: string[];
  groups: ShippingRecoveryGroup[];
  choices: ShippingChoice[];
  message?: string;
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
  recovery?: ShippingRecovery;
}

export interface ShippingChatTurn {
  role: "user" | "assistant" | "system";
  content: string;
}

export interface ShippingRunRequest {
  prompt: string;
  thread_id?: string | null;
  patches?: Record<string, unknown>;
  base_prompt?: string | null;
  history?: ShippingChatTurn[];
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

  run(request: ShippingRunRequest | string): Observable<ShippingRunResult> {
    const body =
      typeof request === "string"
        ? { prompt: request, history: [] as ShippingChatTurn[] }
        : {
            prompt: request.prompt,
            thread_id: request.thread_id || undefined,
            patches: request.patches || {},
            base_prompt: request.base_prompt || undefined,
            history: request.history || [],
          };
    return this.http.post<ShippingRunResult>(`${this.base}/run`, body);
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

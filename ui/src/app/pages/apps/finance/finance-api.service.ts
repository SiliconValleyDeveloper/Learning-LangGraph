import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';

import { environment } from '../../../../environments/environment';

const API_KEY_STORAGE = 'finance_api_key';

export interface FinanceQuote {
  symbol: string;
  exchange: string;
  ltp: number;
  volume?: number | null;
  oi?: number | null;
  bid?: number | null;
  ask?: number | null;
  change_pct?: number | null;
  name?: string | null;
  source: string;
  ts?: string | null;
}

export interface FinanceStatus {
  enabled: boolean;
  product: string;
  broker: boolean;
  orders_supported: boolean;
  api_auth_required: boolean;
  auth?: {
    active_consumers?: number | null;
    create_cli?: string;
  };
  quote?: {
    resolved_source?: string;
    kite_credentials_ready?: boolean;
    ticks_total?: number | null;
    redis?: { ok?: boolean };
  };
  ingest?: {
    companies?: number;
    corp_actions?: number;
    fundamentals?: number;
    last_run?: { status?: string; id?: number } | null;
  };
  agent?: {
    phase?: string;
    levels?: string[];
    analysis_only?: boolean;
    orders_supported?: boolean;
    required_scope?: string;
  };
  redis?: { ok?: boolean };
}

export interface CompanyRow {
  symbol: string;
  exchange: string;
  isin?: string | null;
  name: string;
  series?: string | null;
  status?: string;
  sector?: string | null;
  industry?: string | null;
}

export interface CorpActionRow {
  symbol: string;
  exchange: string;
  action_type: string;
  ex_date?: string | null;
  record_date?: string | null;
  ratio?: string | null;
  amount?: number | null;
  currency?: string | null;
  source?: string;
}

export interface FundamentalLine {
  symbol: string;
  exchange: string;
  period_type: string;
  period: string;
  statement: string;
  line_item: string;
  value: number | null;
  currency: string;
  unit: string | null;
  source: string;
  fetched_at?: string;
}

export interface FundamentalPeriod {
  period_type: string;
  period: string;
}

export interface FundamentalsResponse {
  symbol: string;
  count: number;
  available_periods: FundamentalPeriod[];
  lines: FundamentalLine[];
  by_period: Record<string, Record<string, FundamentalLine[]>>;
}

export interface FilingDocument {
  id: string;
  symbol: string;
  exchange: string;
  doc_type: string;
  title: string;
  filename: string;
  source: string;
  fetched_at?: string | null;
  chunk_count: number;
}

export interface FilingChunkHit {
  rank?: number;
  chunk_id: string;
  document_id: string;
  source: string;
  title?: string;
  symbol?: string;
  doc_type?: string;
  content: string;
  preview: string;
  vector_score?: number;
  rerank_score?: number;
}

export interface FilingsSearchResponse {
  query: string;
  symbol: string | null;
  embed_backend: string;
  rerank_backend: string;
  retrieve_candidates: number;
  count: number;
  chunks: FilingChunkHit[];
}

export interface AgentCitation {
  id: string;
  kind: string;
  source: string;
  title?: string | null;
  score?: number | null;
}

export interface AgentToolTrace {
  tool: string;
  status: string;
  items?: number;
  detail?: string;
}

export interface AgentAnalysisResponse {
  symbol: string;
  exchange: string;
  level: 'L1' | 'L2';
  intent: string;
  route_reason: string;
  plan: string[];
  tool_trace: AgentToolTrace[];
  answer: string;
  citations: AgentCitation[];
  verified: boolean;
  refused: boolean;
  analysis_only: boolean;
  orders_supported: boolean;
}

@Injectable({ providedIn: 'root' })
export class FinanceApiService {
  private readonly base = `${environment.apiUrl}/api/finance`;

  constructor(private readonly http: HttpClient) {}

  getApiKey(): string {
    return localStorage.getItem(API_KEY_STORAGE) || '';
  }

  setApiKey(key: string): void {
    const trimmed = key.trim();
    if (trimmed) {
      localStorage.setItem(API_KEY_STORAGE, trimmed);
    } else {
      localStorage.removeItem(API_KEY_STORAGE);
    }
  }

  private headers(): HttpHeaders {
    const key = this.getApiKey();
    let headers = new HttpHeaders();
    if (key) {
      headers = headers.set('X-API-Key', key);
    }
    return headers;
  }

  status(): Observable<FinanceStatus> {
    return this.http.get<FinanceStatus>(`${this.base}/status`);
  }

  quotes(symbols?: string[], refresh = false): Observable<{
    source: string;
    count: number;
    quotes: FinanceQuote[];
  }> {
    let params = new HttpParams().set('refresh', String(refresh));
    if (symbols?.length) {
      params = params.set('symbols', symbols.join(','));
    }
    return this.http.get<{ source: string; count: number; quotes: FinanceQuote[] }>(
      `${this.base}/quotes`,
      { headers: this.headers(), params }
    );
  }

  quote(symbol: string, exchange = 'NSE'): Observable<FinanceQuote> {
    return this.http.get<FinanceQuote>(
      `${this.base}/quote/${encodeURIComponent(symbol)}`,
      {
        headers: this.headers(),
        params: new HttpParams().set('exchange', exchange).set('refresh', 'true')
      }
    );
  }

  companies(exchange?: string): Observable<{ count: number; companies: CompanyRow[] }> {
    let params = new HttpParams();
    if (exchange) {
      params = params.set('exchange', exchange);
    }
    return this.http.get<{ count: number; companies: CompanyRow[] }>(
      `${this.base}/companies`,
      { headers: this.headers(), params }
    );
  }

  corpActions(symbol?: string): Observable<{ count: number; actions: CorpActionRow[] }> {
    let params = new HttpParams();
    if (symbol) {
      params = params.set('symbol', symbol);
    }
    return this.http.get<{ count: number; actions: CorpActionRow[] }>(
      `${this.base}/corp-actions`,
      { headers: this.headers(), params }
    );
  }

  fundamentals(
    symbol: string,
    opts?: {
      exchange?: string;
      period_type?: string;
      period?: string;
      statement?: string;
    }
  ): Observable<FundamentalsResponse> {
    let params = new HttpParams();
    if (opts?.exchange) {
      params = params.set('exchange', opts.exchange);
    }
    if (opts?.period_type) {
      params = params.set('period_type', opts.period_type);
    }
    if (opts?.period) {
      params = params.set('period', opts.period);
    }
    if (opts?.statement) {
      params = params.set('statement', opts.statement);
    }
    return this.http.get<FundamentalsResponse>(
      `${this.base}/fundamentals/${encodeURIComponent(symbol)}`,
      { headers: this.headers(), params }
    );
  }

  filings(symbol?: string): Observable<{ count: number; documents: FilingDocument[] }> {
    let params = new HttpParams();
    if (symbol) {
      params = params.set('symbol', symbol);
    }
    return this.http.get<{ count: number; documents: FilingDocument[] }>(
      `${this.base}/filings`,
      { headers: this.headers(), params }
    );
  }

  searchFilings(symbol: string, query: string): Observable<FilingsSearchResponse> {
    return this.http.get<FilingsSearchResponse>(
      `${this.base}/filings/${encodeURIComponent(symbol)}/search`,
      {
        headers: this.headers(),
        params: new HttpParams().set('q', query)
      }
    );
  }

  analyse(
    question: string,
    opts?: { symbol?: string; exchange?: string; level?: 'L1' | 'L2' }
  ): Observable<AgentAnalysisResponse> {
    return this.http.post<AgentAnalysisResponse>(
      `${this.base}/agent/analyse`,
      {
        question,
        symbol: opts?.symbol || null,
        exchange: opts?.exchange || 'NSE',
        level: opts?.level || 'L2'
      },
      { headers: this.headers() }
    );
  }
}

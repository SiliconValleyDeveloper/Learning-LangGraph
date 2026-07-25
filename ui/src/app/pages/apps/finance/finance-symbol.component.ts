import { Component, OnInit } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { HttpErrorResponse } from '@angular/common/http';

import {
  CorpActionRow,
  FilingChunkHit,
  FilingDocument,
  FinanceApiService,
  FinanceQuote,
  FundamentalLine,
  FundamentalPeriod,
  FundamentalsResponse,
  FilingsSearchResponse
} from './finance-api.service';

type SymbolTab = 'overview' | 'fundamentals' | 'filings' | 'actions' | 'agent';

const STATEMENT_ORDER = ['income_statement', 'balance_sheet', 'cash_flow'] as const;

const STATEMENT_LABELS: Record<string, string> = {
  income_statement: 'Income statement',
  balance_sheet: 'Balance sheet',
  cash_flow: 'Cash flow'
};

@Component({
  selector: 'app-finance-symbol',
  templateUrl: './finance-symbol.component.html',
  styleUrls: ['./finance-symbol.component.scss'],
  standalone: false
})
export class FinanceSymbolComponent implements OnInit {
  symbol = '';
  exchange = 'NSE';
  tab: SymbolTab = 'overview';
  quote: FinanceQuote | null = null;
  actions: CorpActionRow[] = [];
  error = '';
  loading = false;

  fundamentalsLoading = false;
  fundamentalsError = '';
  fundamentals: FundamentalsResponse | null = null;
  selectedPeriodKey = '';
  statementBlocks: { key: string; label: string; lines: FundamentalLine[] }[] = [];

  filingsLoading = false;
  filingsError = '';
  filingDocs: FilingDocument[] = [];
  filingQuery = 'What did management say about debt?';
  filingSearchLoading = false;
  filingSearchError = '';
  filingSearch: FilingsSearchResponse | null = null;

  readonly tabs: { id: SymbolTab; label: string; ready: boolean }[] = [
    { id: 'overview', label: 'Overview', ready: true },
    { id: 'fundamentals', label: 'Fundamentals', ready: true },
    { id: 'filings', label: 'Filings', ready: true },
    { id: 'actions', label: 'Actions', ready: true },
    { id: 'agent', label: 'Agent', ready: true }
  ];

  constructor(
    private readonly route: ActivatedRoute,
    private readonly financeApi: FinanceApiService
  ) {}

  ngOnInit(): void {
    this.route.paramMap.subscribe(params => {
      this.symbol = decodeURIComponent(params.get('symbol') || '').toUpperCase();
      this.exchange = (this.route.snapshot.queryParamMap.get('exchange') || 'NSE').toUpperCase();
      this.fundamentals = null;
      this.selectedPeriodKey = '';
      this.statementBlocks = [];
      this.fundamentalsError = '';
      this.filingDocs = [];
      this.filingSearch = null;
      this.filingsError = '';
      this.filingSearchError = '';
      this.load();
      if (this.tab === 'fundamentals') {
        this.loadFundamentals();
      }
      if (this.tab === 'filings') {
        this.loadFilings();
      }
    });
  }

  setTab(tab: SymbolTab): void {
    this.tab = tab;
    if (tab === 'fundamentals' && !this.fundamentals && !this.fundamentalsLoading) {
      this.loadFundamentals();
    }
    if (tab === 'filings' && !this.filingDocs.length && !this.filingsLoading) {
      this.loadFilings();
    }
  }

  periodLabel(p: FundamentalPeriod): string {
    return `${p.period_type === 'annual' ? 'Annual' : 'Quarterly'} · ${p.period}`;
  }

  periodKey(p: FundamentalPeriod): string {
    return `${p.period_type}:${p.period}`;
  }

  selectPeriod(key: string): void {
    this.selectedPeriodKey = key;
    this.rebuildStatementBlocks();
  }

  load(): void {
    if (!this.symbol) {
      return;
    }
    this.loading = true;
    this.error = '';
    this.financeApi.quote(this.symbol, this.exchange).subscribe({
      next: quote => {
        this.quote = quote;
        this.loading = false;
      },
      error: (err: HttpErrorResponse) => {
        this.loading = false;
        this.error = err.error?.detail || err.message;
      }
    });
    this.financeApi.corpActions(this.symbol).subscribe({
      next: res => {
        this.actions = res.actions;
      },
      error: () => {
        this.actions = [];
      }
    });
  }

  loadFundamentals(): void {
    if (!this.symbol) {
      return;
    }
    this.fundamentalsLoading = true;
    this.fundamentalsError = '';
    this.financeApi.fundamentals(this.symbol, { exchange: this.exchange }).subscribe({
      next: res => {
        this.fundamentals = res;
        this.fundamentalsLoading = false;
        const first = res.available_periods[0];
        this.selectedPeriodKey = first ? this.periodKey(first) : '';
        this.rebuildStatementBlocks();
      },
      error: (err: HttpErrorResponse) => {
        this.fundamentalsLoading = false;
        this.fundamentals = null;
        this.statementBlocks = [];
        this.fundamentalsError = err.error?.detail || err.message;
      }
    });
  }

  loadFilings(): void {
    if (!this.symbol) {
      return;
    }
    this.filingsLoading = true;
    this.filingsError = '';
    this.financeApi.filings(this.symbol).subscribe({
      next: res => {
        this.filingDocs = res.documents;
        this.filingsLoading = false;
      },
      error: (err: HttpErrorResponse) => {
        this.filingsLoading = false;
        this.filingDocs = [];
        this.filingsError = err.error?.detail || err.message;
      }
    });
  }

  searchFilings(): void {
    const q = this.filingQuery.trim();
    if (!this.symbol || !q) {
      return;
    }
    this.filingSearchLoading = true;
    this.filingSearchError = '';
    this.financeApi.searchFilings(this.symbol, q).subscribe({
      next: res => {
        this.filingSearch = res;
        this.filingSearchLoading = false;
      },
      error: (err: HttpErrorResponse) => {
        this.filingSearchLoading = false;
        this.filingSearch = null;
        this.filingSearchError = err.error?.detail || err.message;
      }
    });
  }

  trackChunk(chunk: FilingChunkHit): string {
    return chunk.chunk_id;
  }

  private rebuildStatementBlocks(): void {
    if (!this.fundamentals || !this.selectedPeriodKey) {
      this.statementBlocks = [];
      return;
    }
    const byStatement = this.fundamentals.by_period[this.selectedPeriodKey] || {};
    this.statementBlocks = STATEMENT_ORDER.filter(key => byStatement[key]?.length).map(key => ({
      key,
      label: STATEMENT_LABELS[key] || key,
      lines: byStatement[key]
    }));
  }
}

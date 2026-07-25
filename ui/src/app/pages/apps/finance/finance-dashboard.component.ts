import { Component, OnInit } from '@angular/core';
import { FormBuilder } from '@angular/forms';
import { HttpErrorResponse } from '@angular/common/http';

import {
  FinanceApiService,
  FinanceQuote,
  FinanceStatus
} from './finance-api.service';

interface TileGroup {
  id: string;
  title: string;
  symbols: string[];
}

@Component({
  selector: 'app-finance-dashboard',
  templateUrl: './finance-dashboard.component.html',
  styleUrls: ['./finance-dashboard.component.scss'],
  standalone: false
})
export class FinanceDashboardComponent implements OnInit {
  status: FinanceStatus | null = null;
  quotes: FinanceQuote[] = [];
  quoteMap = new Map<string, FinanceQuote>();
  error = '';
  loading = false;
  isOnline = false;

  readonly keyForm = this.fb.nonNullable.group({
    apiKey: ''
  });

  readonly groups: TileGroup[] = [
    { id: 'indices', title: 'Indices', symbols: ['NIFTY 50', 'SENSEX'] },
    {
      id: 'stocks',
      title: 'India stocks',
      symbols: ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 'SBIN']
    },
    { id: 'commodities', title: 'Commodities', symbols: ['GOLD', 'CRUDEOIL'] },
    { id: 'fx', title: 'FX', symbols: ['USDINR'] }
  ];

  private readonly allSymbols = this.groups.flatMap(g => g.symbols);

  constructor(
    private readonly financeApi: FinanceApiService,
    private readonly fb: FormBuilder
  ) {}

  ngOnInit(): void {
    this.keyForm.patchValue({ apiKey: this.financeApi.getApiKey() });
    this.loadStatus();
    if (this.financeApi.getApiKey()) {
      this.refreshQuotes();
    }
  }

  saveKey(): void {
    this.financeApi.setApiKey(this.keyForm.controls.apiKey.value);
    this.error = '';
    this.refreshQuotes();
  }

  clearKey(): void {
    this.financeApi.setApiKey('');
    this.keyForm.patchValue({ apiKey: '' });
    this.quotes = [];
    this.quoteMap.clear();
  }

  loadStatus(): void {
    this.financeApi.status().subscribe({
      next: status => {
        this.status = status;
        this.isOnline = true;
      },
      error: () => {
        this.isOnline = false;
        this.status = null;
      }
    });
  }

  refreshQuotes(): void {
    this.loading = true;
    this.error = '';
    this.financeApi.quotes(this.allSymbols, true).subscribe({
      next: res => {
        this.quotes = res.quotes;
        this.quoteMap = new Map(res.quotes.map(q => [q.symbol.toUpperCase(), q]));
        this.loading = false;
        this.loadStatus();
      },
      error: (err: HttpErrorResponse) => {
        this.loading = false;
        this.error = this.formatError(err);
      }
    });
  }

  quoteFor(symbol: string): FinanceQuote | undefined {
    return this.quoteMap.get(symbol.toUpperCase());
  }

  changeClass(q?: FinanceQuote): string {
    if (!q?.change_pct && q?.change_pct !== 0) {
      return '';
    }
    return (q.change_pct ?? 0) >= 0 ? 'up' : 'down';
  }

  symbolLink(symbol: string): string[] {
    return ['/finance/symbol', symbol];
  }

  private formatError(err: HttpErrorResponse): string {
    const detail = err.error?.detail;
    if (err.status === 401) {
      return 'API key required. Create one: python -m projects.finance_agent.consumers create --name demo --tier free';
    }
    if (typeof detail === 'string') {
      return detail;
    }
    return err.message || 'Request failed';
  }
}

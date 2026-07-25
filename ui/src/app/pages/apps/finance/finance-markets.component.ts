import { Component, OnInit } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { HttpErrorResponse } from '@angular/common/http';

import {
  CompanyRow,
  FinanceApiService,
  FinanceQuote
} from './finance-api.service';

interface MarketMeta {
  title: string;
  blurb: string;
  symbols: string[];
  useCompanies: boolean;
}

const MARKETS: Record<string, MarketMeta> = {
  stocks: {
    title: 'Stocks',
    blurb: 'NSE / BSE equities from company master + live quotes.',
    symbols: [],
    useCompanies: true
  },
  indices: {
    title: 'Indices',
    blurb: 'Benchmark indices (sample until live index feed).',
    symbols: ['NIFTY 50', 'SENSEX'],
    useCompanies: false
  },
  commodities: {
    title: 'Commodities',
    blurb: 'Gold, crude, and other commodities (sample quotes).',
    symbols: ['GOLD', 'CRUDEOIL'],
    useCompanies: false
  },
  fx: {
    title: 'FX',
    blurb: 'Currency pairs for macro context.',
    symbols: ['USDINR'],
    useCompanies: false
  }
};

@Component({
  selector: 'app-finance-markets',
  templateUrl: './finance-markets.component.html',
  styleUrls: ['./finance-markets.component.scss'],
  standalone: false
})
export class FinanceMarketsComponent implements OnInit {
  marketId = 'stocks';
  meta: MarketMeta = MARKETS['stocks'];
  companies: CompanyRow[] = [];
  quotes: FinanceQuote[] = [];
  quoteMap = new Map<string, FinanceQuote>();
  error = '';
  loading = false;

  constructor(
    private readonly route: ActivatedRoute,
    private readonly financeApi: FinanceApiService
  ) {}

  ngOnInit(): void {
    this.route.paramMap.subscribe(params => {
      this.marketId = params.get('market') || 'stocks';
      this.meta = MARKETS[this.marketId] || MARKETS['stocks'];
      this.load();
    });
  }

  load(): void {
    this.loading = true;
    this.error = '';
    if (this.meta.useCompanies) {
      this.financeApi.companies().subscribe({
        next: res => {
          this.companies = res.companies;
          const symbols = [...new Set(res.companies.map(c => c.symbol))];
          this.loadQuotes(symbols);
        },
        error: (err: HttpErrorResponse) => {
          this.loading = false;
          this.error = err.error?.detail || err.message;
        }
      });
    } else {
      this.companies = [];
      this.loadQuotes(this.meta.symbols);
    }
  }

  private loadQuotes(symbols: string[]): void {
    if (!symbols.length) {
      this.loading = false;
      return;
    }
    this.financeApi.quotes(symbols, true).subscribe({
      next: res => {
        this.quotes = res.quotes;
        this.quoteMap = new Map(res.quotes.map(q => [q.symbol.toUpperCase(), q]));
        this.loading = false;
      },
      error: (err: HttpErrorResponse) => {
        this.loading = false;
        this.error = err.error?.detail || err.message;
      }
    });
  }

  quoteFor(symbol: string): FinanceQuote | undefined {
    return this.quoteMap.get(symbol.toUpperCase());
  }

  rows(): { symbol: string; name: string; exchange: string; sector?: string | null }[] {
    if (this.meta.useCompanies) {
      return this.companies.map(c => ({
        symbol: c.symbol,
        name: c.name,
        exchange: c.exchange,
        sector: c.sector
      }));
    }
    return this.meta.symbols.map(symbol => ({
      symbol,
      name: this.quoteFor(symbol)?.name || symbol,
      exchange: this.quoteFor(symbol)?.exchange || 'NSE',
      sector: null
    }));
  }
}

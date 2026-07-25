import { Component, OnInit } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { HttpErrorResponse } from '@angular/common/http';

import {
  AgentAnalysisResponse,
  FinanceApiService
} from './finance-api.service';

@Component({
  selector: 'app-finance-agent',
  templateUrl: './finance-agent.component.html',
  styleUrls: ['./finance-agent.component.scss'],
  standalone: false
})
export class FinanceAgentComponent implements OnInit {
  symbol = 'RELIANCE';
  exchange = 'NSE';
  level: 'L1' | 'L2' = 'L2';
  question = 'Give me a cited analysis brief covering fundamentals, filings, and key risks.';
  loading = false;
  error = '';
  result: AgentAnalysisResponse | null = null;

  constructor(
    private readonly route: ActivatedRoute,
    private readonly financeApi: FinanceApiService
  ) {}

  ngOnInit(): void {
    const symbol = this.route.snapshot.queryParamMap.get('symbol');
    const question = this.route.snapshot.queryParamMap.get('question');
    if (symbol) {
      this.symbol = symbol.toUpperCase();
    }
    if (question) {
      this.question = question;
    }
  }

  run(): void {
    const question = this.question.trim();
    const symbol = this.symbol.trim().toUpperCase();
    if (!question || !symbol) {
      return;
    }
    this.loading = true;
    this.error = '';
    this.result = null;
    this.financeApi.analyse(question, {
      symbol,
      exchange: this.exchange,
      level: this.level
    }).subscribe({
      next: result => {
        this.result = result;
        this.loading = false;
      },
      error: (err: HttpErrorResponse) => {
        this.loading = false;
        this.error = err.error?.detail || err.message;
      }
    });
  }
}

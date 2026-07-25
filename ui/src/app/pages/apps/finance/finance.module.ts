import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { RouterModule, Routes } from '@angular/router';

import { FinanceAgentComponent } from './finance-agent.component';
import { FinanceDashboardComponent } from './finance-dashboard.component';
import { FinanceMarketsComponent } from './finance-markets.component';
import { FinancePlaceholderComponent } from './finance-placeholder.component';
import { FinanceShellComponent } from './finance-shell.component';
import { FinanceSymbolComponent } from './finance-symbol.component';

const routes: Routes = [
  {
    path: '',
    component: FinanceShellComponent,
    children: [
      { path: '', component: FinanceDashboardComponent },
      { path: 'markets/:market', component: FinanceMarketsComponent },
      { path: 'symbol/:symbol', component: FinanceSymbolComponent },
      { path: 'watchlist', component: FinancePlaceholderComponent },
      { path: 'agent', component: FinanceAgentComponent },
      { path: 'research', component: FinancePlaceholderComponent }
    ]
  }
];

@NgModule({
  declarations: [
    FinanceAgentComponent,
    FinanceShellComponent,
    FinanceDashboardComponent,
    FinanceMarketsComponent,
    FinanceSymbolComponent,
    FinancePlaceholderComponent
  ],
  imports: [CommonModule, FormsModule, ReactiveFormsModule, RouterModule.forChild(routes)]
})
export class FinanceModule {}

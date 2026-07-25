import { Component, OnInit } from '@angular/core';

import { LabContextService } from '../../../core/lab-context.service';

@Component({
  selector: 'app-finance-shell',
  templateUrl: './finance-shell.component.html',
  styleUrls: ['./finance-shell.component.scss'],
  standalone: false
})
export class FinanceShellComponent implements OnInit {
  constructor(private readonly labContext: LabContextService) {}

  ngOnInit(): void {
    this.labContext.setPage(
      'Finance · Markets Analysis',
      'Quotes · NSE/BSE knowledge · research agent (not a broker)'
    );
  }
}

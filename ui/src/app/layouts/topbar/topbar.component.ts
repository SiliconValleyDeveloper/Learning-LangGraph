import { Component, OnDestroy, OnInit } from '@angular/core';
import { Subscription } from 'rxjs';

import { LabContextService, LabPageContext } from '../../core/lab-context.service';

@Component({
  selector: 'app-topbar',
  templateUrl: './topbar.component.html',
  styleUrls: ['./topbar.component.scss'],
  standalone: false
})
export class TopbarComponent implements OnInit, OnDestroy {
  page: LabPageContext = {
    title: 'LangGraph Lab',
    subtitle: 'Local LangGraph execution console'
  };

  private subscription?: Subscription;

  constructor(private readonly labContext: LabContextService) {}

  ngOnInit(): void {
    this.subscription = this.labContext.page$.subscribe(page => {
      this.page = page;
    });
  }

  ngOnDestroy(): void {
    this.subscription?.unsubscribe();
  }
}

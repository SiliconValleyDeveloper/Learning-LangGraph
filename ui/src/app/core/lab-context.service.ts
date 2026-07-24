import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';

export interface LabPageContext {
  title: string;
  subtitle: string;
}

@Injectable({ providedIn: 'root' })
export class LabContextService {
  private readonly pageSubject = new BehaviorSubject<LabPageContext>({
    title: 'LangGraph Lab',
    subtitle: 'Local LangGraph execution console'
  });

  readonly page$ = this.pageSubject.asObservable();

  setPage(title: string, subtitle = 'Local LangGraph execution console'): void {
    this.pageSubject.next({ title, subtitle });
  }
}

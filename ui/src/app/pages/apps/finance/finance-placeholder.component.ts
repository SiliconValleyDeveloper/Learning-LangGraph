import { Component } from '@angular/core';

@Component({
  selector: 'app-finance-placeholder',
  template: `
    <section class="placeholder">
      <h2>{{ title }}</h2>
      <p>{{ body }}</p>
      <a routerLink="/finance">Back to dashboard</a>
    </section>
  `,
  styles: [
    `
      .placeholder {
        padding: 1rem;
        border: 1px solid rgba(15, 28, 36, 0.12);
        border-radius: 0.75rem;
        background: rgba(248, 251, 252, 0.92);
      }
      h2 {
        margin: 0 0 0.4rem;
        font-family: Sora, Trebuchet MS, sans-serif;
      }
      p {
        color: rgba(15, 28, 36, 0.62);
      }
      a {
        color: #0d6e6e;
        font-weight: 600;
        text-decoration: none;
      }
    `
  ],
  standalone: false
})
export class FinancePlaceholderComponent {
  title = 'Coming soon';
  body = 'This section lands in a later finance phase.';

  constructor() {
    const path = typeof location !== 'undefined' ? location.pathname : '';
    if (path.includes('watchlist')) {
      this.title = 'Watchlist';
      this.body = 'Persisted analysis watchlists arrive in F8.';
    } else if (path.includes('agent')) {
      this.title = 'Agent console';
      this.body = 'L2 analysis briefs and autonomous research goals arrive in F7 / F9. No trading.';
    }
  }
}

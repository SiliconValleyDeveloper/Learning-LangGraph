import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-layout',
  templateUrl: './layout.component.html',
  styleUrls: ['./layout.component.scss'],
  standalone: false
})
export class LayoutComponent implements OnInit {
  ngOnInit(): void {
    const root = document.documentElement;
    root.setAttribute('data-layout', 'vertical');
    root.setAttribute('data-bs-theme', 'light');
    root.setAttribute('data-layout-width', 'fluid');
    root.setAttribute('data-layout-position', 'fixed');
    root.setAttribute('data-topbar', 'light');
    root.setAttribute('data-sidebar', 'dark');
    root.setAttribute('data-sidebar-size', 'lg');
    root.setAttribute('data-sidebar-image', 'none');
    root.setAttribute('data-layout-style', 'default');
    root.setAttribute('data-preloader', 'disable');
    root.setAttribute('data-sidebar-visibility', 'hidden');
  }
}

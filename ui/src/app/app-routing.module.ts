import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';

import { LayoutComponent } from './layouts/layout.component';

const routes: Routes = [
  {
    path: '',
    component: LayoutComponent,
    children: [
      {
        path: 'chat',
        loadChildren: () =>
          import('./pages/apps/apps.module').then(module => module.AppsModule)
      },
      { path: '', pathMatch: 'full', redirectTo: 'chat' }
    ]
  },
  { path: '**', redirectTo: 'chat' }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }

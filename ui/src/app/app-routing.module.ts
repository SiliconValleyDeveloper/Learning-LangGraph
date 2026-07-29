import { NgModule } from "@angular/core";
import { RouterModule, Routes } from "@angular/router";

import { LayoutComponent } from "./layouts/layout.component";

const routes: Routes = [
  {
    path: "",
    component: LayoutComponent,
    children: [
      {
        path: "chat",
        loadChildren: () =>
          import("./pages/apps/apps.module").then(
            (module) => module.AppsModule,
          ),
      },
      {
        path: "finance",
        loadChildren: () =>
          import("./pages/apps/finance/finance.module").then(
            (module) => module.FinanceModule,
          ),
      },
      {
        path: "shipping",
        loadChildren: () =>
          import("./pages/apps/shipping/shipping.module").then(
            (module) => module.ShippingModule,
          ),
      },
      { path: "", pathMatch: "full", redirectTo: "chat" },
    ],
  },
  { path: "**", redirectTo: "chat" },
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule],
})
export class AppRoutingModule {}

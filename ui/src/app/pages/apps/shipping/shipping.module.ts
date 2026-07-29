import { CommonModule } from "@angular/common";
import { NgModule } from "@angular/core";
import { FormsModule } from "@angular/forms";
import { RouterModule, Routes } from "@angular/router";

import { ShippingConsoleComponent } from "./shipping-console.component";

const routes: Routes = [{ path: "", component: ShippingConsoleComponent }];

@NgModule({
  declarations: [ShippingConsoleComponent],
  imports: [CommonModule, FormsModule, RouterModule.forChild(routes)],
})
export class ShippingModule {}

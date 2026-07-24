import { CommonModule } from '@angular/common';
import { NgModule } from '@angular/core';
import { ReactiveFormsModule } from '@angular/forms';

import { AppsRoutingModule } from './apps-routing.module';
import { ChatComponent } from 'src/app/pages/apps/chat/chat.component';
import { DocRagComponent } from 'src/app/pages/apps/doc-rag/doc-rag.component';
import { PersonFinderComponent } from 'src/app/pages/apps/person-finder/person-finder.component';

@NgModule({
  declarations: [ChatComponent, PersonFinderComponent, DocRagComponent],
  imports: [CommonModule, ReactiveFormsModule, AppsRoutingModule]
})
export class AppsModule {}

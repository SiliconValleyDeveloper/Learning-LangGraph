import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';

import { ChatComponent } from './chat/chat.component';
import { DocRagComponent } from './doc-rag/doc-rag.component';
import { PersonFinderComponent } from './person-finder/person-finder.component';

const routes: Routes = [
  { path: '', component: ChatComponent },
  { path: 'rag', component: ChatComponent, data: { conceptId: 'rag' } },
  { path: 'concept/:conceptId', component: ChatComponent },
  { path: 'person-finder', component: PersonFinderComponent },
  { path: 'doc-rag', component: DocRagComponent }
];

@NgModule({
  imports: [RouterModule.forChild(routes)],
  exports: [RouterModule]
})
export class AppsRoutingModule { }

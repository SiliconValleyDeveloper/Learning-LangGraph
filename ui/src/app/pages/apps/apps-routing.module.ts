import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';

import { ChatComponent } from './chat/chat.component';
import { DocRagComponent } from './doc-rag/doc-rag.component';
import { PersonFinderComponent } from './person-finder/person-finder.component';
import { RagArchitectComponent } from './rag-architect/rag-architect.component';

const routes: Routes = [
  { path: '', component: ChatComponent },
  { path: 'rag', component: ChatComponent, data: { conceptId: 'rag' } },
  { path: 'mcp', component: ChatComponent, data: { conceptId: 'mcp_agent' } },
  { path: 'concept/:conceptId', component: ChatComponent },
  { path: 'person-finder', component: PersonFinderComponent },
  { path: 'doc-rag', component: DocRagComponent },
  { path: 'rag-architect', component: RagArchitectComponent }
];

@NgModule({
  imports: [RouterModule.forChild(routes)],
  exports: [RouterModule]
})
export class AppsRoutingModule { }

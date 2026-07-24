import { Component, OnInit } from '@angular/core';
import { FormBuilder, Validators } from '@angular/forms';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';

import { LabContextService } from '../../../core/lab-context.service';
import { environment } from '../../../../environments/environment';

interface DocMeta {
  name: string;
  bytes: number;
  chunk_count: number;
}

interface ChunkPreview {
  source: string;
  score: number;
  preview: string;
}

interface AskResponse {
  answer: string;
  sources: string[];
  chunk_previews: ChunkPreview[];
  workspace: {
    id: string;
    document_count: number;
    chunk_count: number;
  };
}

interface ChatTurn {
  role: 'user' | 'assistant';
  content: string;
  sources?: string[];
}

@Component({
  selector: 'app-doc-rag',
  templateUrl: './doc-rag.component.html',
  styleUrls: ['./doc-rag.component.scss'],
  standalone: false
})
export class DocRagComponent implements OnInit {
  isOnline = false;
  isBusy = false;
  isUploading = false;
  model = 'qwen3:8b';
  errorMessage = '';
  workspaceId = '';
  documents: DocMeta[] = [];
  documentCount = 0;
  chunkCount = 0;
  turns: ChatTurn[] = [];
  lastChunks: ChunkPreview[] = [];

  readonly form;

  constructor(
    private readonly http: HttpClient,
    private readonly formBuilder: FormBuilder,
    private readonly labContext: LabContextService
  ) {
    this.form = this.formBuilder.nonNullable.group({
      question: ['', [Validators.required, Validators.maxLength(4000)]]
    });
  }

  ngOnInit(): void {
    this.labContext.setPage('Doc Q&A', 'Upload documents and ask grounded questions');
    this.http.get<{ status: string; model: string }>(`${environment.apiUrl}/api/health`).subscribe({
      next: response => {
        this.isOnline = response.status === 'ok' || response.status === 'degraded';
        this.model = response.model;
        this.ensureWorkspace();
      },
      error: () => {
        this.isOnline = false;
        this.errorMessage = 'API offline. Start uvicorn on port 8000.';
      }
    });
  }

  ensureWorkspace(): void {
    const saved = localStorage.getItem('docRagWorkspaceId');
    if (saved) {
      this.workspaceId = saved;
      this.refreshWorkspace();
      return;
    }
    this.createWorkspace();
  }

  createWorkspace(): void {
    this.http.post<{ workspace_id: string }>(`${environment.apiUrl}/api/doc-rag/workspaces`, {}).subscribe({
      next: response => {
        this.workspaceId = response.workspace_id;
        localStorage.setItem('docRagWorkspaceId', this.workspaceId);
        this.documents = [];
        this.documentCount = 0;
        this.chunkCount = 0;
        this.turns = [];
        this.lastChunks = [];
        this.errorMessage = '';
      },
      error: (error: HttpErrorResponse) => {
        this.errorMessage = error.error?.detail || 'Could not create workspace';
      }
    });
  }

  refreshWorkspace(): void {
    if (!this.workspaceId) {
      return;
    }
    this.http
      .get<{
        documents: DocMeta[];
        document_count: number;
        chunk_count: number;
      }>(`${environment.apiUrl}/api/doc-rag/workspaces/${this.workspaceId}`)
      .subscribe({
        next: response => {
          this.documents = response.documents;
          this.documentCount = response.document_count;
          this.chunkCount = response.chunk_count;
        },
        error: () => this.createWorkspace()
      });
  }

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file || !this.workspaceId || this.isUploading) {
      return;
    }
    this.isUploading = true;
    this.errorMessage = '';
    const body = new FormData();
    body.append('file', file, file.name);
    this.http
      .post<{
        uploaded: DocMeta & { replaced?: boolean };
        documents: DocMeta[];
        document_count: number;
        chunk_count: number;
      }>(`${environment.apiUrl}/api/doc-rag/workspaces/${this.workspaceId}/upload`, body)
      .subscribe({
        next: response => {
          this.documents = response.documents;
          this.documentCount = response.document_count;
          this.chunkCount = response.chunk_count;
          this.isUploading = false;
          input.value = '';
        },
        error: (error: HttpErrorResponse) => {
          this.isUploading = false;
          this.errorMessage = error.error?.detail || 'Upload failed';
          input.value = '';
        }
      });
  }

  seedSamples(): void {
    if (!this.workspaceId || this.isBusy) {
      return;
    }
    this.isBusy = true;
    this.errorMessage = '';
    this.http
      .post<{
        documents: DocMeta[];
        document_count: number;
        chunk_count: number;
      }>(`${environment.apiUrl}/api/doc-rag/workspaces/${this.workspaceId}/seed`, {})
      .subscribe({
        next: response => {
          this.documents = response.documents;
          this.documentCount = response.document_count;
          this.chunkCount = response.chunk_count;
          this.isBusy = false;
        },
        error: (error: HttpErrorResponse) => {
          this.isBusy = false;
          this.errorMessage = error.error?.detail || 'Could not seed sample docs';
        }
      });
  }

  deleteDocument(name: string): void {
    if (!this.workspaceId || this.isBusy) {
      return;
    }
    this.isBusy = true;
    this.http
      .delete<{
        documents: DocMeta[];
        document_count: number;
        chunk_count: number;
      }>(`${environment.apiUrl}/api/doc-rag/workspaces/${this.workspaceId}/documents/${encodeURIComponent(name)}`)
      .subscribe({
        next: response => {
          this.documents = response.documents;
          this.documentCount = response.document_count;
          this.chunkCount = response.chunk_count;
          this.isBusy = false;
        },
        error: (error: HttpErrorResponse) => {
          this.isBusy = false;
          this.errorMessage = error.error?.detail || 'Delete failed';
        }
      });
  }

  ask(): void {
    if (this.form.invalid || !this.workspaceId || this.isBusy) {
      return;
    }
    const question = this.form.controls.question.value.trim();
    if (!question) {
      return;
    }
    this.isBusy = true;
    this.errorMessage = '';
    this.turns = [...this.turns, { role: 'user', content: question }];
    this.form.reset({ question: '' });

    this.http.post<AskResponse>(`${environment.apiUrl}/api/doc-rag/ask`, {
      workspace_id: this.workspaceId,
      question
    }).subscribe({
      next: response => {
        this.turns = [
          ...this.turns,
          {
            role: 'assistant',
            content: response.answer,
            sources: response.sources
          }
        ];
        this.lastChunks = response.chunk_previews || [];
        this.documentCount = response.workspace.document_count;
        this.chunkCount = response.workspace.chunk_count;
        this.isBusy = false;
      },
      error: (error: HttpErrorResponse) => {
        this.isBusy = false;
        this.errorMessage = error.error?.detail || 'Ask failed';
      }
    });
  }

  resetConversation(): void {
    this.turns = [];
    this.lastChunks = [];
    this.errorMessage = '';
  }

  newWorkspace(): void {
    localStorage.removeItem('docRagWorkspaceId');
    this.createWorkspace();
  }
}

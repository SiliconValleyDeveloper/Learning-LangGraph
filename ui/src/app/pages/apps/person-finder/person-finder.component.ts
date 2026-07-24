import { Component, OnInit } from '@angular/core';
import { FormBuilder, Validators } from '@angular/forms';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';

import { LabContextService } from '../../../core/lab-context.service';
import { environment } from '../../../../environments/environment';
import {
  ChatResponse,
  ConceptInfo,
  ExecutionStats,
  EMPTY_STATS,
  TraceStep,
  ToolEvent
} from '../chat/chat.model';

@Component({
  selector: 'app-person-finder',
  templateUrl: './person-finder.component.html',
  styleUrls: ['./person-finder.component.scss'],
  standalone: false
})
export class PersonFinderComponent implements OnInit {
  readonly conceptId = 'person_finder';

  isOnline = false;
  isSending = false;
  model = 'qwen3:8b';
  errorMessage = '';
  teachPoints: string[] = [];
  summary = '';
  reply = '';
  profile: Record<string, unknown> | null = null;
  sources: string[] = [];
  toolEvents: ToolEvent[] = [];
  traceSteps: TraceStep[] = [];
  stats: ExecutionStats = { ...EMPTY_STATS };
  threadId: string = crypto.randomUUID();

  readonly form;

  constructor(
    private readonly http: HttpClient,
    private readonly formBuilder: FormBuilder,
    private readonly labContext: LabContextService
  ) {
    this.form = this.formBuilder.nonNullable.group({
      name: ['', [Validators.required, Validators.maxLength(200)]],
      email: ['', [Validators.maxLength(200)]],
      company: ['', [Validators.maxLength(200)]],
      role: ['', [Validators.maxLength(200)]],
      linkedin: ['', [Validators.maxLength(400)]],
      notes: ['', [Validators.maxLength(2000)]]
    });
  }

  ngOnInit(): void {
    this.labContext.setPage('Person Finder', 'Lookup people with a LangGraph agent');
    this.http.get<{ status: string; model: string; concepts?: ConceptInfo[] }>(
      `${environment.apiUrl}/api/health`
    ).subscribe({
      next: response => {
        this.isOnline = response.status === 'ok' || response.status === 'degraded';
        this.model = response.model;
        const concept = response.concepts?.find(item => item.id === this.conceptId);
        if (concept) {
          this.teachPoints = concept.teach;
          this.summary = concept.summary;
          this.labContext.setPage(concept.title, concept.phase || 'Person Finder');
        } else {
          this.loadConcept();
        }
      },
      error: () => {
        this.isOnline = false;
        this.loadConcept();
      }
    });
  }

  fillSample(sample: 'chase' | 'karpathy'): void {
    this.resetThread();
    if (sample === 'chase') {
      this.form.setValue({
        name: 'Harrison Chase',
        email: '',
        company: 'LangChain',
        role: 'CEO',
        linkedin: '',
        notes: 'Focus on public professional information only'
      });
      return;
    }
    this.form.setValue({
      name: 'Andrej Karpathy',
      email: '',
      company: '',
      role: 'AI researcher',
      linkedin: '',
      notes: 'Focus on public AI work and education content'
    });
  }

  research(): void {
    if (this.form.invalid || this.isSending) {
      this.form.markAllAsTouched();
      return;
    }

    // Fresh thread every search so prior people never contaminate results.
    this.threadId = crypto.randomUUID();

    const value = this.form.getRawValue();
    const lines = [
      `Name: ${value.name.trim()}`,
      value.email.trim() ? `Email: ${value.email.trim()}` : '',
      value.company.trim() ? `Company: ${value.company.trim()}` : '',
      value.role.trim() ? `Role: ${value.role.trim()}` : '',
      value.linkedin.trim() ? `LinkedIn: ${value.linkedin.trim()}` : '',
      value.notes.trim() ? `Notes: ${value.notes.trim()}` : ''
    ].filter(Boolean);

    this.isSending = true;
    this.errorMessage = '';
    this.reply = '';
    this.profile = null;
    this.sources = [];
    this.toolEvents = [];
    this.traceSteps = [];
    this.stats = { ...EMPTY_STATS };

    this.http
      .post<ChatResponse>(`${environment.apiUrl}/api/run`, {
        message: lines.join('\n'),
        concept_id: this.conceptId,
        thread_id: this.threadId
      })
      .subscribe({
        next: response => {
          this.isSending = false;
          this.threadId = response.thread_id;
          this.reply = response.reply;
          this.toolEvents = response.tool_events || [];
          this.traceSteps = response.trace || [];
          this.stats = response.stats;
          const extra = response.state_extra || {};
          const profile = (extra['profile'] as Record<string, unknown> | undefined) || null;
          this.profile = profile
            ? Object.fromEntries(
                Object.entries(profile).filter(([key]) => key !== '_reflection')
              )
            : null;
          const fromProfile = Array.isArray(this.profile?.['sources'])
            ? (this.profile?.['sources'] as string[])
            : [];
          this.sources = fromProfile.filter(item => typeof item === 'string');
        },
        error: (error: HttpErrorResponse) => {
          this.isSending = false;
          this.errorMessage =
            typeof error.error?.detail === 'string'
              ? error.error.detail
              : error.message || 'Person Finder request failed';
        }
      });
  }

  resetThread(): void {
    this.threadId = crypto.randomUUID();
    this.reply = '';
    this.profile = null;
    this.sources = [];
    this.toolEvents = [];
    this.traceSteps = [];
    this.stats = { ...EMPTY_STATS };
    this.errorMessage = '';
  }

  profileKeys(): string[] {
    return this.profile ? Object.keys(this.profile) : [];
  }

  displayValue(value: unknown): string {
    if (Array.isArray(value)) {
      return value.map(item => String(item)).join(', ');
    }
    if (value == null) {
      return '—';
    }
    if (typeof value === 'object') {
      return JSON.stringify(value);
    }
    return String(value);
  }

  private loadConcept(): void {
    this.http
      .get<ConceptInfo>(`${environment.apiUrl}/api/concepts/${this.conceptId}`)
      .subscribe({
        next: concept => {
          this.teachPoints = concept.teach;
          this.summary = concept.summary;
          this.labContext.setPage(concept.title, concept.phase || 'Person Finder');
        }
      });
  }
}

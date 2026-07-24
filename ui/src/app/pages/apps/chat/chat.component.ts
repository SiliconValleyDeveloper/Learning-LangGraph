import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Component, ElementRef, OnInit, ViewChild } from '@angular/core';
import { FormBuilder, Validators } from '@angular/forms';
import { ActivatedRoute } from '@angular/router';

import { LabContextService } from '../../../core/lab-context.service';
import { environment } from '../../../../environments/environment';
import {
  ChatResponse,
  ConceptInfo,
  DEFAULT_GRAPH,
  EMPTY_STATS,
  ExecutionStats,
  GraphAnatomy,
  GraphEdgeView,
  GraphNodeView,
  HealthResponse,
  PathNodeView,
  RagChunk,
  RagKnowledgeFile,
  StateSnapshot,
  ToolEvent,
  TraceStep,
  UiMessage
} from './chat.model';

@Component({
  selector: 'app-chat',
  templateUrl: './chat.component.html',
  styleUrls: ['./chat.component.scss'],
  standalone: false
})
export class ChatComponent implements OnInit {
  @ViewChild('conversation') conversation?: ElementRef<HTMLElement>;
  @ViewChild('conceptBar') conceptBar?: ElementRef<HTMLElement>;

  readonly svgWidth = 320;
  readonly svgHeight = 360;
  readonly promptForm;

  concepts: ConceptInfo[] = [];
  selectedConceptId = 'tools';
  threadId = crypto.randomUUID();
  suggestions: string[] = [];
  teachPoints: string[] = [];
  ragQuestion = '';
  ragChunks: RagChunk[] = [];
  ragSources: string[] = [];
  ragKnowledgeFiles: RagKnowledgeFile[] = [];
  ragDocumentCount = 0;
  ragPublicCount = 0;
  ragPrivateCount = 0;
  ragLesson = 'graph';
  isDocUploading = false;
  docChunkCount = 0;
  webSearchEnabled = false;
  webResults: Array<{ title: string; url: string; snippet: string }> = [];
  searchPhase: 'idle' | 'searching' | 'reading' | 'done' = 'idle';
  searchPulseLabels = [
    'Planning search queries…',
    'Scanning the web…',
    'Filtering sources…',
    'Grading evidence…',
    'Writing a grounded answer…'
  ];
  searchPulseIndex = 0;
  private searchPulseTimer: ReturnType<typeof setInterval> | null = null;

  messages: UiMessage[] = [];
  isSending = false;
  isOnline = false;
  model = 'qwen3:8b';
  errorMessage = '';
  graph: GraphAnatomy = DEFAULT_GRAPH;
  stats: ExecutionStats = { ...EMPTY_STATS };
  traceSteps: TraceStep[] = [];
  activeTraceIndex = -1;
  hitlPending: Record<string, unknown> | null = null;
  stateSnapshot: StateSnapshot = {
    message_count: 0,
    last_message_type: 'None',
    memory_enabled: true
  };
  private traceRunId = 0;
  private layoutMap: Record<string, { x: number; y: number }> = {};

  /** Conversation panel: expanded shows stream + composer; collapsed keeps a slim composer. */
  chatExpanded = true;

  constructor(
    private readonly http: HttpClient,
    private readonly formBuilder: FormBuilder,
    private readonly route: ActivatedRoute,
    private readonly labContext: LabContextService
  ) {
    this.promptForm = this.formBuilder.nonNullable.group({
      message: ['', [Validators.required, Validators.maxLength(4000)]]
    });
  }

  ngOnInit(): void {
    this.route.queryParamMap.subscribe(params => {
      this.ragLesson = params.get('lesson') || 'graph';
    });
    this.route.paramMap.subscribe(params => {
      const conceptId =
        params.get('conceptId') ||
        String(this.route.snapshot.data['conceptId'] || this.selectedConceptId);
      this.selectedConceptId = conceptId;
      if (this.concepts.length) {
        this.selectConcept(conceptId);
      }
    });
    this.http.get<HealthResponse>(`${environment.apiUrl}/api/health`).subscribe({
      next: response => {
        this.isOnline = response.status === 'ok' || response.status === 'degraded';
        this.model = response.model;
        if (response.concepts?.length) {
          this.concepts = response.concepts;
          this.selectConcept(this.selectedConceptId);
        } else {
          this.http.get<ConceptInfo[]>(`${environment.apiUrl}/api/concepts`).subscribe({
            next: concepts => {
              this.concepts = concepts;
              this.selectConcept(this.selectedConceptId);
            }
          });
        }
      },
      error: () => {
        this.isOnline = false;
        this.http.get<ConceptInfo[]>(`${environment.apiUrl}/api/concepts`).subscribe({
          next: concepts => {
            this.concepts = concepts;
            this.selectConcept(this.selectedConceptId);
          }
        });
      }
    });
  }

  get selectedConcept(): ConceptInfo | undefined {
    return this.concepts.find(concept => concept.id === this.selectedConceptId);
  }

  get isRagConcept(): boolean {
    return (
      this.selectedConceptId === 'rag' ||
      this.selectedConceptId === 'rag_complex' ||
      this.selectedConceptId === 'doc_rag' ||
      this.selectedConceptId === 'advanced_chatbot'
    );
  }

  get supportsWebSearchToggle(): boolean {
    return (
      this.selectedConceptId === 'advanced_chatbot' ||
      this.selectedConceptId === 'web_search'
    );
  }

  get isDocRagConcept(): boolean {
    return this.selectedConceptId === 'doc_rag' || this.selectedConceptId === 'advanced_chatbot';
  }

  get isWebSearchConcept(): boolean {
    return this.selectedConceptId === 'web_search' || this.webSearchEnabled;
  }

  toggleWebSearch(): void {
    if (this.selectedConceptId === 'web_search') {
      this.webSearchEnabled = true;
      return;
    }
    this.webSearchEnabled = !this.webSearchEnabled;
  }

  toggleChatExpanded(): void {
    this.chatExpanded = !this.chatExpanded;
  }


  get docUploadApiBase(): string {
    return this.selectedConceptId === 'advanced_chatbot'
      ? `${environment.apiUrl}/api/advanced-chat`
      : `${environment.apiUrl}/api/doc-rag`;
  }

  get ragActiveStage(): string {
    if (!this.ragChunks.length) {
      if (this.isSending) {
        return 'question';
      }
      const lessonStages: Record<string, string> = {
        llm: 'llm',
        ecosystem: 'index',
        chunking: 'index',
        retrieval: 'retrieve',
        graph: 'index'
      };
      return lessonStages[this.ragLesson] || 'index';
    }
    const activeNode = this.traceSteps[this.activeTraceIndex]?.node;
    if (
      activeNode === 'retrieve' ||
      activeNode === 'rewrite' ||
      activeNode === 'classify' ||
      activeNode === 'grade' ||
      activeNode === 'bump_retry'
    ) {
      return 'retrieve';
    }
    if (
      activeNode === 'generate' ||
      activeNode === 'verify'
    ) {
      return 'llm';
    }
    return this.activeTraceIndex >= this.traceSteps.length ? 'answer' : 'question';
  }

  selectConcept(conceptId: string): void {
    const concept = this.concepts.find(item => item.id === conceptId);
    if (!concept) {
      return;
    }
    this.selectedConceptId = concept.id;
    this.labContext.setPage(concept.title, concept.phase || 'Local LangGraph execution console');
    this.threadId = crypto.randomUUID();
    this.suggestions = concept.sample_prompts;
    this.teachPoints = concept.teach;
    this.hitlPending = null;
    this.traceSteps = [];
    this.activeTraceIndex = -1;
    this.stats = { ...EMPTY_STATS };
    this.ragQuestion = '';
    this.ragChunks = [];
    this.ragSources = [];
    this.ragKnowledgeFiles = [];
    this.ragDocumentCount = 0;
    this.ragPublicCount = 0;
    this.ragPrivateCount = 0;
    this.docChunkCount = 0;
    this.webResults = [];
    this.searchPhase = 'idle';
    this.stopSearchPulse();
    if (concept.id === 'web_search') {
      this.webSearchEnabled = true;
    } else if (concept.id !== 'advanced_chatbot') {
      this.webSearchEnabled = false;
    }
    this.graph = {
      nodes: concept.topology_nodes.map(node => node.id),
      node_count: concept.node_count,
      edges: concept.topology_edges.map(
        edge => `${edge.source} → ${edge.target}`
      ),
      edge_count: concept.edge_count,
      conditional_edges: concept.topology_edges
        .filter(edge => edge.kind === 'conditional')
        .map(edge => `${edge.source} → ${edge.target}`),
      conditional_count: concept.conditional_count,
      tools: concept.tools,
      tool_count: concept.tool_count,
      state_keys: concept.state_keys,
      topology_nodes: concept.topology_nodes,
      topology_edges: concept.topology_edges
    };
    this.rebuildLayout();
    this.messages = [
      {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: this.isDocRagConcept
          ? `${concept.title}\n\n${concept.summary}\n\nUpload docs in the panel (or use sample prompts — README auto-seeds if empty).`
          : `${concept.title}\n\n${concept.summary}\n\nTry a sample prompt to watch nodes and edges light up.`,
        time: new Date()
      }
    ];
    if (this.isDocRagConcept) {
      this.refreshDocWorkspace();
    }
  }

  refreshDocWorkspace(): void {
    if (!this.isDocRagConcept) {
      return;
    }
    this.http
      .get<{
        documents: Array<{ name: string; bytes: number; chunk_count: number }>;
        document_count: number;
        chunk_count: number;
      }>(`${this.docUploadApiBase}/workspaces/${this.threadId}`)
      .subscribe({
        next: response => {
          this.ragKnowledgeFiles = response.documents.map(doc => ({
            name: doc.name,
            path: doc.name,
            visibility: 'uploaded',
            private: false
          }));
          this.ragDocumentCount = response.document_count;
          this.ragPublicCount = response.document_count;
          this.ragPrivateCount = 0;
          this.docChunkCount = response.chunk_count;
        },
        error: () => {
          /* empty workspace until first upload/ask */
        }
      });
  }

  onDocFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file || !this.isDocRagConcept || this.isDocUploading) {
      return;
    }
    this.isDocUploading = true;
    this.errorMessage = '';
    const body = new FormData();
    body.append('file', file, file.name);
    this.http
      .post<{
        documents: Array<{ name: string; bytes: number; chunk_count: number }>;
        document_count: number;
        chunk_count: number;
      }>(`${this.docUploadApiBase}/workspaces/${this.threadId}/upload`, body)
      .subscribe({
        next: response => {
          this.ragKnowledgeFiles = response.documents.map(doc => ({
            name: doc.name,
            path: doc.name,
            visibility: 'uploaded',
            private: false
          }));
          this.ragDocumentCount = response.document_count;
          this.ragPublicCount = response.document_count;
          this.ragPrivateCount = 0;
          this.docChunkCount = response.chunk_count;
          this.isDocUploading = false;
          input.value = '';
        },
        error: (error: HttpErrorResponse) => {
          this.isDocUploading = false;
          this.errorMessage = error.error?.detail || 'Upload failed';
          input.value = '';
        }
      });
  }

  seedDocSamples(): void {
    if (!this.isDocRagConcept || this.isDocUploading) {
      return;
    }
    if (this.selectedConceptId === 'advanced_chatbot') {
      // Advanced chatbot auto-seeds README on first ask; just refresh list.
      this.refreshDocWorkspace();
      return;
    }
    this.isDocUploading = true;
    this.http
      .post<{
        documents: Array<{ name: string; bytes: number; chunk_count: number }>;
        document_count: number;
        chunk_count: number;
      }>(`${this.docUploadApiBase}/workspaces/${this.threadId}/seed`, {})
      .subscribe({
        next: response => {
          this.ragKnowledgeFiles = response.documents.map(doc => ({
            name: doc.name,
            path: doc.name,
            visibility: 'uploaded',
            private: false
          }));
          this.ragDocumentCount = response.document_count;
          this.ragPublicCount = response.document_count;
          this.ragPrivateCount = 0;
          this.docChunkCount = response.chunk_count;
          this.isDocUploading = false;
        },
        error: (error: HttpErrorResponse) => {
          this.isDocUploading = false;
          this.errorMessage = error.error?.detail || 'Could not load sample docs';
        }
      });
  }

  sendSuggestion(suggestion: string): void {
    this.promptForm.controls.message.setValue(suggestion);
    this.sendMessage();
  }

  scrollConcepts(direction: -1 | 1): void {
    const element = this.conceptBar?.nativeElement;
    if (!element) {
      return;
    }
    element.scrollBy({
      left: direction * Math.max(element.clientWidth * 0.7, 320),
      behavior: 'smooth'
    });
  }

  sendMessage(): void {
    const content = this.promptForm.controls.message.value.trim();
    if (!content || this.promptForm.invalid || this.isSending) {
      return;
    }

    this.messages = [
      ...this.messages,
      {
        id: crypto.randomUUID(),
        role: 'user',
        content,
        time: new Date()
      }
    ];
    this.promptForm.reset();
    this.isSending = true;
    this.traceSteps = [];
    this.activeTraceIndex = -1;
    this.stats = { ...EMPTY_STATS };
    this.hitlPending = null;
    this.traceRunId++;
    this.errorMessage = '';
    this.webResults = [];
    if (this.isRagConcept) {
      this.ragQuestion = content;
      this.ragChunks = [];
      this.ragSources = [];
    }
    const useWeb =
      this.selectedConceptId === 'web_search' || this.webSearchEnabled;
    if (useWeb) {
      this.startSearchPulse();
    } else {
      this.searchPhase = 'idle';
    }
    this.scrollToBottom();

    this.http
      .post<ChatResponse>(`${environment.apiUrl}/api/run`, {
        message: content,
        thread_id: this.threadId,
        concept_id: this.selectedConceptId,
        web_search: useWeb && this.selectedConceptId === 'advanced_chatbot'
      })
      .subscribe({
        next: response => this.applyRunResponse(response),
        error: (error: HttpErrorResponse) => this.handleError(error)
      });
  }

  private startSearchPulse(): void {
    this.searchPhase = 'searching';
    this.searchPulseIndex = 0;
    this.stopSearchPulse();
    this.searchPulseTimer = setInterval(() => {
      this.searchPulseIndex =
        (this.searchPulseIndex + 1) % this.searchPulseLabels.length;
      if (this.searchPulseIndex >= 2) {
        this.searchPhase = 'reading';
      }
    }, 900);
  }

  private stopSearchPulse(): void {
    if (this.searchPulseTimer) {
      clearInterval(this.searchPulseTimer);
      this.searchPulseTimer = null;
    }
  }

  approveHitl(approve: boolean): void {
    if (!this.hitlPending || this.isSending) {
      return;
    }
    this.isSending = true;
    this.errorMessage = '';
    this.http
      .post<ChatResponse>(`${environment.apiUrl}/api/hitl/resume`, {
        thread_id: this.threadId,
        approve
      })
      .subscribe({
        next: response => this.applyRunResponse(response),
        error: (error: HttpErrorResponse) => this.handleError(error)
      });
  }

  private applyRunResponse(response: ChatResponse): void {
    this.stopSearchPulse();
    this.searchPhase = 'done';
    const webHits = response.state_extra?.['web_results'];
    this.webResults = Array.isArray(webHits)
      ? (webHits as Array<{ title: string; url: string; snippet: string }>)
      : [];
    this.messages = [
      ...this.messages,
      {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: response.reply,
        time: new Date(),
        toolEvents: response.tool_events,
        stats: response.stats,
        interrupted: response.interrupted,
        webResults: this.webResults.length ? this.webResults : undefined
      }
    ];
    this.isSending = false;
    this.isOnline = true;
    this.traceSteps = response.trace;
    this.stateSnapshot = response.state;
    if (this.isRagConcept) {
      const chunks = response.state_extra?.['retrieved_chunks'];
      const sources = response.state_extra?.['sources'];
      const knowledgeFiles = response.state_extra?.['knowledge_files'];
      this.ragQuestion =
        String(response.state_extra?.['question'] || this.ragQuestion);
      this.ragChunks = Array.isArray(chunks) ? chunks as RagChunk[] : [];
      this.ragSources = Array.isArray(sources)
        ? sources.map(source => String(source))
        : [];
      this.ragKnowledgeFiles = Array.isArray(knowledgeFiles)
        ? knowledgeFiles as RagKnowledgeFile[]
        : [];
      this.ragDocumentCount = Number(response.state_extra?.['document_count'] || 0);
      this.ragPublicCount = Number(response.state_extra?.['public_document_count'] || 0);
      this.ragPrivateCount = Number(response.state_extra?.['private_document_count'] || 0);
      this.docChunkCount = Number(response.state_extra?.['chunk_count'] || this.docChunkCount);
    }
    this.hitlPending = response.interrupted ? response.pending || null : null;
    this.graph = {
      ...DEFAULT_GRAPH,
      ...response.graph,
      topology_nodes:
        response.graph.topology_nodes?.length
          ? response.graph.topology_nodes
          : this.graph.topology_nodes,
      topology_edges:
        response.graph.topology_edges?.length
          ? response.graph.topology_edges
          : this.graph.topology_edges
    };
    this.rebuildLayout();
    this.stats = response.stats;
    this.playTrace();
    this.scrollToBottom();
  }

  private handleError(error: HttpErrorResponse): void {
    this.stopSearchPulse();
    this.searchPhase = 'idle';
    this.errorMessage =
      error.error?.detail ||
      'The learning API is unavailable. Start FastAPI (and Ollama if needed).';
    this.isSending = false;
    this.scrollToBottom();
  }

  formatArgs(args: Record<string, unknown>): string {
    return Object.entries(args)
      .map(([key, value]) => `${key}=${JSON.stringify(value)}`)
      .join(', ');
  }

  trackMessage(_: number, message: UiMessage): string {
    return message.id;
  }

  trackTool(_: number, tool: ToolEvent): string {
    return `${tool.name}-${JSON.stringify(tool.args)}`;
  }

  get visibleState(): StateSnapshot {
    return this.traceSteps[this.activeTraceIndex]?.state || this.stateSnapshot;
  }

  get currentTraceSummary(): string {
    if (this.isSending && this.traceSteps.length === 0) {
      return 'Running concept graph…';
    }
    if (
      this.activeTraceIndex === this.traceSteps.length &&
      this.traceSteps.length > 0
    ) {
      return this.hitlPending
        ? 'Paused for human approval'
        : 'Reached END for this concept run';
    }
    const step = this.traceSteps[this.activeTraceIndex];
    if (!step) {
      return 'Pick a concept and send a prompt to animate the graph';
    }
    if (step.decision) {
      return `${step.node}: ${step.decision}`;
    }
    if (step.edge_from && step.edge_to) {
      return `${step.edge_from} → ${step.edge_to} · ${step.summary}`;
    }
    return step.summary;
  }

  get metricCards(): Array<{ label: string; value: number | string; hint: string }> {
    const stats =
      this.activeTraceIndex >= 0 && this.traceSteps.length
        ? this.partialStats()
        : this.stats;
    return [
      { label: 'Steps', value: stats.steps, hint: 'updates' },
      { label: 'Nodes', value: stats.nodes_visited, hint: 'unique' },
      { label: 'Edges', value: stats.edges_traversed, hint: 'walked' },
      { label: 'Cond.', value: stats.conditional_decisions, hint: 'branch' },
      { label: 'Tools', value: stats.tool_calls, hint: 'calls' },
      { label: 'Loops', value: stats.loops, hint: 'cycles' },
      { label: 'Agent', value: stats.agent_runs, hint: 'runs' },
      {
        label: 'State',
        value: this.visibleState.message_count || stats.state_messages,
        hint: 'msgs'
      }
    ];
  }

  get graphNodes(): GraphNodeView[] {
    const activeId = this.activeNodeId();
    const visited = this.visitedNodeIds();
    return (this.graph.topology_nodes || []).map(node => {
      const point = this.layoutMap[node.id] || { x: 160, y: 180 };
      return {
        id: node.id,
        label: node.label,
        kind: node.kind,
        x: point.x,
        y: point.y,
        active: activeId === node.id,
        visited: visited.has(node.id)
      };
    });
  }

  get graphEdges(): GraphEdgeView[] {
    const activeEdgeIds = this.activeEdgeIds();
    const visitedEdgeIds = this.visitedEdgeIds();
    return (this.graph.topology_edges || []).map(edge => {
      const from = this.layoutMap[edge.source] || { x: 0, y: 0 };
      const to = this.layoutMap[edge.target] || { x: 0, y: 0 };
      const path = this.edgePath(edge.source, edge.target, edge.kind, from, to);
      return {
        id: edge.id,
        source: edge.source,
        target: edge.target,
        kind: edge.kind,
        label: edge.label || '',
        d: path.d,
        labelX: path.labelX,
        labelY: path.labelY,
        active: activeEdgeIds.has(edge.id),
        visited: visitedEdgeIds.has(edge.id)
      };
    });
  }

  get pathNodes(): PathNodeView[] {
    const path = this.stats.path.length
      ? this.stats.path
      : this.isSending
        ? [this.graph.topology_nodes[1]?.id || 'node']
        : [];
    const endPath = [...path];
    if (
      this.activeTraceIndex === this.traceSteps.length &&
      this.traceSteps.length > 0 &&
      !this.hitlPending
    ) {
      endPath.push('__end__');
    }
    return endPath.map((node, index) => ({
      id: `${node}-${index}`,
      label: node === '__end__' ? 'END' : node,
      kind: node === '__end__' ? 'end' : node,
      index,
      active:
        this.activeTraceIndex === index ||
        (node === '__end__' && this.activeTraceIndex === this.traceSteps.length),
      done: this.activeTraceIndex > index
    }));
  }

  nodeRadius(kind: string): number {
    if (kind === 'conditional') {
      return 22;
    }
    if (kind === 'start' || kind === 'end') {
      return 24;
    }
    return 28;
  }

  private rebuildLayout(): void {
    const nodes = this.graph.topology_nodes || [];
    const map: Record<string, { x: number; y: number }> = {};
    const centerX = this.svgWidth / 2;
    const start = nodes.find(node => node.kind === 'start') || nodes[0];
    const end = nodes.find(node => node.kind === 'end');

    if (start) {
      map[start.id] = { x: centerX, y: 36 };
    }

    let y = 110;
    const byRow: string[][] = [];
    let remaining = nodes.filter(node => node.id !== start?.id && node.id !== end?.id);
    // Simple layers: non-branch first column-ish using topology order
    const layerSize = Math.max(1, Math.ceil(remaining.length / 3));
    while (remaining.length) {
      byRow.push(remaining.slice(0, layerSize).map(node => node.id));
      remaining = remaining.slice(layerSize);
    }

    byRow.forEach(row => {
      const count = row.length;
      row.forEach((id, index) => {
        const x =
          count === 1
            ? centerX
            : 56 + (index * (this.svgWidth - 112)) / Math.max(count - 1, 1);
        map[id] = { x, y };
      });
      y += 90;
    });

    if (end) {
      map[end.id] = { x: centerX, y: Math.min(y, this.svgHeight - 36) };
    }

    // Ensure every node has a point
    nodes.forEach((node, index) => {
      if (!map[node.id]) {
        map[node.id] = { x: centerX, y: 80 + index * 50 };
      }
    });

    // Prefer known nice layouts for tools concept
    if (this.selectedConceptId === 'tools') {
      Object.assign(map, {
        __start__: { x: 160, y: 36 },
        agent: { x: 160, y: 118 },
        tools_condition: { x: 160, y: 200 },
        tools: { x: 58, y: 292 },
        __end__: { x: 262, y: 292 }
      });
    }

    this.layoutMap = map;
  }

  private activeNodeId(): string {
    if (this.isSending && this.traceSteps.length === 0) {
      return this.graph.topology_nodes[1]?.id || '';
    }
    if (
      this.activeTraceIndex === this.traceSteps.length &&
      this.traceSteps.length > 0
    ) {
      return this.hitlPending ? 'approve' : '__end__';
    }
    const step = this.traceSteps[this.activeTraceIndex];
    if (!step) {
      return '';
    }
    if (step.decision && step.node === 'agent') {
      return 'tools_condition';
    }
    if (step.decision && step.node === 'supervisor') {
      return 'route';
    }
    if (step.decision && step.node === 'draft') {
      return 'approve';
    }
    if (step.decision && step.node === 'classify') {
      return 'route';
    }
    return step.node;
  }

  private visitedNodeIds(): Set<string> {
    const ids = new Set<string>(['__start__']);
    if (this.activeTraceIndex < 0) {
      return this.isSending ? ids : new Set<string>();
    }
    const slice = this.traceSteps.slice(
      0,
      Math.min(this.activeTraceIndex + 1, this.traceSteps.length)
    );
    for (const step of slice) {
      ids.add(step.node);
      if (step.edge_to) {
        ids.add(step.edge_to);
      }
      if (step.decision?.includes('tools_condition') || step.node === 'agent') {
        ids.add('tools_condition');
      }
      if (step.decision?.includes('route') || step.node === 'supervisor' || step.node === 'classify') {
        ids.add('route');
      }
      if (step.decision?.includes('interrupt') || step.node === 'draft') {
        ids.add('approve');
      }
    }
    if (this.activeTraceIndex === this.traceSteps.length && !this.hitlPending) {
      ids.add('__end__');
    }
    return ids;
  }

  private findEdgeId(source: string, target: string): string | null {
    const edge = (this.graph.topology_edges || []).find(
      item => item.source === source && item.target === target
    );
    return edge?.id || null;
  }

  private activeEdgeIds(): Set<string> {
    const ids = new Set<string>();
    const step = this.traceSteps[this.activeTraceIndex];
    if (!step) {
      return ids;
    }
    if (step.edge_from && step.node) {
      const direct = this.findEdgeId(step.edge_from, step.node);
      if (direct) {
        ids.add(direct);
      }
    }
    if (step.edge_from && step.edge_to) {
      const outbound = this.findEdgeId(step.edge_from, step.edge_to);
      if (outbound) {
        ids.add(outbound);
      }
      // via conditional hub
      for (const hub of ['tools_condition', 'route', 'approve', 'checkpoint']) {
        const a = this.findEdgeId(step.edge_from, hub);
        const b = this.findEdgeId(hub, step.edge_to);
        if (a) {
          ids.add(a);
        }
        if (b) {
          ids.add(b);
        }
      }
    }
    return ids;
  }

  private visitedEdgeIds(): Set<string> {
    const ids = new Set<string>();
    if (this.activeTraceIndex < 0) {
      return ids;
    }
    const slice = this.traceSteps.slice(
      0,
      Math.min(this.activeTraceIndex + 1, this.traceSteps.length)
    );
    for (const step of slice) {
      if (step.edge_from) {
        const into = this.findEdgeId(step.edge_from, step.node);
        if (into) {
          ids.add(into);
        }
      }
      if (step.edge_to) {
        const out = this.findEdgeId(step.node, step.edge_to);
        if (out) {
          ids.add(out);
        }
        for (const hub of ['tools_condition', 'route', 'approve', 'checkpoint']) {
          const a = this.findEdgeId(step.node, hub);
          const b = this.findEdgeId(hub, step.edge_to);
          if (a) {
            ids.add(a);
          }
          if (b) {
            ids.add(b);
          }
        }
      }
    }
    return ids;
  }

  private edgePath(
    source: string,
    target: string,
    kind: string,
    from: { x: number; y: number },
    to: { x: number; y: number }
  ): { d: string; labelX: number; labelY: number } {
    if (kind === 'loop') {
      const midX = Math.min(from.x, to.x) - 40;
      const midY = (from.y + to.y) / 2;
      return {
        d: `M ${from.x - 24} ${from.y} C ${midX} ${from.y - 10}, ${midX} ${to.y + 10}, ${to.x - 28} ${to.y}`,
        labelX: midX + 18,
        labelY: midY
      };
    }
    const dx = to.x - from.x;
    const startY = from.y + (Math.abs(dx) < 8 ? 28 : 18);
    const endY = to.y - (Math.abs(dx) < 8 ? 28 : 18);
    if (Math.abs(dx) < 8) {
      return {
        d: `M ${from.x} ${startY} L ${to.x} ${endY}`,
        labelX: from.x + 18,
        labelY: (from.y + to.y) / 2
      };
    }
    const cx = (from.x + to.x) / 2;
    const cy = (from.y + to.y) / 2 - 8;
    return {
      d: `M ${from.x} ${startY} Q ${cx} ${cy} ${to.x} ${endY}`,
      labelX: cx,
      labelY: cy - 8
    };
  }

  private partialStats(): ExecutionStats {
    const slice = this.traceSteps.slice(
      0,
      Math.min(this.activeTraceIndex + 1, this.traceSteps.length)
    );
    const path = slice.map(step => step.node);
    const toolCalls = slice.reduce(
      (count, step) => count + (step.tool_names?.length || 0),
      0
    );
    const decisions = slice.filter(step => !!step.decision).length;
    return {
      steps: slice.length,
      nodes_visited: new Set(path).size,
      agent_runs: path.filter(node =>
        ['agent', 'chat', 'greet', 'classify', 'draft', 'supervisor'].includes(node)
      ).length,
      tool_node_runs: path.filter(node =>
        ['tools', 'researcher', 'writer', 'send', 'billing', 'tech', 'general'].includes(
          node
        )
      ).length,
      edges_traversed: slice.length,
      conditional_decisions: decisions,
      tool_calls: toolCalls,
      unique_tools: new Set(slice.flatMap(step => step.tool_names || [])).size,
      loops: slice.filter(step =>
        ['agent', 'supervisor'].includes(step.edge_to || '')
      ).length,
      state_messages: this.visibleState.message_count,
      path
    };
  }

  private playTrace(): void {
    const runId = ++this.traceRunId;
    this.activeTraceIndex = -1;
    this.traceSteps.forEach((_, index) => {
      setTimeout(() => {
        if (runId === this.traceRunId) {
          this.activeTraceIndex = index;
        }
      }, 420 * (index + 1));
    });
    setTimeout(() => {
      if (runId === this.traceRunId) {
        this.activeTraceIndex = this.traceSteps.length;
      }
    }, 420 * (this.traceSteps.length + 1));
  }

  private scrollToBottom(): void {
    setTimeout(() => {
      const element = this.conversation?.nativeElement;
      if (element) {
        element.scrollTo({ top: element.scrollHeight, behavior: 'smooth' });
      }
    });
  }
}

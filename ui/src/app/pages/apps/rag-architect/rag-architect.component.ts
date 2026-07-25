import { Component, OnInit } from '@angular/core';
import { FormBuilder, Validators } from '@angular/forms';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';

import { LabContextService } from '../../../core/lab-context.service';
import { environment } from '../../../../environments/environment';

type Strategy = 'baseline' | 'hybrid' | 'hyde' | 'crag' | 'graph' | 'agentic';
type ViewTab = 'architecture' | 'ask';
type LayerId = 'knowledge' | 'retrieval' | 'validation';

interface Hit {
  chunk_id: string;
  source: string;
  score: number;
  content: string;
}

interface AskResponse {
  question: string;
  strategy: Strategy;
  answer: string;
  sources: string[];
  grade: string;
  verified: boolean;
  notes: string[];
  hits: Hit[];
}

interface StatusResponse {
  strategies: Strategy[];
  default_strategy: Strategy;
  embedding_model: string;
  chat_model: string;
  index: {
    documents: number;
    chunks: number;
    data_dir: string;
  };
}

interface EvalMetrics {
  context_recall: number;
  faithfulness: number;
  citation_rate: number;
}

interface EvalResponse {
  metrics: Record<string, EvalMetrics>;
}

interface ChatTurn {
  role: 'user' | 'assistant';
  content: string;
  strategy?: string;
  sources?: string[];
  grade?: string;
  verified?: boolean;
}

interface ArchLayer {
  id: LayerId;
  title: string;
  subtitle: string;
  job: string;
  contains: string[];
  example: {
    label: string;
    detail: string;
  };
  interview: string;
}

interface FlowNode {
  id: string;
  label: string;
  layer: LayerId | 'io';
  x: number;
  y: number;
}

interface FlowEdge {
  from: string;
  to: string;
  label?: string;
  kind?: 'normal' | 'loop' | 'fail';
}

@Component({
  selector: 'app-rag-architect',
  templateUrl: './rag-architect.component.html',
  styleUrls: ['./rag-architect.component.scss'],
  standalone: false
})
export class RagArchitectComponent implements OnInit {
  readonly strategyHelp: Record<Strategy, string> = {
    baseline: 'Dense retrieve → generate',
    hybrid: 'Dense + BM25 → RRF → rerank',
    hyde: 'Hypothetical doc → dense retrieve',
    crag: 'Retrieve → grade → rewrite/retry',
    graph: 'Vector + entity-graph hops',
    agentic: 'Plan tool: kb / graph / escalate'
  };

  readonly layers: ArchLayer[] = [
    {
      id: 'knowledge',
      title: '1 · Knowledge layer',
      subtitle: 'Where enterprise truth lives',
      job: 'Ingest sources, chunk them, build indexes and ACLs so retrieval has something trustworthy to search.',
      contains: [
        'Source docs (handbook, playbooks, policies, FAQs)',
        'Chunking (fixed / parent-child / sentence-window)',
        'Dense embeddings + BM25 postings',
        'Optional entity graph (P1 → PagerDuty → break-glass)',
        'Visibility / ACL labels'
      ],
      example: {
        label: 'Contoso Ops seed',
        detail:
          'employee_handbook.md stores “Ticket code for leave requests: HR-LEAVE-24”. That text is chunked, embedded, and also indexed by BM25 so the exact ID can be found.'
      },
      interview:
        'Start here: “Our knowledge layer holds internal docs with ACLs, chunking strategy, and dual indexes (dense + sparse).”'
    },
    {
      id: 'retrieval',
      title: '2 · Retrieval layer',
      subtitle: 'How we find evidence',
      job: 'Turn a user question into ranked evidence using rewrite, hybrid search, HyDE, graph hops, and rerank.',
      contains: [
        'Query rewrite / multi-query',
        'Dense semantic search',
        'Sparse BM25 (ticket IDs, codes)',
        'RRF fusion + lexical rerank',
        'HyDE · Graph hops · Agentic tool choice'
      ],
      example: {
        label: 'Q: leave ticket code?',
        detail:
          'Dense may paraphrase “leave policy”. BM25 locks onto HR-LEAVE-24. Hybrid RRF merges both so the leave-policy chunk ranks first.'
      },
      interview:
        '“Retrieval is not one vector call — we fuse sparse + dense, then rerank before generation.”'
    },
    {
      id: 'validation',
      title: '3 · Validation layer',
      subtitle: 'How we stay safe',
      job: 'Decide if evidence is good enough, generate with citations, refuse when weak, and measure quality offline.',
      contains: [
        'Evidence grading (CRAG pass/fail)',
        'Grounded generate with [1] [2] citations',
        'Verify / repair faithfulness',
        'Refuse-when-weak path',
        'Offline eval (recall · faith · cite rate)'
      ],
      example: {
        label: 'Out-of-scope Q',
        detail:
          '“What is the capital of France?” → retrieve Contoso chunks → grade=fail → refuse instead of hallucinating from model memory.'
      },
      interview:
        '“Validation closes the loop: grade, cite, refuse, evaluate — so answers are enterprise-safe.”'
    }
  ];

  readonly workedExample = {
    question: 'What ticket code is used for leave requests?',
    steps: [
      {
        layer: 'knowledge' as LayerId,
        title: 'Knowledge',
        text: 'Handbook chunk already indexed: “…Ticket code for leave requests: HR-LEAVE-24”.'
      },
      {
        layer: 'retrieval' as LayerId,
        title: 'Retrieval (hybrid)',
        text: 'Dense finds leave-policy meaning; BM25 boosts HR-LEAVE-24; RRF puts that chunk at rank #1.'
      },
      {
        layer: 'validation' as LayerId,
        title: 'Validation',
        text: 'Generate: “The ticket code is HR-LEAVE-24 [1].” Verify citations → verified=true.'
      }
    ]
  };

  readonly svgWidth = 720;
  readonly svgHeight = 340;

  readonly baseNodes: FlowNode[] = [
    { id: 'q', label: 'User question', layer: 'io', x: 60, y: 170 },
    { id: 'docs', label: 'Docs / ACL', layer: 'knowledge', x: 200, y: 56 },
    { id: 'chunk', label: 'Chunk + embed', layer: 'knowledge', x: 200, y: 140 },
    { id: 'bm25', label: 'BM25 index', layer: 'knowledge', x: 200, y: 224 },
    { id: 'graph', label: 'Entity graph', layer: 'knowledge', x: 200, y: 300 },
    { id: 'rewrite', label: 'Rewrite / HyDE', layer: 'retrieval', x: 380, y: 56 },
    { id: 'dense', label: 'Dense top-k', layer: 'retrieval', x: 380, y: 140 },
    { id: 'sparse', label: 'BM25 top-k', layer: 'retrieval', x: 380, y: 224 },
    { id: 'fuse', label: 'RRF + rerank', layer: 'retrieval', x: 520, y: 170 },
    { id: 'grade', label: 'Grade evidence', layer: 'validation', x: 640, y: 90 },
    { id: 'gen', label: 'Generate + cite', layer: 'validation', x: 640, y: 180 },
    { id: 'verify', label: 'Verify / refuse', layer: 'validation', x: 640, y: 270 }
  ];

  readonly baseEdges: FlowEdge[] = [
    { from: 'q', to: 'rewrite' },
    { from: 'docs', to: 'chunk' },
    { from: 'chunk', to: 'bm25' },
    { from: 'chunk', to: 'dense' },
    { from: 'bm25', to: 'sparse' },
    { from: 'graph', to: 'fuse', label: 'hops' },
    { from: 'rewrite', to: 'dense' },
    { from: 'rewrite', to: 'sparse' },
    { from: 'dense', to: 'fuse' },
    { from: 'sparse', to: 'fuse' },
    { from: 'fuse', to: 'grade' },
    { from: 'grade', to: 'gen', label: 'pass' },
    { from: 'grade', to: 'rewrite', label: 'fail · CRAG', kind: 'loop' },
    { from: 'gen', to: 'verify' }
  ];

  readonly strategyFlows: Record<Strategy, string[]> = {
    baseline: ['q', 'dense', 'fuse', 'gen', 'verify'],
    hybrid: ['q', 'rewrite', 'dense', 'sparse', 'fuse', 'gen', 'verify'],
    hyde: ['q', 'rewrite', 'dense', 'fuse', 'gen', 'verify'],
    crag: ['q', 'rewrite', 'dense', 'sparse', 'fuse', 'grade', 'gen', 'verify'],
    graph: ['q', 'dense', 'graph', 'fuse', 'gen', 'verify'],
    agentic: ['q', 'rewrite', 'dense', 'sparse', 'graph', 'fuse', 'grade', 'gen', 'verify']
  };

  activeTab: ViewTab = 'architecture';
  selectedLayer: LayerId = 'knowledge';
  isOnline = false;
  isBusy = false;
  isEvalBusy = false;
  chatModel = 'qwen3:8b';
  embedModel = 'nomic-embed-text';
  documentCount = 0;
  chunkCount = 0;
  strategies: Strategy[] = ['baseline', 'hybrid', 'hyde', 'crag', 'graph', 'agentic'];
  selectedStrategy: Strategy = 'hybrid';
  errorMessage = '';
  turns: ChatTurn[] = [];
  lastHits: Hit[] = [];
  lastNotes: string[] = [];
  evalRows: { strategy: string; recall: number; faith: number; cite: number }[] = [];

  readonly form;
  readonly suggestions = [
    'What ticket code is used for leave requests?',
    'What is the P1 acknowledge time?',
    'For a P1, what PagerDuty service do we page?',
    'How long does a prod-break-glass session last?'
  ];

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
    this.labContext.setPage('RAG Architect', 'Layers · graph · strategies');
    this.refreshStatus();
  }

  get selectedLayerData(): ArchLayer {
    return this.layers.find(layer => layer.id === this.selectedLayer) || this.layers[0];
  }

  get activeFlowIds(): Set<string> {
    return new Set(this.strategyFlows[this.selectedStrategy] || []);
  }

  setTab(tab: ViewTab): void {
    this.activeTab = tab;
  }

  selectLayer(layer: LayerId): void {
    this.selectedLayer = layer;
  }

  onNodeClick(node: FlowNode): void {
    if (node.layer !== 'io') {
      this.selectedLayer = node.layer;
    }
  }

  nodeClass(node: FlowNode): string {
    const active = this.activeFlowIds.has(node.id) || node.layer === 'knowledge';
    return [
      'flow-node',
      `flow-node--${node.layer}`,
      active ? 'flow-node--active' : 'flow-node--dim',
      this.selectedLayer === node.layer ? 'flow-node--focus' : ''
    ]
      .filter(Boolean)
      .join(' ');
  }

  edgeClass(edge: FlowEdge): string {
    const active =
      this.activeFlowIds.has(edge.from) &&
      (this.activeFlowIds.has(edge.to) || edge.kind === 'loop');
    return [
      'flow-edge',
      edge.kind === 'loop' ? 'flow-edge--loop' : '',
      edge.kind === 'fail' ? 'flow-edge--fail' : '',
      active ? 'flow-edge--active' : 'flow-edge--dim'
    ]
      .filter(Boolean)
      .join(' ');
  }

  edgePath(edge: FlowEdge): string {
    const from = this.baseNodes.find(n => n.id === edge.from);
    const to = this.baseNodes.find(n => n.id === edge.to);
    if (!from || !to) {
      return '';
    }
    const x1 = from.x + 54;
    const y1 = from.y;
    const x2 = to.x - 54;
    const y2 = to.y;
    if (edge.kind === 'loop') {
      const midX = (x1 + x2) / 2;
      return `M ${x1} ${y1} C ${midX} ${y1 - 70}, ${midX} ${y2 - 70}, ${x2} ${y2}`;
    }
    const mid = (x1 + x2) / 2;
    return `M ${x1} ${y1} C ${mid} ${y1}, ${mid} ${y2}, ${x2} ${y2}`;
  }

  edgeLabelPos(edge: FlowEdge): { x: number; y: number } | null {
    if (!edge.label) {
      return null;
    }
    const from = this.baseNodes.find(n => n.id === edge.from);
    const to = this.baseNodes.find(n => n.id === edge.to);
    if (!from || !to) {
      return null;
    }
    return {
      x: (from.x + to.x) / 2,
      y: (from.y + to.y) / 2 - (edge.kind === 'loop' ? 48 : 10)
    };
  }

  tryWorkedExample(): void {
    this.selectedStrategy = 'hybrid';
    this.activeTab = 'ask';
    this.form.controls.question.setValue(this.workedExample.question);
  }

  refreshStatus(): void {
    this.http.get<StatusResponse>(`${environment.apiUrl}/api/rag-architect/status`).subscribe({
      next: response => {
        this.isOnline = true;
        this.strategies = response.strategies?.length
          ? response.strategies
          : this.strategies;
        this.selectedStrategy = response.default_strategy || 'hybrid';
        this.chatModel = response.chat_model;
        this.embedModel = response.embedding_model;
        this.documentCount = response.index?.documents ?? 0;
        this.chunkCount = response.index?.chunks ?? 0;
        this.errorMessage = '';
      },
      error: () => {
        this.isOnline = false;
        this.errorMessage = 'API offline. Start uvicorn on port 8000.';
      }
    });
  }

  selectStrategy(strategy: Strategy): void {
    this.selectedStrategy = strategy;
    if (strategy === 'crag' || strategy === 'agentic') {
      this.selectedLayer = 'validation';
    } else if (strategy === 'baseline' || strategy === 'hybrid' || strategy === 'hyde' || strategy === 'graph') {
      this.selectedLayer = 'retrieval';
    }
  }

  useSuggestion(text: string): void {
    this.form.controls.question.setValue(text);
    this.activeTab = 'ask';
  }

  rebuildIndex(): void {
    if (this.isBusy) {
      return;
    }
    this.isBusy = true;
    this.errorMessage = '';
    this.http.post<{ index: StatusResponse['index'] }>(
      `${environment.apiUrl}/api/rag-architect/rebuild`,
      {}
    ).subscribe({
      next: response => {
        this.documentCount = response.index?.documents ?? this.documentCount;
        this.chunkCount = response.index?.chunks ?? this.chunkCount;
        this.isBusy = false;
      },
      error: (error: HttpErrorResponse) => {
        this.isBusy = false;
        this.errorMessage = error.error?.detail || 'Rebuild failed';
      }
    });
  }

  ask(): void {
    if (this.form.invalid || this.isBusy) {
      return;
    }
    const question = this.form.controls.question.value.trim();
    if (!question) {
      return;
    }
    this.isBusy = true;
    this.errorMessage = '';
    this.turns = [
      ...this.turns,
      { role: 'user', content: question, strategy: this.selectedStrategy }
    ];
    this.form.reset({ question: '' });

    this.http
      .post<AskResponse>(`${environment.apiUrl}/api/rag-architect/ask`, {
        question,
        strategy: this.selectedStrategy
      })
      .subscribe({
        next: response => {
          this.turns = [
            ...this.turns,
            {
              role: 'assistant',
              content: response.answer,
              strategy: response.strategy,
              sources: response.sources,
              grade: response.grade,
              verified: response.verified
            }
          ];
          this.lastHits = response.hits || [];
          this.lastNotes = response.notes || [];
          this.isBusy = false;
          this.selectedLayer = 'validation';
        },
        error: (error: HttpErrorResponse) => {
          this.isBusy = false;
          this.errorMessage = error.error?.detail || 'Ask failed';
        }
      });
  }

  runEval(): void {
    if (this.isEvalBusy || this.isBusy) {
      return;
    }
    this.isEvalBusy = true;
    this.errorMessage = '';
    this.http
      .post<EvalResponse>(`${environment.apiUrl}/api/rag-architect/eval`, {
        strategies: ['baseline', 'hybrid', 'crag']
      })
      .subscribe({
        next: response => {
          this.evalRows = Object.entries(response.metrics || {}).map(([strategy, m]) => ({
            strategy,
            recall: m.context_recall,
            faith: m.faithfulness,
            cite: m.citation_rate
          }));
          this.isEvalBusy = false;
          this.selectedLayer = 'validation';
        },
        error: (error: HttpErrorResponse) => {
          this.isEvalBusy = false;
          this.errorMessage = error.error?.detail || 'Eval failed';
        }
      });
  }

  get canClear(): boolean {
    return (
      this.turns.length > 0 ||
      this.lastHits.length > 0 ||
      this.lastNotes.length > 0 ||
      this.evalRows.length > 0 ||
      !!this.errorMessage ||
      !!this.form.controls.question.value.trim()
    );
  }

  clearAll(): void {
    this.turns = [];
    this.lastHits = [];
    this.lastNotes = [];
    this.evalRows = [];
    this.errorMessage = '';
    this.form.reset({ question: '' });
  }
}

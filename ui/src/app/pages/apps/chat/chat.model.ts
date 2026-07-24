export interface ToolEvent {
  name: string;
  args: Record<string, unknown>;
  result: string;
}

export interface RagChunk {
  rank: number;
  source: string;
  visibility?: string;
  private?: boolean;
  start_index: number;
  content: string;
}

export interface RagKnowledgeFile {
  name: string;
  path: string;
  visibility: string;
  private: boolean;
}

export interface ApiChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface StateSnapshot {
  message_count: number;
  last_message_type: string;
  memory_enabled: boolean;
}

export interface TraceStep {
  sequence: number;
  node: string;
  summary: string;
  state: StateSnapshot;
  edge_from?: string | null;
  edge_to?: string | null;
  decision?: string | null;
  tool_names?: string[];
}

export interface TopologyNode {
  id: string;
  label: string;
  kind: 'start' | 'agent' | 'tools' | 'end' | 'conditional' | string;
}

export interface TopologyEdge {
  id: string;
  source: string;
  target: string;
  kind: 'normal' | 'conditional' | 'loop' | string;
  label?: string;
}

export interface GraphAnatomy {
  nodes: string[];
  node_count: number;
  edges: string[];
  edge_count: number;
  conditional_edges: string[];
  conditional_count: number;
  tools: string[];
  tool_count: number;
  state_keys: string[];
  topology_nodes: TopologyNode[];
  topology_edges: TopologyEdge[];
}

export interface ExecutionStats {
  steps: number;
  nodes_visited: number;
  agent_runs: number;
  tool_node_runs: number;
  edges_traversed: number;
  conditional_decisions: number;
  tool_calls: number;
  unique_tools: number;
  loops: number;
  state_messages: number;
  path: string[];
}

export interface ChatResponse {
  reply: string;
  thread_id: string;
  concept_id?: string;
  concept_title?: string;
  messages: ApiChatMessage[];
  tool_events: ToolEvent[];
  trace: TraceStep[];
  state: StateSnapshot;
  graph: GraphAnatomy;
  stats: ExecutionStats;
  interrupted?: boolean;
  pending?: Record<string, unknown> | null;
  state_extra?: Record<string, unknown>;
}

export interface ConceptInfo {
  id: string;
  title: string;
  phase: string;
  summary: string;
  teach: string[];
  needs_ollama: boolean;
  supports_hitl: boolean;
  sample_prompts: string[];
  tools: string[];
  state_keys: string[];
  topology_nodes: TopologyNode[];
  topology_edges: TopologyEdge[];
  node_count: number;
  edge_count: number;
  conditional_count: number;
  tool_count: number;
}

export interface UiMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  time: Date;
  toolEvents?: ToolEvent[];
  stats?: ExecutionStats;
  interrupted?: boolean;
  webResults?: Array<{ title: string; url: string; snippet: string }>;
}

export interface HealthResponse {
  status: string;
  model: string;
  graph?: GraphAnatomy;
  concepts?: ConceptInfo[];
}

export interface GraphNodeView {
  id: string;
  label: string;
  kind: string;
  x: number;
  y: number;
  active: boolean;
  visited: boolean;
}

export interface GraphEdgeView {
  id: string;
  source: string;
  target: string;
  kind: string;
  label: string;
  d: string;
  labelX: number;
  labelY: number;
  active: boolean;
  visited: boolean;
}

export interface PathNodeView {
  id: string;
  label: string;
  kind: string;
  index: number;
  active: boolean;
  done: boolean;
}

export const EMPTY_STATS: ExecutionStats = {
  steps: 0,
  nodes_visited: 0,
  agent_runs: 0,
  tool_node_runs: 0,
  edges_traversed: 0,
  conditional_decisions: 0,
  tool_calls: 0,
  unique_tools: 0,
  loops: 0,
  state_messages: 0,
  path: []
};

export const DEFAULT_GRAPH: GraphAnatomy = {
  nodes: ['__start__', 'agent', 'tools', '__end__'],
  node_count: 4,
  edges: [
    '__start__ → agent',
    'tools → agent',
    'agent → __end__ (when no tool_calls)'
  ],
  edge_count: 4,
  conditional_edges: ['agent → tools_condition → tools | __end__'],
  conditional_count: 1,
  tools: ['calculator', 'get_time', 'word_count'],
  tool_count: 3,
  state_keys: ['messages'],
  topology_nodes: [
    { id: '__start__', label: 'START', kind: 'start' },
    { id: 'agent', label: 'agent', kind: 'agent' },
    { id: 'tools_condition', label: 'tools_condition', kind: 'conditional' },
    { id: 'tools', label: 'tools', kind: 'tools' },
    { id: '__end__', label: 'END', kind: 'end' }
  ],
  topology_edges: [
    { id: 'e_start_agent', source: '__start__', target: 'agent', kind: 'normal', label: '' },
    { id: 'e_agent_cond', source: 'agent', target: 'tools_condition', kind: 'normal', label: '' },
    { id: 'e_cond_tools', source: 'tools_condition', target: 'tools', kind: 'conditional', label: 'yes' },
    { id: 'e_cond_end', source: 'tools_condition', target: '__end__', kind: 'conditional', label: 'no' },
    { id: 'e_tools_agent', source: 'tools', target: 'agent', kind: 'loop', label: 'loop' }
  ]
};

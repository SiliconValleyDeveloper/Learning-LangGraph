import {
  Component,
  ElementRef,
  HostListener,
  OnDestroy,
  OnInit,
  ViewChild,
} from "@angular/core";
import { HttpErrorResponse } from "@angular/common/http";

import {
  ShippingApiService,
  ShippingGraph,
  ShippingHealth,
  ShippingChoice,
  ShippingRecovery,
  ShippingRunResult,
} from "./shipping-api.service";

type ResultView = "summary" | "json" | "graph";
type ChatRole = "assistant" | "user" | "system";

interface ChatMessage {
  id: number;
  role: ChatRole;
  content: string;
  meta?: string;
  time: string;
  typing?: boolean;
  chips?: string[];
  choices?: ShippingChoice[];
  choicesActive?: boolean;
}

type GraphShape = "terminal" | "agent" | "tool" | "conditional" | "hitl";

interface VisualGraphNode {
  id: string;
  stateId: string;
  label: string;
  detail: string;
  x: number;
  y: number;
  width: number;
  height: number;
  shape: GraphShape;
}

interface VisualGraphEdge {
  id: string;
  source: string;
  target: string;
  path: string;
  label: string;
  labelX: number;
  labelY: number;
  condition?:
    | "chat"
    | "rag"
    | "db"
    | "write"
    | "retry"
    | "fix"
    | "blocked"
    | "db_fallback";
}

@Component({
  selector: "app-shipping-console",
  templateUrl: "./shipping-console.component.html",
  styleUrls: ["./shipping-console.component.scss"],
  standalone: false,
})
export class ShippingConsoleComponent implements OnInit, OnDestroy {
  @ViewChild("chatWindow") private chatWindow?: ElementRef<HTMLDivElement>;
  @ViewChild("messageInput") private messageInput?: ElementRef<HTMLTextAreaElement>;

  health: ShippingHealth | null = null;
  graph: ShippingGraph | null = null;
  result: ShippingRunResult | null = null;
  messages: ChatMessage[] = [
    {
      id: 1,
      role: "assistant",
      content:
        "Hello — I’m your shipping operations copilot. I can search sailings, prepare quotations, create bookings with human approval, and track shipments. What would you like to do?",
      meta: "Shipping assistant",
      time: this.now(),
    },
  ];

  prompt = "";
  reviewer = "operations.manager@example.com";
  reviewerNote = "";
  view: ResultView = "graph";

  loading = false;
  approvalLoading = false;
  typing = false;
  chatOpen = false;
  graphZoom = 0.8;
  flowGuideOpen = false;
  error = "";
  private messageId = 1;
  private typeTimer: number | null = null;

  /** Structured continuation while completing an incomplete request. */
  basePrompt = "";
  parameterPatches: Record<string, unknown> = {};
  activeRecovery: ShippingRecovery | null = null;
  continuationThreadId: string | null = null;
  activeChoiceMessageId: number | null = null;
  manualField = "";
  manualValue = "";

  readonly examples = [
    {
      label: "Database overview",
      prompt: "Give me an overview and counts of all shipping data",
    },
    {
      label: "Recent quotations",
      prompt: "List the latest 10 quotations for ACME-IN",
    },
    {
      label: "Booking counts",
      prompt: "How many confirmed bookings are there?",
    },
    {
      label: "What is sailing id?",
      prompt: "what is sailing id?",
    },
    {
      label: "Miami to London",
      prompt: "Find sailings from MIA to LON",
    },
    {
      label: "Search sailings",
      prompt: "Find sailings from INNSA to SGSIN",
    },
    {
      label: "Create quotation",
      prompt:
        "Create a quotation for ACME-IN from Miami to London Gateway on voyage MS410W aboard MV Meridian Star for 2 x 40GP containers carrying 18,000 kg of general merchandise. No dangerous goods.",
    },
    {
      label: "Compliance block",
      prompt:
        "Create quotation for ACME-IN from INNSA to AEDXB, 1x20GP, 3000kg dangerous goods",
    },
  ];

  readonly visualGraphNodes: VisualGraphNode[] = [
    { id: "start", stateId: "__start__", label: "START", detail: "prompt + history", x: 560, y: 10, width: 90, height: 50, shape: "terminal" },
    { id: "intent", stateId: "intent", label: "intent router", detail: "rules · Qwen · history", x: 490, y: 90, width: 220, height: 90, shape: "conditional" },
    { id: "chat", stateId: "chat", label: "chat reply", detail: "Qwen, no tools", x: 80, y: 250, width: 180, height: 70, shape: "agent" },
    { id: "rewrite", stateId: "rewrite", label: "rewrite query", detail: "preserve identifiers", x: 400, y: 250, width: 180, height: 70, shape: "agent" },
    { id: "operations", stateId: "operations", label: "operations", detail: "SQL read / write validate", x: 900, y: 250, width: 180, height: 70, shape: "agent" },
    { id: "retrieve", stateId: "retrieve", label: "retrieve", detail: "PostgreSQL + policy", x: 400, y: 365, width: 180, height: 70, shape: "tool" },
    { id: "db_answer", stateId: "db_answer", label: "DB answer", detail: "compose SQL result", x: 1120, y: 365, width: 180, height: 70, shape: "agent" },
    { id: "pricing", stateId: "pricing", label: "pricing", detail: "proposal only", x: 900, y: 365, width: 180, height: 70, shape: "agent" },
    { id: "rerank", stateId: "rerank", label: "dynamic rerank", detail: "relevance top-k", x: 400, y: 480, width: 180, height: 70, shape: "agent" },
    { id: "risk", stateId: "risk", label: "risk gate", detail: "block or review", x: 890, y: 480, width: 200, height: 90, shape: "conditional" },
    { id: "grade", stateId: "grade", label: "grade evidence", detail: "pass · weak · fail", x: 390, y: 595, width: 200, height: 90, shape: "conditional" },
    { id: "approval", stateId: "approval_request", label: "checkpoint", detail: "persist proposal", x: 900, y: 610, width: 180, height: 70, shape: "tool" },
    { id: "generate", stateId: "generate", label: "generate", detail: "grounded + [S#]", x: 400, y: 735, width: 180, height: 70, shape: "agent" },
    { id: "human", stateId: "human", label: "HUMAN", detail: "interrupt / decision", x: 900, y: 720, width: 180, height: 80, shape: "hitl" },
    { id: "fix", stateId: "fix", label: "fix answer", detail: "bounded repair", x: 100, y: 850, width: 180, height: 70, shape: "agent" },
    { id: "verify", stateId: "verify", label: "verify answer", detail: "citations + refs", x: 390, y: 840, width: 200, height: 90, shape: "conditional" },
    { id: "execute", stateId: "execute", label: "execute", detail: "recheck + write + audit", x: 900, y: 850, width: 180, height: 70, shape: "tool" },
    { id: "response", stateId: "response", label: "response", detail: "answer · JSON · trace", x: 670, y: 980, width: 180, height: 70, shape: "agent" },
    { id: "end", stateId: "__end__", label: "END", detail: "completed", x: 720, y: 1080, width: 80, height: 54, shape: "terminal" },
  ];

  readonly visualGraphEdges: VisualGraphEdge[] = [
    { id: "start-intent", source: "start", target: "intent", path: "M605 60 L600 90", label: "prompt + history", labelX: 620, labelY: 72 },
    { id: "intent-chat", source: "intent", target: "chat", path: "M490 135 C340 135 170 185 170 250", label: "chat", labelX: 320, labelY: 150, condition: "chat" },
    { id: "intent-rewrite", source: "intent", target: "rewrite", path: "M560 180 L500 250", label: "rag", labelX: 500, labelY: 210, condition: "rag" },
    { id: "intent-operations-db", source: "intent", target: "operations", path: "M710 135 C820 135 960 185 990 250", label: "db", labelX: 800, labelY: 140, condition: "db" },
    { id: "intent-operations", source: "intent", target: "operations", path: "M710 160 C840 185 1000 210 990 250", label: "write", labelX: 860, labelY: 180, condition: "write" },
    { id: "chat-response", source: "chat", target: "response", path: "M170 320 C170 940 500 1015 670 1015", label: "direct answer", labelX: 185, labelY: 610, condition: "chat" },
    { id: "rewrite-retrieve", source: "rewrite", target: "retrieve", path: "M490 320 L490 365", label: "query", labelX: 510, labelY: 348 },
    { id: "retrieve-rerank", source: "retrieve", target: "rerank", path: "M490 435 L490 480", label: "candidates", labelX: 525, labelY: 463 },
    { id: "rerank-grade", source: "rerank", target: "grade", path: "M490 550 L490 595", label: "top-k", labelX: 515, labelY: 578 },
    { id: "grade-rewrite", source: "grade", target: "rewrite", path: "M390 640 C300 640 300 285 400 285", label: "fail → retry", labelX: 285, labelY: 470, condition: "retry" },
    { id: "grade-generate", source: "grade", target: "generate", path: "M490 685 L490 735", label: "pass / give up", labelX: 540, labelY: 715 },
    { id: "grade-operations", source: "grade", target: "operations", path: "M590 640 C780 640 930 400 990 320", label: "db fallback", labelX: 780, labelY: 560, condition: "db_fallback" },
    { id: "generate-verify", source: "generate", target: "verify", path: "M490 805 L490 840", label: "draft", labelX: 512, labelY: 828 },
    { id: "verify-fix", source: "verify", target: "fix", path: "M390 875 C350 850 320 850 280 875", label: "unsupported", labelX: 330, labelY: 842, condition: "fix" },
    { id: "fix-verify", source: "fix", target: "verify", path: "M280 915 C325 965 365 965 410 930", label: "recheck", labelX: 335, labelY: 972, condition: "fix" },
    { id: "verify-response", source: "verify", target: "response", path: "M590 895 C640 895 705 930 760 980", label: "verified", labelX: 675, labelY: 925 },
    { id: "operations-pricing", source: "operations", target: "pricing", path: "M990 320 L990 365", label: "valid write", labelX: 1010, labelY: 348, condition: "write" },
    { id: "operations-db-answer", source: "operations", target: "db_answer", path: "M1080 285 C1120 300 1180 330 1210 365", label: "db answer", labelX: 1145, labelY: 310, condition: "db" },
    { id: "db-answer-response", source: "db_answer", target: "response", path: "M1210 435 C1210 900 900 1015 850 1015", label: "sql result", labelX: 1180, labelY: 700, condition: "db" },
    { id: "operations-response", source: "operations", target: "response", path: "M1080 285 C1170 285 1170 1015 850 1015", label: "invalid", labelX: 1195, labelY: 610, condition: "blocked" },
    { id: "pricing-risk", source: "pricing", target: "risk", path: "M990 435 L990 480", label: "proposal", labelX: 1020, labelY: 462 },
    { id: "risk-response", source: "risk", target: "response", path: "M890 525 C790 525 760 800 760 980", label: "hard block", labelX: 790, labelY: 660, condition: "blocked" },
    { id: "risk-approval", source: "risk", target: "approval", path: "M990 570 L990 610", label: "reviewable", labelX: 1025, labelY: 598 },
    { id: "approval-human", source: "approval", target: "human", path: "M990 680 L990 720", label: "interrupt", labelX: 1020, labelY: 708 },
    { id: "human-execute", source: "human", target: "execute", path: "M990 800 L990 850", label: "approve / reject", labelX: 1045, labelY: 835 },
    { id: "execute-response", source: "execute", target: "response", path: "M900 885 C860 900 835 950 820 980", label: "result", labelX: 860, labelY: 940 },
    { id: "response-end", source: "response", target: "end", path: "M760 1050 L760 1080", label: "", labelX: 780, labelY: 1070 },
  ];

  constructor(private readonly shippingApi: ShippingApiService) {}

  ngOnInit(): void {
    this.shippingApi.health().subscribe({
      next: (health) => {
        this.health = health;
      },
      error: (err: HttpErrorResponse) => {
        this.health = {
          status: "offline",
          postgres_ok: false,
          error: err.message,
        };
      },
    });
    this.shippingApi.graph().subscribe({
      next: (graph) => {
        this.graph = graph;
      },
      error: () => {
        this.graph = null;
      },
    });
  }

  ngOnDestroy(): void {
    this.clearTypeTimer();
  }

  @HostListener("document:keydown.escape")
  closeOnEscape(): void {
    this.closeChat();
  }

  toggleChat(): void {
    this.chatOpen ? this.closeChat() : this.openChat();
  }

  openChat(): void {
    this.chatOpen = true;
    window.setTimeout(() => {
      this.scrollToBottom();
      this.messageInput?.nativeElement.focus();
    }, 180);
  }

  closeChat(): void {
    this.chatOpen = false;
  }

  useExample(prompt: string): void {
    this.clearRecovery();
    this.prompt = prompt;
    this.openChat();
    window.setTimeout(() => this.run(), 0);
  }

  selectChoice(choice: ShippingChoice, messageId?: number): void {
    if (this.loading || this.approvalLoading || this.typing) {
      return;
    }
    if (
      messageId != null &&
      this.activeChoiceMessageId != null &&
      messageId !== this.activeChoiceMessageId
    ) {
      return;
    }
    if (choice.kind === "dismiss") {
      this.clearRecovery("Started a new request.");
      return;
    }
    const patches = this.parseChoiceValue(choice);
    this.addMessage("user", choice.label, "Selected option");
    this.continueWithPatches(patches);
  }

  applyManualField(): void {
    const field = this.manualField.trim();
    const value = this.manualValue.trim();
    if (!field || !value || this.loading || this.approvalLoading || this.typing) {
      return;
    }
    this.addMessage("user", `${field} = ${value}`, "Manual entry");
    this.manualValue = "";
    this.continueWithPatches({ [field]: value });
  }

  clearRecovery(systemNote?: string): void {
    this.basePrompt = "";
    this.parameterPatches = {};
    this.activeRecovery = null;
    this.continuationThreadId = null;
    this.activeChoiceMessageId = null;
    this.manualField = "";
    this.manualValue = "";
    this.deactivateOldChoices();
    if (systemNote) {
      this.addMessage("system", systemNote, "Request reset");
    }
  }

  setView(view: ResultView): void {
    this.view = view;
  }

  zoomGraph(delta: number): void {
    this.graphZoom = Math.min(1.4, Math.max(0.6, this.graphZoom + delta));
  }

  resetGraphZoom(): void {
    this.graphZoom = 0.8;
  }

  toggleFlowGuide(): void {
    this.flowGuideOpen = !this.flowGuideOpen;
  }

  sendOnEnter(event: KeyboardEvent): void {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      this.run();
    }
  }

  run(): void {
    const prompt = this.prompt.trim();
    if (!prompt || this.loading || this.approvalLoading || this.typing) {
      return;
    }
    this.addMessage("user", prompt);
    this.prompt = "";
    if (this.result?.interrupted) {
      const decision = this.approvalDecision(prompt);
      if (decision === null) {
        this.addMessage(
          "assistant",
          'A database write is waiting. Please explicitly reply "approve" or "reject", or use the buttons in this message.',
          "Approval required",
        );
        return;
      }
      this.decide(decision, false);
      return;
    }

    if (this.activeRecovery?.active) {
      if (this.looksLikeNewIntent(prompt)) {
        this.clearRecovery();
        this.executeRun({ prompt });
        return;
      }
      const field =
        this.activeRecovery.missing_fields[0] ||
        this.activeRecovery.invalid_fields[0] ||
        this.manualField ||
        "customer_code";
      this.continueWithPatches({ [field]: prompt }, /* skipUserEcho */ true);
      return;
    }

    this.executeRun({ prompt });
  }

  decide(approve: boolean, announce = true): void {
    if (!this.result?.interrupted || !this.reviewer.trim()) {
      return;
    }
    this.approvalLoading = true;
    this.error = "";
    if (announce) {
      this.addMessage(
        "system",
        `${approve ? "Approved" : "Rejected"} by ${this.reviewer.trim()}${
          this.reviewerNote.trim() ? ` — ${this.reviewerNote.trim()}` : ""
        }`,
        "Human decision",
      );
    }
    this.shippingApi
      .decide(
        this.result.thread_id,
        approve,
        this.reviewer.trim(),
        this.reviewerNote.trim(),
      )
      .subscribe({
        next: (result) => {
          this.approvalLoading = false;
          this.result = result;
          this.clearRecovery();
          this.graph = result.graph;
          this.view = "graph";
          this.typeOut(
            result.assistant_message,
            "Shipping assistant",
            this.toolChips(result),
          );
        },
        error: (err: HttpErrorResponse) => {
          this.approvalLoading = false;
          this.error = err.error?.detail || err.message;
          this.typeOut(
            `The approval decision failed: ${this.error}`,
            "Error",
          );
        },
      });
  }

  private continueWithPatches(
    patch: Record<string, unknown>,
    skipUserEcho = false,
  ): void {
    void skipUserEcho;
    this.parameterPatches = { ...this.parameterPatches, ...patch };
    this.executeRun({
      prompt: this.basePrompt || Object.values(patch).join(" "),
      base_prompt: this.basePrompt || undefined,
      patches: this.parameterPatches,
      thread_id: this.continuationThreadId,
    });
  }

  private executeRun(request: {
    prompt: string;
    base_prompt?: string;
    patches?: Record<string, unknown>;
    thread_id?: string | null;
  }): void {
    this.loading = true;
    this.error = "";
    this.result = null;
    this.shippingApi
      .run({
        ...request,
        history: this.recentChatHistory(request.prompt),
      })
      .subscribe({
      next: (result) => {
        this.loading = false;
        this.result = result;
        this.graph = result.graph;
        this.view = "graph";
        this.applyRecoveryState(result, request.prompt, request.base_prompt);
        this.typeOut(
          result.assistant_message,
          result.interrupted ? "Waiting for approval" : "Shipping assistant",
          this.toolChips(result),
          result.choices,
        );
      },
      error: (err: HttpErrorResponse) => {
        this.loading = false;
        this.error = err.error?.detail || err.message;
        this.typeOut(
          `I couldn't complete that request: ${this.error}`,
          "Error",
        );
      },
    });
  }

  private recentChatHistory(excludePrompt?: string): Array<{
    role: "user" | "assistant" | "system";
    content: string;
  }> {
    const turns = this.messages
      .filter((message) => !message.typing && message.content.trim())
      .map((message) => ({
        role: message.role,
        content: message.content.trim().slice(0, 800),
      }));
    if (
      excludePrompt &&
      turns.length &&
      turns[turns.length - 1].role === "user" &&
      turns[turns.length - 1].content === excludePrompt.trim()
    ) {
      turns.pop();
    }
    return turns.slice(-8);
  }

  private applyRecoveryState(
    result: ShippingRunResult,
    prompt: string,
    basePrompt?: string,
  ): void {
    const recovery = result.recovery;
    if (recovery?.active) {
      this.basePrompt = basePrompt || this.basePrompt || prompt;
      this.continuationThreadId = result.thread_id;
      this.activeRecovery = recovery;
      const scalarFilled: Record<string, unknown> = {};
      for (const [key, value] of Object.entries(recovery.filled || {})) {
        if (
          value == null ||
          typeof value === "object" ||
          key.startsWith("unrecognized_") ||
          key === "selected_sailing"
        ) {
          continue;
        }
        scalarFilled[key] = value;
      }
      this.parameterPatches = {
        ...this.parameterPatches,
        ...scalarFilled,
      };
      if (!this.manualField) {
        this.manualField =
          recovery.missing_fields[0] || recovery.invalid_fields[0] || "";
      }
      return;
    }
    this.basePrompt = "";
    this.parameterPatches = {};
    this.activeRecovery = null;
    this.continuationThreadId = null;
    this.activeChoiceMessageId = null;
  }

  private looksLikeNewIntent(prompt: string): boolean {
    const normalized = prompt.trim().toLowerCase();
    if (normalized.length > 80) {
      return true;
    }
    if (
      /\b(more detail|more details|tell me more|elaborate|matching\b.{0,40}\brecords?)\b/.test(
        normalized,
      )
    ) {
      return true;
    }
    return /^(find|search|create|list|how many|track|show|give me|what is|overview)\b/.test(
      normalized,
    );
  }

  private fieldFromChoice(choice: ShippingChoice): string {
    if (choice.field) {
      return choice.field;
    }
    const kind = String(choice.kind || "");
    if (kind.includes("customer")) {
      return "customer_code";
    }
    if (kind.includes("sailing") || kind.includes("voyage")) {
      return "voyage_number";
    }
    if (kind.includes("origin") || kind === "port") {
      return "origin";
    }
    if (kind.includes("destination")) {
      return "destination";
    }
    if (kind.includes("quote")) {
      return "quote_ref";
    }
    if (kind.includes("booking")) {
      return "booking_ref";
    }
    if (kind.includes("container")) {
      return "container_type";
    }
    if (kind.includes("entity")) {
      return "entity";
    }
    if (kind.includes("status")) {
      return "status";
    }
    return "customer_code";
  }

  private parseChoiceValue(choice: ShippingChoice): Record<string, unknown> {
    const raw = (choice.value || "").trim();
    const patches: Record<string, unknown> = {};
    if (!raw.includes(":")) {
      const field = choice.field || this.fieldFromChoice(choice);
      patches[field] = raw;
      return patches;
    }
    for (const part of raw.split(/,\s*/)) {
      if (!part.includes(":")) {
        continue;
      }
      const index = part.indexOf(":");
      const key = part.slice(0, index).trim();
      const value = part.slice(index + 1).trim();
      if (key === "sailing_id") {
        const asNumber = Number(value);
        patches[key] = Number.isFinite(asNumber) ? asNumber : value;
      } else if (key === "container_qty" || key === "limit") {
        const asNumber = Number(value);
        if (Number.isFinite(asNumber)) {
          patches[key] = asNumber;
        }
      } else if (key === "cargo_weight_kg") {
        const asNumber = Number(value.replace(/,/g, ""));
        patches[key] = Number.isFinite(asNumber) ? asNumber : value;
      } else if (key === "dangerous_goods") {
        patches[key] = ["true", "1", "yes"].includes(value.toLowerCase());
      } else {
        patches[key] = value;
      }
    }
    if (!Object.keys(patches).length && choice.field) {
      patches[choice.field] = raw;
    }
    return patches;
  }

  private deactivateOldChoices(): void {
    this.messages = this.messages.map((message) =>
      message.choices?.length
        ? { ...message, choicesActive: false }
        : message,
    );
  }

  nodeState(id: string): "idle" | "active" | "waiting" | "blocked" {
    if (!this.result) {
      return "idle";
    }
    if (id === "__start__") {
      return "active";
    }
    if (id === "human" && this.result.interrupted) {
      return "waiting";
    }
    if (id === "human" && this.result.response?.approval?.status) {
      return "active";
    }
    if (
      id === "risk" &&
      (this.result.state.risk_review?.hard_blocks?.length || 0) > 0
    ) {
      return "blocked";
    }
    const map: Record<string, string[]> = {
      intent: ["intent_agent", "understand_agent"],
      rewrite: ["rewrite_agent"],
      retrieve: ["retrieve_agent"],
      rerank: ["rerank_agent"],
      grade: ["grade_agent"],
      generate: ["generate_agent"],
      verify: ["verify_agent"],
      fix: ["fix_agent"],
      operations: ["operations_agent"],
      db_answer: ["db_answer_agent"],
      pricing: ["pricing_agent"],
      risk: ["risk_agent"],
      approval_request: ["approval_agent"],
      execute: ["execution_agent"],
      response: ["response_agent"],
    };
    const agents = new Set(this.result.trace.map((step) => step.agent));
    if ((map[id] || []).some((agent) => agents.has(agent))) {
      return "active";
    }
    if (id === "__end__" && !this.result.interrupted) {
      return "active";
    }
    return "idle";
  }

  nodeLabel(id: string, fallback: string): string {
    return this.graph?.nodes.find((node) => node.id === id)?.label || fallback;
  }

  visualNodeState(
    node: VisualGraphNode,
  ): "idle" | "active" | "waiting" | "blocked" {
    if (node.id === "chat") {
      return this.result?.state.lane === "chat" ? "active" : "idle";
    }
    return this.nodeState(node.stateId);
  }

  visualNodePoints(node: VisualGraphNode): string {
    if (node.shape === "conditional") {
      return [
        `${node.x + node.width / 2},${node.y}`,
        `${node.x + node.width},${node.y + node.height / 2}`,
        `${node.x + node.width / 2},${node.y + node.height}`,
        `${node.x},${node.y + node.height / 2}`,
      ].join(" ");
    }
    const inset = 24;
    return [
      `${node.x + inset},${node.y}`,
      `${node.x + node.width - inset},${node.y}`,
      `${node.x + node.width},${node.y + node.height / 2}`,
      `${node.x + node.width - inset},${node.y + node.height}`,
      `${node.x + inset},${node.y + node.height}`,
      `${node.x},${node.y + node.height / 2}`,
    ].join(" ");
  }

  visualEdgeActive(edge: VisualGraphEdge): boolean {
    if (!this.result) {
      return false;
    }
    const lane = this.result.state.lane;
    const agents = new Set(this.result.trace.map((step) => step.agent));
    if (edge.condition === "chat" && lane !== "chat") {
      return false;
    }
    if (edge.condition === "rag" && lane !== "rag") {
      return false;
    }
    if (edge.condition === "db" && lane !== "db") {
      return false;
    }
    if (edge.condition === "write" && lane !== "write") {
      return false;
    }
    if (edge.condition === "db_fallback") {
      // Active only when RAG graded fail and then operations/db_answer ran.
      if (
        !agents.has("grade_agent") ||
        !agents.has("operations_agent") ||
        !agents.has("db_answer_agent") ||
        agents.has("generate_agent")
      ) {
        return false;
      }
    }
    if (
      edge.condition === "retry" &&
      (this.result.state.retrieval_attempts || 0) < 2
    ) {
      return false;
    }
    if (edge.condition === "fix" && (this.result.state.fix_attempts || 0) < 1) {
      return false;
    }
    if (edge.id === "operations-response" && !this.result.state.errors.length) {
      return false;
    }
    if (
      edge.id === "risk-response" &&
      !(this.result.state.risk_review?.hard_blocks?.length || 0)
    ) {
      return false;
    }
    const source = this.visualGraphNodes.find((node) => node.id === edge.source);
    const target = this.visualGraphNodes.find((node) => node.id === edge.target);
    if (!source || !target) {
      return false;
    }
    return (
      this.visualNodeState(source) !== "idle" &&
      this.visualNodeState(target) !== "idle"
    );
  }

  get toolCallCount(): number {
    if (this.result?.state.action === "conversation") {
      return 0;
    }
    if (this.result?.state.lane === "rag") {
      return (
        this.result.trace.filter((step) => step.agent === "retrieve_agent").length ||
        0
      );
    }
    const toolAgents = new Set([
      "retrieve_agent",
      "operations_agent",
      "db_answer_agent",
      "pricing_agent",
      "execution_agent",
    ]);
    return this.result?.trace.filter((step) => toolAgents.has(step.agent)).length || 0;
  }

  get agentRunCount(): number {
    return this.result?.trace.length || 0;
  }

  get loopCount(): number {
    const retrievalRetries = Math.max(
      0,
      (this.result?.state.retrieval_attempts || 0) - 1,
    );
    return retrievalRetries + (this.result?.state.fix_attempts || 0);
  }

  get stateMessageCount(): number {
    return Math.max(0, this.messages.length - 1);
  }

  get conditionalNodeCount(): number {
    const outgoing = new Map<string, number>();
    for (const edge of this.graph?.edges || []) {
      outgoing.set(edge.source, (outgoing.get(edge.source) || 0) + 1);
    }
    return Array.from(outgoing.values()).filter((count) => count > 1).length;
  }

  get runPath(): string[] {
    if (!this.result) {
      return [];
    }
    const labels = this.result.trace.map((step) =>
      step.agent.replace(/_agent$/, "").replaceAll("_", " "),
    );
    if (this.result.interrupted) {
      labels.push("human approval");
    }
    return labels;
  }

  get lastRunType(): string {
    return this.result?.state.action?.replaceAll("_", " ") || "—";
  }

  get memoryLabel(): string {
    if (!this.result) {
      return "Ready";
    }
    return this.result.interrupted ? "Checkpoint saved" : "SqliteSaver";
  }

  get currentStepLabel(): string {
    const trace = this.result?.trace || [];
    const last = trace[trace.length - 1]?.agent;
    if (!last) {
      return "Waiting for a prompt";
    }
    return last.replace(/_agent$/, "").replaceAll("_", " ");
  }

  get executionLabel(): string {
    if (!this.result) {
      return "Ready";
    }
    if (this.result.interrupted) {
      return "Waiting for human approval";
    }
    if (this.result.state.errors.length) {
      return "Completed with issues";
    }
    if (this.result.state.lane === "rag" && this.result.state.verified) {
      return "Verified answer";
    }
    if (this.result.state.lane === "db") {
      return "DB answer";
    }
    return this.result.response?.status || this.result.state.status;
  }

  get shortThreadId(): string {
    return this.result?.thread_id?.slice(0, 8) || "—";
  }

  get recoveryGroups() {
    return this.activeRecovery?.groups || [];
  }

  get filledEntries(): Array<{ key: string; value: string }> {
    const filled = this.activeRecovery?.filled || {};
    return Object.keys(filled)
      .filter((key) => !["selected_sailing"].includes(key))
      .map((key) => ({ key, value: String(filled[key]) }));
  }

  get remainingFields(): string[] {
    const recovery = this.activeRecovery;
    if (!recovery) {
      return [];
    }
    return Array.from(
      new Set([...(recovery.missing_fields || []), ...(recovery.invalid_fields || [])]),
    );
  }

  get manualFieldOptions(): string[] {
    const known = [
      "customer_code",
      "origin",
      "destination",
      "voyage_number",
      "sailing_id",
      "container_type",
      "container_qty",
      "cargo_weight_kg",
      "quote_ref",
      "booking_ref",
      "entity",
      "status",
    ];
    return Array.from(new Set([...this.remainingFields, ...known]));
  }

  formatAction(action: string): string {
    return (action || "request").replace(/_/g, " ");
  }

  private addMessage(
    role: ChatRole,
    content: string,
    meta?: string,
    chips?: string[],
    choices?: ShippingChoice[],
  ): ChatMessage {
    if (choices?.length) {
      this.deactivateOldChoices();
    }
    const message: ChatMessage = {
      id: ++this.messageId,
      role,
      content,
      meta,
      time: this.now(),
      chips,
      choices,
      choicesActive: Boolean(choices?.length),
    };
    if (choices?.length) {
      this.activeChoiceMessageId = message.id;
    }
    this.messages = [...this.messages, message];
    this.scrollToBottom();
    return message;
  }

  private typeOut(
    content: string,
    meta?: string,
    chips?: string[],
    choices?: ShippingChoice[],
  ): void {
    this.clearTypeTimer();
    const message = this.addMessage("assistant", "", meta, chips, choices);
    const answer = content || "I completed the request.";
    const step = Math.max(1, Math.ceil(answer.length / 120));
    let position = 0;
    message.typing = true;
    this.typing = true;

    const tick = () => {
      position = Math.min(answer.length, position + step);
      message.content = answer.slice(0, position);
      this.messages = [...this.messages];
      this.scrollToBottom();
      if (position < answer.length) {
        this.typeTimer = window.setTimeout(tick, 18);
        return;
      }
      message.typing = false;
      this.typing = false;
      this.typeTimer = null;
      this.messages = [...this.messages];
      this.scrollToBottom();
    };
    tick();
  }

  private clearTypeTimer(): void {
    if (this.typeTimer) {
      window.clearTimeout(this.typeTimer);
      this.typeTimer = null;
    }
    this.typing = false;
  }

  private scrollToBottom(): void {
    window.setTimeout(() => {
      const element = this.chatWindow?.nativeElement;
      if (element) {
        element.scrollTop = element.scrollHeight;
      }
    }, 30);
  }

  private toolChips(result: ShippingRunResult): string[] {
    return Array.from(
      new Set(
        result.trace
          .map((step) => step.agent)
          .filter((agent) => agent && agent !== "response_agent"),
      ),
    ).slice(0, 4);
  }

  private now(): string {
    return new Intl.DateTimeFormat("en", {
      hour: "numeric",
      minute: "2-digit",
    }).format(new Date());
  }

  private approvalDecision(message: string): boolean | null {
    const normalized = message.trim().toLowerCase().replace(/[.!]+$/, "");
    if (
      /^(approve|approved|confirm|confirmed|yes,? approve|yes,? execute|proceed|go ahead)$/.test(
        normalized,
      )
    ) {
      return true;
    }
    if (
      /^(reject|rejected|decline|declined|deny|denied|do not execute|cancel)$/.test(
        normalized,
      )
    ) {
      return false;
    }
    return null;
  }

  get prettyJson(): string {
    return JSON.stringify(this.result, null, 2);
  }

  get backendHint(): string {
    if (!this.health) {
      return "Connecting…";
    }
    return this.health.status === "ok"
      ? "Qwen 3 · PostgreSQL connected"
      : "Shipping API offline";
  }

  get statusLabel(): string {
    if (!this.result) {
      return "Ready";
    }
    if (this.result.interrupted) {
      return "Awaiting human approval";
    }
    return this.result.response?.status || this.result.state.status;
  }
}

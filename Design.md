# Smart LLM Routing Proxy – Architectural Design Document

---

# 1. Executive Summary & Topology

The Smart LLM Routing Proxy is a stateless, performance-engineered orchestration layer situated directly between Anthropic-compatible (`/v1/messages`) execution clients (such as the Claude Code CLI) and heterogeneous upstream provider gateways via LiteLLM.

```text
+-------------------+           +-----------------------+           +-------------------+
|  Claude Code CLI  |  -------> |  Smart Routing Proxy  |  -------> |  LiteLLM Gateway  |
+-------------------+           +-----------------------+           +-------------------+
                                            |
                                            +---> [Sanitization & Repair]
                                            +---> [Tiered Prefix Cache]
                                            +---> [Intent Classifier]
```

The system targets runtime optimizations inside multi-turn, tool-heavy software engineering agent interactions. It minimizes token exposure footprint, enforces multi-provider wire-format normalization, and applies predictive economic model routing without mutating agent behavior or breaking active multi-turn tool loops.

---

# 2. Design Tenets & System Goals

## 2.1 Primary Goals

### Inference Cost Reduction
Mitigate runaway token spend by dynamically identifying and offloading low-to-medium complexity agent turns to cost-effective model families.

### Transparent Multi-Provider Interoperability
Sit invisibly between standard Anthropic client requests and upstream provider backends, handling data transformations transparently.

### Session Preservation
Secure multi-turn agent execution states over hours of operations, preventing syntax-level or protocol-level crashes during mid-tool loops.

### Auditable Financial Accounting
Compute real-time savings metrics split across model execution arbitrage and aggressive context-reduction heads.

---

## 2.2 Non-Goals

### Semantic Equivalence
The proxy does not guarantee identical qualitative reasoning characteristics between routed models; it optimizes for task-level competency thresholds.

### Persistent External State Management
The proxy operates without external database requirements, avoiding distributed coordination bottlenecks.

---

## 2.3 Core Design Principles

### Stateless Request Autonomy
The proxy acts as a pure, functional pass-through wrapper. Because client applications natively pass the full conversation timeline with every turn, the proxy treats each transaction as entirely self-contained. This allows for fluid mid-session provider adjustments and instant hot-swapping under transport errors.

### Safety Defensively Prioritized
Financial savings are aggressively sacrificed if payload complexity flags are raised. When ambiguous tool schemas, active file-mutation pipelines (`MultiEdit`), or structural anomalies are detected, the proxy forces an immediate fallback to the high-overhead baseline model (`claude-3-5-sonnet`).

---

# 3. Detailed Request Pipeline & Flow

When a client hits the `/v1/messages` endpoint, the request transitions through a deterministic execution and optimization pipeline:

```text
Client Request (POST /v1/messages)
                       │
                       ▼
         ┌───────────────────────────┐
         │  Header Key Extraction    │
         │  --> Extracts and enforces client-side API tokens
         └─────────────┬-------------┘
                       │
                       ▼
         ┌───────────────────────────┐
         │   Lazy Model Discovery    │
         │  --> Locks available upstream targets on first hit
         └─────────────┬-------------┘
                       │
                       ▼
         ┌───────────────────────────┐
         │ Timeline & ID Sanitation  │
         │  --> Fixes tool-call dot/colon ID formatting mismatches
         └─────────────┬-------------┘
                       │
                       ▼
         ┌───────────────────────────┐
         │ Tiered History Cache Check│
         │  --> Searches high/coarse granularity prefix hits
         └─────────────┬-------------┘
                       │
                       ▼
         ┌───────────────────────────┐
         │ Async Intent Execution    │
         │  --> Offloads routing classification to worker thread
         └─────────────┬-------------┘
                       │
                       ▼
         ┌───────────────────────────┐
         │ Schema Filter & Pruning   │
         │  --> Strips unused built-in & MCP JSON definitions
         └─────────────┬-------------┘
                       │
                       ▼
         ┌───────────────────────────┐
         │ Protocol Adaption Layer   │
         │  --> Applies Gemini history/system transformations
         └─────────────┬-------------┘
                       │
                       ▼
         Forwarded Upstream Gateway
         --> Employs optimized high-throughput httpx client pools
```

---

# 4. Subsystem Deep-Dives

## 4.1 Tiered Search History Cache Engine

As deep programming sessions progress, parsing and processing massive message arrays sequentially on every turn becomes an \( O(N) \) structural bottleneck.

The proxy implements a non-naive, sliding-granularity lookback strategy inside `_clean_history_incremental`:

### Tier 1 — High Granularity Lookback
Evaluates the immediate conversational neighborhood (the last 20 conversational frames with a step-index of 1) to account for rapid-fire local changes.

### Tier 2 — Coarse Granularity Lookback
Jumps progressively backwards (every 10th index from 30 up to 130 frames deep) to evaluate historical stability milestones.

### Mechanism
When a stable historical prefix is identified via SHA-256 content hashing, the proxy instantly retrieves the pre-cleaned cache record and appends only the un-cached "tail" segment.

This isolates mutating computations and protects system latency.

---

## 4.2 Automated Payload Sanitization & Conversation Repair

To bridge differences between Anthropic and Gemini schemas, the proxy acts as a runtime validation firewall.

### Tool Call Identifier Sanitization

Downstream non-Anthropic backends frequently return alphanumeric transaction identifiers containing periods or colons (e.g., `tool_id: 1.2`).

If these tracking markers are presented back to an Anthropic model on a subsequent turn, the client parser immediately flags an unrecoverable `HTTP 400 Bad Request`.

The proxy sweeps history frames and normalizes all ID syntax structures cleanly.

### Orphaned Block Remediation

Traverses incoming conversation states and purges dangling `tool_result` sequences that lack an associated parent `tool_use` counterpart block.

### Role Alternation Normalization

Smooths over contiguous, same-role message segments (e.g., consecutive user blocks) by combining and merging text components instead of dropping payloads, ensuring compliance with strict upstream APIs.

---

## 4.3 Intent Classification and Dynamic Schema Optimization

To reduce the context bloat caused by frameworks that bundle entire tool definition arrays with every frame, the engine performs context compression by combining intent analysis with structural pruning.

### Fingerprint Signature Lookup

Request attributes (User prompt fragments, Global Tool configuration tables, System Directives) are hashed using an MD5-based signature block, referencing an in-memory classification cache.

### Thread-Isolated Inference Classification

Upon a cache miss, evaluation tasks are dispatched asynchronously to a dedicated background thread executor using `run_in_executor`.

The routing model determines:

- Complexity indicators
- Tool usage dependencies
- History skipping flags

### Intent-Driven Pruning

If the classifier flags that the agent's turn requires no active tool access, the proxy strips all tool schemas entirely.

If tools are required, only the specific schemas highlighted by the classifier are retained.

This tracking component computes the token size delta between original and optimized schemas, splitting savings tracking across two heads:

- Built-in Tools (`Read`, `Write`, `Bash`, `Edit`, etc.)
- Heavy enterprise MCP Tools (`Jira`, `GitHub`, `Confluence`)

---

# 5. Cross-Ecosystem Protocol Adaptations

When the routing layer targets a non-Anthropic destination model family (e.g., Google Gemini via LiteLLM), specific format conversions are applied dynamically.

---

## 5.1 Extended Thinking Sequence Transformations

Models utilizing active token reasoning outputs generate structural internal thinking payloads.

The proxy captures these raw reasoning segments, translates them into plain-text blocks, and appends them as standard text elements to protect target execution parsers from schema collisions.

---

## 5.2 System Prompt Injection Bypassing

Google Vertex AI enforces a strict, undocumented threshold (~8,000 characters) on the `system_instruction` configuration field.

Agent frameworks routinely exceed this boundary during heavy engineering loops due to expansive environmental declarations, environment variables, and instruction suites.

### Mitigation Strategy (`_inject_system_into_first_message`)

When routing to a Gemini backend, the proxy intercepts the system payload, removes it from the configuration header, wraps it neatly within explicit structural `<SYSTEM_CONTEXT>` XML tags, and prepends it directly to the first user content array element.

### Architectural Win

This shifts the massive instruction layout out of the restricted configuration metadata pool and leverages Gemini's expansive 1M+ token context window, while model fine-tuning guarantees strong alignment with the XML structural boundaries.

---
# 6. Resiliency & Streaming Infrastructure Architecture

## 6.1 Multi-Level Fallback & Failover Chains

To prevent developer session disruptions from upstream provider rate limits (HTTP 429) or server faults (HTTP 5xx), the proxy builds and executes dynamic, ordered destination chains.

'''text
[Initial Target: gemini-flash] ──► (HTTP 429 / 500 Fault)
                                            │
                                            ▼ [Intercept & Rollback Payload]
                                 [Fallback Tier 1: claude-haiku] ──► (Success)

The sequence is constructed at runtime based on the turn's assessed complexity tier. For example, a low-complexity operational step maps out an emergency fallback pipeline traversing gemini-flash -> claude-haiku -> claude-sonnet. If the primary endpoint fails, the proxy interceptor catches the exception, recovers the original unmutated request payload from memory, and retries the loop down the sequence hierarchy silently.

## 6.2 Streaming (SSE) Continuity and Token Accounting

Because coding agents depend on rapid stream token returns for responsiveness, the proxy splits execution processing between standard HTTP responses and asynchronous Server-Sent Event (SSE) streams via an optimized httpx.AsyncClient pool.

### Connection Pool Tuning

Configured to maintain:

- 200 open parallel sockets
- Up to 50 active keep-alive connections

This eliminates connection assembly overhead and TCP/TLS handshaking latency on successive tool iteration turns.

### Stream Token Parsing

During an active streaming session, the proxy processes chunk payloads in real time.

It:

- Monitors streaming usage summaries
- Logs token accounting metrics
- Rewrites model target metadata blocks dynamically before forwarding responses back to the client

This guarantees precise financial metric captures without blocking execution paths.

---

# 7. Financial Accounting & Observability

## 7.1 Multi-Head Yield Calculations

Real-time financial performance auditing measures gross yields against fixed baseline reference metrics (`claude-3-5-sonnet`) across four distinct accounting heads.

\[
\text{Net Savings} =
\text{Model Routing Savings}
+
\text{Built-in Tool Savings}
+
\text{MCP Tool Savings}
-
\text{Classifier Overhead}
\]

Where:

- **Model Routing Savings**  
  Savings from model price differences for identical token spans.

- **Built-in & MCP Tool Savings**  
  Saved token costs from stripped tool schemas, valued at the baseline model's pricing.

- **Classifier Overhead**  
  Subtracted directly from gross yields, capturing the real-world operational cost of routing requests with a faster model like `gemini-2.5-flash`.

---

## 7.2 Observability Architecture

### Request-Scoped Reference Tags

All transactions receive a short-lived, random 4-hex identification marker (`[req-xxxx]`).

This marker prefixes interleaved logging traces across concurrent asynchronous execution steps, simplifying log correlation and debugging.

### Diagnostic Ledger Outputs

Every 10 minutes, short-form operational efficiencies are flushed to system output streams.

On every hour boundary, comprehensive cost ledgers are compiled and saved to disk:

```text
/proxylogs/savings-YYYY-MM-DD-HH.txt
```

for audit tracing.

---

# 8. Known System Constraints & Technical Boundaries

## Local Worker Memory Isolation

Caching records, statistical counters, and history indices are stored process-locally in memory.

Scaling the proxy horizontally requires externalizing state storage to a central cache layer (e.g., Redis).

---

## Contextual Shift Risks

Compressing long historical segments or formatting internal thinking chains into plain text can cause minor variations in agent behavior.

---

## Upstream Framework Dependencies

The structural formatting components remain coupled to:

- The Anthropic API message layout patterns
- LiteLLM downstream schema translation mechanics

Changes to these upstream specifications require updates to the normalization modules.

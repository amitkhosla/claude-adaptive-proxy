# Smart LLM Routing Proxy – Design Document

## Overview

This document describes the architecture, design decisions, operational behavior, and implementation details of the smart LLM routing proxy implemented in this repository.

The proxy sits in front of Anthropic-compatible `/v1/messages` APIs and dynamically routes requests between Claude and Gemini-family models based on:

* Request complexity
* Tool requirements
* Tool schema compatibility
* Prompt size
* Conversation state
* Cost optimization goals

The implementation focuses on reducing operational cost while preserving compatibility with Claude Code-style tool execution workflows and maintaining high reliability under large-context, multi-turn coding sessions.

---

# Goals

## Primary Goals

1. Reduce inference cost without materially degrading output quality.
2. Transparently route requests between multiple model providers.
3. Preserve compatibility with Anthropic-style tool calling.
4. Handle long-running coding sessions safely.
5. Avoid tool-chain corruption during mid-tool execution.
6. Support graceful fallback and retry behavior.
7. Produce measurable savings metrics.

## Non-Goals

* Perfect semantic equivalence across providers.
* Full provider abstraction layer.
* Persistent distributed caching.
* Advanced policy engines.
* Request deduplication across users.

---

# High-Level Architecture

## Request Flow

```text
Client Request
    |
    v
FastAPI Proxy
    |
    +--> Request Sanitization
    |
    +--> Conversation Repair
    |
    +--> Model Discovery
    |
    +--> Request Classification
    |
    +--> Routing Decision
    |
    +--> Tool Filtering
    |
    +--> Gemini Compatibility Sanitization
    |
    +--> Forward to Upstream Provider
    |
    +--> Retry / Fallback Chain
    |
    +--> Response Rewriting
    |
    +--> Usage + Savings Tracking
    |
    v
Client Response
```

---

# Core Design Principles

## 1. Stateless Routing

Each request is independently routable because the complete conversation history is always supplied by the client.

This enables:

* Mid-conversation provider switching
* Tool-loop routing changes
* Adaptive cost optimization
* Retry across providers

The proxy does not rely on hidden conversational state.

## 2. Safety Before Optimization

The router aggressively avoids risky provider transitions when:

* Tool schemas are complex
* Tool loops are active
* Message history becomes structurally invalid
* Provider compatibility is uncertain

This intentionally sacrifices some savings in favor of correctness.

## 3. Incremental Compatibility Repair

Instead of assuming providers behave identically, the proxy normalizes request history before forwarding.

This includes:

* Empty content cleanup
* Orphaned tool result removal
* Gemini-safe schema flattening
* Tool input serialization
* Role alternation fixes
* Extended-thinking normalization

---

# Configuration Model

Configuration is loaded from `config.yaml` and acts as the single source of truth.

Key configuration areas:

* Model routing chains
* Pricing tables
* Baseline model definition
* Gemini-safe tool allowlists
* Large prompt thresholds
* Classifier preferences

Example categories observed in implementation:

```yaml
chains:
  gemini:
  gemini_large:
  claude:
  claude_1m:

pricing:
  claude-sonnet:
  claude-haiku:
  gemini-2.5-flash:
```

The implementation dynamically loads configuration during startup. fileciteturn0file11L19-L41

---

# Model Routing Architecture

## Routing Inputs

The routing engine evaluates:

* User message
* Tool list
* System prompt
* Prompt size
* Conversation state
* Tool schema complexity
* Model availability

## Classification Pipeline

The classifier performs a lightweight LLM classification step to determine:

* Whether tools are required
* Request complexity
* Relevant tools
* Whether history may be skipped

Classifier output format:

```json
{
  "needs_tools": true,
  "complexity": "medium",
  "relevant_tools": ["Read", "Grep"],
  "can_skip_history": false
}
```

The classifier explicitly distinguishes between:

* Low complexity
* Medium complexity
* High complexity

and avoids overestimating complexity simply because multiple tools are involved. fileciteturn0file17L36-L88

## Classification Caching

A short-lived in-memory cache avoids repeated classifier calls for identical requests.

Observed implementation:

* TTL-based cache
* MD5-based cache key
* Includes:

  * User message
  * Tool names
  * System prompt prefix

fileciteturn0file11L42-L64

---

# Model Selection Strategy

## Claude Usage

Claude-family models are preferred when:

* Complex tools are required
* Tool schemas contain nested structures
* Editing workflows are active
* Bash or write operations are involved
* High reasoning depth is needed

## Gemini Usage

Gemini-family models are preferred when:

* Requests are conversational
* Tools are simple
* Schemas are flat
* Prompt cost dominates execution cost
* High throughput is desired

## Tool Safety Rules

The implementation maintains an explicit Gemini-safe tool allowlist.

Supported strategies:

* Exact tool name matching
* Prefix matching for MCP tools
* Schema complexity inspection for unknown tools

Complex schemas are rejected for Gemini routing. fileciteturn0file8L34-L79

## Large Prompt Routing

Prompt size estimation is performed before routing.

The estimator computes total payload size across:

* System prompt
* Messages
* Tool outputs

The implementation specifically targets large-context coding sessions where historical tool output may dominate payload size. fileciteturn0file8L11-L33

---

# Tool Filtering Optimization

## Problem

Claude Code tool schemas can become extremely large.

Examples:

* MCP tools
* Jira integrations
* GitHub integrations
* Nested object schemas

These schemas significantly increase prompt token usage.

## Solution

The proxy strips unused tools before forwarding.

Strategies:

1. Remove all tools if no tools are needed.
2. Retain only classifier-selected tools.
3. Compute token savings from removed schemas.

Observed logging behavior:

```text
Tools: 30 -> 4
(stripped 26: ~120k builtin + ~400k MCP tokens saved)
```

fileciteturn0file6L1-L23

## Built-in vs MCP Tool Handling

The implementation explicitly separates:

* Claude Code built-in tools
* MCP tools

Built-ins include:

* Read
* Write
* Edit
* Bash
* Grep
* LS
* WebSearch
* TodoRead
* NotebookEdit

fileciteturn0file7L31-L41

This distinction matters because MCP schemas are often substantially larger.

---

# Gemini Compatibility Layer

Gemini via LiteLLM has stricter structural validation than Claude.

The proxy implements a substantial compatibility layer.

## Key Repairs

### 1. Empty Text Block Removal

Gemini rejects empty text blocks.

The proxy removes:

```json
{"type": "text", "text": ""}
```

from both:

* Top-level content
* Nested tool_result content

fileciteturn0file15L65-L84

---

### 2. Extended Thinking Conversion

Gemini cannot safely replay Anthropic thinking blocks.

The proxy:

* Extracts thinking blocks
* Converts them into plain text
* Appends them as a normal text section

This preserves reasoning context while avoiding provider incompatibilities. fileciteturn0file15L1-L63

---

### 3. Tool Input Serialization

Built-in Claude tools are flattened for Gemini compatibility.

Nested object inputs are serialized into JSON strings to align with simplified schemas.

Important distinction:

* Built-in tools are flattened
* MCP tools preserve native object structures

This avoids schema/type mismatches. fileciteturn0file19L1-L44

---

### 4. Conversation Sanitization

Gemini-specific validation includes:

* Ensuring assistant messages contain text
* Removing orphaned tool results
* Fixing malformed history
* Maintaining role alternation

fileciteturn0file18L21-L74

---

### 5. History Flattening

Long conversations are compressed into textual summaries.

The implementation:

* Preserves recent messages intact
* Summarizes older history
* Avoids splitting active tool chains
* Maintains role alternation correctness

This significantly reduces payload size for long-running coding sessions. fileciteturn0file1L1-L62

---

# Conversation Repair Logic

## Orphaned Tool Results

Provider switching and message trimming can orphan tool results.

The implementation removes tool_result blocks whose corresponding tool_use no longer exists.

This prevents provider-side validation failures.

## Consecutive Role Merging

Gemini enforces stricter role alternation than Claude.

The proxy merges consecutive same-role messages instead of dropping them.

This preserves:

* Tool calls
* Tool responses
* Context continuity

while satisfying Gemini constraints.

fileciteturn0file15L54-L63

---

# Request Forwarding

## HTTP Layer

The implementation uses a shared async `httpx.AsyncClient`.

Configuration includes:

* Increased connection limits
* Large read timeouts
* Keepalive tuning
* Connection pooling

The implementation is explicitly optimized for:

* Large payloads
* Long-running streams
* Parallel tool-heavy workloads

fileciteturn0file9L70-L80

---

# Streaming Architecture

Streaming responses are handled separately from non-streaming responses.

The stream layer:

* Tracks SSE token usage
* Rewrites model metadata
* Handles fallback retries
* Preserves streaming continuity

The implementation supports chained retries during active streams. fileciteturn0file12L1-L74

---

# Fallback Strategy

## Multi-Level Retry Chain

The implementation supports progressive fallback chains.

Example:

```text
gemini-flash
    -> gemini-2.5-flash
        -> claude-haiku
            -> claude-sonnet
```

The fallback chain is dynamically constructed based on:

* Complexity
* Availability
* Current model
* Configured chains

fileciteturn0file6L44-L64

## Failure Handling

The proxy retries on:

* HTTP 400
* Streaming failures
* Provider incompatibilities
* Network exceptions

The original model request payload is preserved to guarantee rollback capability.

fileciteturn0file10L1-L76

---

# Observability and Diagnostics

## Request-Scoped Logging

Each request receives a unique request ID.

Example:

```text
[req-a1b2] Gemini route selected
```

This enables:

* Correlated debugging
* Streaming diagnostics
* Retry tracing
* Cost attribution

fileciteturn0file7L1-L15

---

## Debug Logging

The proxy writes structured JSONL debug records including:

* Original model
* Chosen model
* Routing reason
* Tool counts
* Error bodies
* Response previews

The implementation intentionally logs summaries rather than full message payloads to reduce log size and avoid leaking large prompts. fileciteturn0file0L1-L52

---

## Full Request Logging

Optional verbose logging captures:

* Original payload
* Sanitized payload
* Error responses
* Retry attempts

This is intended for deep production debugging.

---

# Cost Accounting System

## Savings Model

Savings are computed relative to a baseline model.

Baseline:

```text
claude-sonnet
```

Savings are broken into:

1. Model routing savings
2. Built-in tool schema savings
3. MCP schema savings
4. Classifier overhead

fileciteturn0file14L1-L52

## Cost Attribution

Per-request accounting includes:

* Input tokens
* Output tokens
* Tool schema token reductions
* Classifier cost overhead

## Hourly Savings Reports

The implementation periodically generates human-readable savings reports.

Reports include:

* Per-model breakdowns
* Net savings
* Percent savings
* Savings category attribution

fileciteturn0file5L1-L53

---

# Startup and Lifecycle

The FastAPI lifespan handler initializes:

* Prompt history
* Background reporters
* Model discovery
* Shared runtime state

Model discovery is lazy-loaded on first authenticated request. fileciteturn0file0L53-L80 fileciteturn0file16L1-L15

---

# Performance Characteristics

## Strengths

### Significant Token Reduction

Tool filtering and history flattening dramatically reduce payload size.

### Reduced Average Request Cost

Simple requests are routed to lower-cost models.

### High Resilience

Fallback chains reduce user-visible failures.

### Long Context Survivability

History compression mitigates runaway prompt growth.

---

# Known Trade-Offs

## Increased Complexity

The proxy introduces substantial compatibility logic.

This increases:

* Maintenance burden
* Edge-case surface area
* Provider-specific coupling

## Risk of Semantic Drift

History flattening and thinking conversion may alter subtle reasoning context.

## In-Memory State Limitations

Caches and statistics are process-local.

This implementation is not yet horizontally coordinated.

## Provider Coupling

Some logic is tightly coupled to:

* Anthropic message structure
* LiteLLM translation behavior
* Gemini validation constraints

---

# Recommended Future Improvements

## 1. Persistent Distributed Cache

Move classification and cleaned-history caches to Redis.

## 2. Structured Metrics Export

Expose Prometheus metrics:

* Routing counts
* Fallback counts
* Cost savings
* Provider latency
* Error rates

## 3. Adaptive Routing Feedback Loop

Use real request outcomes to improve routing quality.

Possible signals:

* Retry frequency
* User interruption rate
* Tool failure rate
* Output acceptance

## 4. Schema Fingerprinting

Precompute tool schema complexity hashes instead of recomputing per request.

## 5. Better History Compression

Current flattening is heuristic.

A semantic summarization layer would preserve more context fidelity.

## 6. Multi-Worker Shared State

Current caches and cost statistics are process-local.

Production deployments should externalize:

* History cache
* Classification cache
* Savings aggregation

---

# Conclusion

This implementation is not a simple reverse proxy.

It is effectively a compatibility-aware adaptive inference router optimized for coding-assistant workloads.

The design addresses several difficult operational realities:

* Extremely large prompts
* Tool-heavy conversations
* Provider incompatibilities
* Long-running sessions
* Cost pressure
* Streaming reliability

The strongest aspects of the implementation are:

* Defensive request sanitation
* Pragmatic provider compatibility handling
* Aggressive token optimization
* Multi-stage fallback behavior
* Fine-grained cost attribution

The largest architectural risk is the growing amount of provider-specific normalization logic, particularly around Gemini compatibility and Anthropic tool semantics.

As additional providers are added, this layer should likely evolve into a formal transformation pipeline with explicit provider adapters rather than incremental conditional logic.

# Claude Adaptive Proxy

An intelligent middleware layer between Claude Code / Claude CLI and LiteLLM that dynamically routes requests to the most cost-efficient model while reducing unnecessary context overhead, fixing provider incompatibilities, and stabilizing long-running coding sessions.

---

# Why This Exists

Modern coding agents default to expensive frontier models for *every* request.

But most developer workflows are not uniformly complex.

A simple grep operation does not need Sonnet.  
A file summary does not need Opus.  
A tool orchestration request does not need deep reasoning.

At the same time, coding agents silently waste massive amounts of context on unused tool schemas and bloated conversation history.

This proxy exists to solve both problems:

- use the *right* model for the task
- send only the *necessary* context

---

# The Hidden Problem: Tool Schemas Waste Massive Context

Claude Code sends the schema of every available tool on nearly every request.

That often includes:

- filesystem tools
- bash tools
- grep/glob tools
- Jira MCP tools
- GitHub MCP tools
- Confluence integrations
- internal Claude tools

Even when most of them are irrelevant.

In real-world workflows, tool definitions can consume more tokens than the actual user prompt.

This creates:

- unnecessary token spend
- larger prompts
- slower requests
- increased model confusion
- Gemini instability in long sessions
- degraded routing efficiency

Most users never notice this because it happens invisibly.

---

# How Claude Adaptive Proxy Fixes It

The proxy classifies request intent and sends only the tools actually needed for the task.

| Request | Tools Sent |
|---|---|
| Summarize this file | Read |
| Search the repository | Grep + Glob |
| Create Jira ticket | Jira MCP only |
| Explain architecture | No tools |

Everything else gets stripped.

This dramatically reduces prompt size while preserving workflow behavior.

---

# Core Features

## Intelligent Model Routing

Routes requests dynamically based on:

- task complexity
- reasoning depth
- tool requirements
- conversation state
- context structure

Example routing:

| Task Type | Routed Model |
|---|---|
| Simple code lookup | Gemini Flash |
| Medium reasoning | Claude Haiku OR Gemini Pro|
| Complex architecture work | Claude Sonnet |
| Too Complex architecture work | Claude Opus |

---

## Smart Tool Filtering

Instead of sending every tool on every request, the proxy:

- detects relevant tools
- strips unused schemas
- reduces context overhead
- improves inference efficiency

This is one of the largest sources of token savings.

---

## Context Compression

Long coding sessions generate massive histories.

The proxy intelligently:

- compresses old context
- preserves recent reasoning chains
- strips redundant thinking blocks
- caches cleaned histories
- removes orphaned tool results

This significantly improves Gemini stability in long-running sessions.

---

## Claude ↔ Gemini Compatibility Layer

Anthropic and Gemini APIs behave differently.

The proxy automatically handles:

- tool schema normalization
- tool call ID sanitization
- Gemini history constraints
- system prompt restructuring
- tool_result formatting fixes
- role alternation fixes
- thinking block cleanup
- malformed message recovery

Without this layer, many cross-provider workflows fail.

---

## Streaming Compatible

Works transparently with streaming responses.

No workflow changes required.

---

## Cost Analytics & Savings Reports

Built-in reporting includes:

- per-model usage
- baseline cost comparison
- routing savings
- tool schema savings
- classifier overhead
- hourly savings reports

You can quantify exactly how much the proxy saves.

---

# Key Benefits

## Lower Cost

Reduce unnecessary Sonnet usage automatically.

---

## Smaller Context Windows

Stop wasting tokens on irrelevant tools and bloated histories.

---

## Better Reliability

Fixes many provider incompatibilities automatically.

---

## Faster Requests

Smaller payloads and lighter models improve responsiveness.

---

## Longer Stable Sessions

Gemini and multi-provider coding sessions become significantly more reliable.

---

## Zero Workflow Changes

Works with existing Claude Code setups immediately.

---

# Ideal Use Cases

- Claude Code power users
- teams using LiteLLM
- multi-model coding workflows
- cost-sensitive AI infrastructure
- long-running coding sessions
- Gemini + Claude interoperability
- agentic development environments
- MCP-heavy workflows

---

# Philosophy

Most AI infrastructure today is static.

It assumes the largest model should handle every task with every tool attached at all times.

That is not intelligence.

Real intelligence is context selection.

Knowing:

- what matters
- what does not
- what should be preserved
- what should be removed
- and which model is actually sufficient

This proxy operationalizes that idea.

---

# Status

Experimental but production-oriented.

Designed for developers running heavy daily Claude Code workflows.

---

# Future Roadmap

- adaptive latency-aware routing
- semantic caching
- routing learning system
- automatic fallback retries
- provider health scoring
- observability dashboard
- prompt fingerprint analytics
- quality feedback loops
- distributed routing support

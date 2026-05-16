# Claude Adaptive Proxy (Smart LLM Routing Proxy)

An intelligent, high-performance middleware layer situated between Anthropic-compatible clients (such as the Claude Code CLI) and heterogeneous upstream provider gateways via LiteLLM.

By executing:

- Real-time request intent classification
- Dynamic schema pruning
- Automated wire-format normalization

The proxy eliminates runaway token spend and guarantees session continuity inside complex, multi-turn software engineering agent loops.

---

# 🚀 Core Capabilities

| Capability | Purpose |
|---|---|
| Intent Classification | Detects task complexity dynamically |
| Schema Pruning | Removes unused tool definitions before inference |
| Multi-Tier Routing | Routes traffic across Flash, Pro, and Sonnet tiers |
| Protocol Repair | Fixes cross-provider incompatibilities |
| SSE Streaming Support | Preserves low-latency streaming continuity |
| Prefix Cache | Optimizes long conversational histories |
| Failover Recovery | Retries safely across provider hierarchies |
```

## Up to 80% Token Reduction

Automatically identifies and strips unused tool schemas (e.g., heavy Jira, GitHub, or Confluence MCP integrations) before they are sent to the model.

---

## Predictive Economic Routing

Routes simple operations (like `grep` or file summaries) to fast, cost-effective models like `gemini-2.5-flash`, reserving frontier models like `claude-3-5-sonnet` strictly for complex reasoning tasks.

---

## Cross-Provider Stability

Acts as a validation firewall by fixing:

- Tool call ID mismatches
- Role-alternation bugs
- Vertex AI system prompt limitations

that typically crash multi-provider agent sessions.

---

## Zero Workflow Changes

Drops directly into your existing environment as a transparent pass-through wrapper.

---

# 🏗️ System Architecture Overview

The proxy acts as a stateless, functional pass-through wrapper executing the following optimization pipeline:

```text
               [Claude Code CLI / Client]
                           │
         (Anthropic Wire Format: /v1/messages)
                           ▼
            ┌───────────────────────────────┐
            │   Smart LLM Routing Proxy     │
            └───────────────┬───────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        ▼                                       ▼

   (Low Complexity)                 (Medium / High Complexity)

┌───────────────────────────┐     ┌───────────────────────────┐
│     Gemini 2.5 Flash      │     │      Gemini 2.5 Pro       │
│  Fast / Cost Optimized    │     │   Advanced Reasoning      │
└─────────────┬─────────────┘     └─────────────┬─────────────┘
              │                                 │
              └──────────────┬──────────────────┘
                             ▼

                 ┌───────────────────────────┐
                 │     Claude 3.5 Sonnet     │
                 │  Fallback / Recovery Tier │
                 └───────────────────────────┘
```

---

## 📘 Detailed Technical Mechanics

For internal implementation specifics—including:

- Tiered Sliding-Granularity Prefix Cache
- Thread-Isolated Inference Classification
- Vertex AI System Prompt Injection Bypassing

[Detailed design document](Design.md)


---

# ⚙️ Configuration & Deployment Matrix

The proxy is entirely stateless and process-local.

Configure it via standard environment variables:

| Environment Variable     | Required | Description                                                  | Default / Example |
|--------------------------|----------|--------------------------------------------------------------|------------------|
| `PROXY_HOST`             | No       | Network bind interface for the routing proxy layer           | `127.0.0.1` |
| `PROXY_PORT`             | No       | Listening port for client application redirection            | `8080` |
| `LITELLM_MASTER_KEY`     | Yes      | Authentication token matching your upstream gateway          | `sk-litellm-...` |
| `HTTPX_MAX_CONNECTIONS`  | No       | Maximum parallel sockets allocated in the client pool        | `200` |
| `HTTPX_MAX_KEEPALIVE`    | No       | Retained idle connections for rapid mid-loop turns           | `50` |

---

# 🏃 Quick Start

## 1. Start the Proxy

Ensure your environment variables are set, then spin up the proxy instance:

```bash
# Example using Python
# (adjust based on your runtime engine, e.g., Go/Rust/Node)

python main.py
```

---

## 2. Configure Claude Code to Target the Proxy

Redirect your client's API endpoint base to point to the local proxy instance:

```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:8080/v1"
```

Run Claude as you normally would.

The proxy handles the rest invisibly.

---

# 📊 Financial Auditing & Observability

The system computes precise operational efficiency margins by dividing token reduction yields across four discrete accounting heads evaluated against fixed baseline metrics (`claude-3-5-sonnet` standard costs).

\[
\text{Net Savings}
=
\text{Model Routing Savings}
+
\text{Built-in Tool Savings}
+
\text{MCP Tool Savings}
-
\text{Classifier Overhead}
\]

---

## Observability Outputs

- Short-form efficiency metrics are flushed to stdout every 10 minutes.
- Audit-ready financial ledgers are written every hour directly to disk.

```text
/proxylogs/savings-YYYY-MM-DD-HH.txt
```

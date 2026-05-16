"""
Classifier — routes every request to the cheapest appropriate model.

Decision flow (in order, stops at first match):
  1. SIZE CHECK       free  — prompt > 100KB -> 1M context model
  2. MID-TOOL GUARD   free  — current turn in tool loop -> Gemini (simple tools) or Claude (complex)
  3. CACHE CHECK      free  — same request seen in last 60s -> reuse result
  4. CLASSIFY         1 LLM call -> needs_tools, complexity, relevant_tools

Key insight: LLM calls are stateless. Each request gets full context. Model can safely
switch every turn — including mid-tool-loop — because Claude Code sends the complete
messages array every time. LiteLLM handles format translation per call.

Gemini is used when:
  - No tools needed (pure text/discussion)
  - Only simple tools needed (flat schemas: Read, Grep, Glob, simple MCPs)
Claude is used when:
  - Complex tool schemas (Edit, Write, Bash, Task — risky for LiteLLM translation)
"""
import hashlib
import json
import time
from pathlib import Path

import httpx
import yaml

# ── Load all settings from config.yaml (single source of truth) ───────────────

def _load_config() -> dict:
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)

_config = _load_config()

CLAUDE_1M_CHAINS      = _config["chains"]["claude_1m"]
GEMINI_LARGE_CHAINS   = _config["chains"]["gemini_large"]
GEMINI_CHAINS         = _config["chains"]["gemini"]
CLAUDE_CHAINS         = _config["chains"]["claude"]
CLASSIFIER_PREFERENCE = _config["classifier_preference"]
LARGE_PROMPT_BYTES    = _config.get("large_prompt_threshold_bytes", 100_000)

# Gemini safe tools — exact names AND prefix patterns (e.g. "jira" matches "mcp__jira__*")
_raw_safe = _config.get("gemini_safe_tools", [])
GEMINI_SAFE_TOOLS_EXACT    = {t for t in _raw_safe if not t.islower() or "_" in t}  # e.g. "Read", "Glob"
GEMINI_SAFE_TOOLS_PREFIXES = tuple(f"mcp__{t}__" for t in _raw_safe if t.islower() and "_" not in t)  # e.g. "jira" -> "mcp__jira__"


# ── Classification cache ───────────────────────────────────────────────────────

_cache: dict = {}
_CACHE_TTL   = 60  # seconds


def _cache_key(user_message: str, tool_names: list[str], system_prompt: str) -> str:
    raw = f"{user_message}|{','.join(sorted(tool_names))}|{system_prompt[:200]}"
    return hashlib.md5(raw.encode()).hexdigest()


def _get_cached(key: str) -> dict | None:
    if key in _cache:
        result, ts = _cache[key]
        if time.time() - ts < _CACHE_TTL:
            return result
        del _cache[key]
    return None


def _set_cache(key: str, result: dict) -> None:
    _cache[key] = (result, time.time())


# ── LiteLLM model discovery ────────────────────────────────────────────────────

def fetch_available_models(base_url: str, api_key: str) -> set:
    try:
        resp = httpx.get(
            f"{base_url.rstrip('/')}/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            verify=False,
            timeout=15,
        )
        return {m["id"] for m in resp.json().get("data", [])}
    except Exception:
        return set()


def best_from(chain: list, available: set, fallback: str) -> str:
    for m in chain:
        if m in available:
            return m
    return fallback


# ── Prompt size estimator ──────────────────────────────────────────────────────

def estimate_prompt_bytes(body: dict) -> int:
    """
    Total size of all messages + system prompt in bytes.
    This grows as tool results (file contents, command output) accumulate over turns.
    A prompt that looks simple ("continue") can be 200KB if prior turns read large files.
    """
    total = 0
    system = body.get("system", "")
    if isinstance(system, str):
        total += len(system.encode())
    elif isinstance(system, list):
        total += len(json.dumps(system).encode())
    for msg in body.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(content.encode())
        elif isinstance(content, list):
            total += len(json.dumps(content).encode())
    return total


# ── Gemini tool safety check ───────────────────────────────────────────────────

def _is_schema_simple(schema: dict) -> bool:
    """
    True if the tool's input schema is simple enough for reliable LiteLLM translation.
    Simple = few properties, all primitive types, no nested objects/arrays.
    """
    props = schema.get("properties", {})
    if len(props) > 5:
        return False
    for prop_schema in props.values():
        if isinstance(prop_schema, dict) and prop_schema.get("type") in ("object", "array"):
            return False
    return True


def _is_tool_gemini_safe(name: str) -> bool:
    """Check if a tool name matches the exact safe list or any safe prefix."""
    if name in GEMINI_SAFE_TOOLS_EXACT:
        return True
    if GEMINI_SAFE_TOOLS_PREFIXES and name.startswith(GEMINI_SAFE_TOOLS_PREFIXES):
        return True
    return False


def tools_are_gemini_safe(tool_names: list[str], all_tools: list[dict]) -> bool:
    """
    True if every named tool is safe to route to Gemini via LiteLLM translation.
    Known simple Claude Code tools are always safe.
    MCP tools matching a safe prefix (e.g. "jira" -> "mcp__jira__*") are safe.
    Unknown tools are checked by schema complexity.
    """
    if not tool_names:
        return True
    tool_map = {t.get("name"): t for t in all_tools if isinstance(t, dict)}
    for name in tool_names:
        if _is_tool_gemini_safe(name):
            continue
        # Unknown tool — inspect schema
        tool   = tool_map.get(name, {})
        schema = tool.get("input_schema", tool.get("parameters", {}))
        if not _is_schema_simple(schema):
            return False
    return True


# ── Mid-turn tool execution detection ─────────────────────────────────────────

def is_mid_tool_execution(messages: list) -> bool:
    """
    True only when the CURRENT turn is mid-tool-execution.
    Detected by: most recent user message contains tool_result blocks.
    This means the model just requested a tool, Claude Code ran it, and is feeding
    the result back. Does NOT lock based on any historical tool usage.
    """
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        return True
            return False  # last user message has no tool_result
    return False


def get_active_tool_names(messages: list) -> list[str]:
    """Names of tools called in the most recent assistant message."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", [])
            if isinstance(content, list):
                return [
                    b.get("name", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name")
                ]
    return []


def count_active_tool_calls(messages: list) -> int:
    return len(get_active_tool_names(messages))


# ── Classifier LLM call ────────────────────────────────────────────────────────

def _clean_for_gemini(messages: list) -> list:
    """Gemini is strict: no empty text blocks, no trailing assistant roles."""
    cleaned = []
    for msg in messages:
        content = msg.get("content", "")
        if not content: continue
        if isinstance(content, list):
            # Remove empty text blocks or null types
            new_content = [b for b in content if isinstance(b, dict) and (b.get("text") or b.get("type") != "text")]
            if not new_content: continue
            msg["content"] = new_content
        cleaned.append(msg)
    return cleaned


def classify_via_llm(
    user_message: str,
    tool_names: list[str],
    classifier_model: str,
    base_url: str,
    api_key: str,
    system_prompt: str = "",
) -> dict:
    """One cheap LLM call to classify the request."""
    tools_list = (
        "\n".join(f"  - {t}" for t in tool_names)
        if tool_names else "  (none)"
    )
    system_context = (
        f"System context (first 500 chars): {system_prompt[:500]}\n\n"
        if system_prompt else ""
    )

    system = """You are a request classifier for an AI coding assistant proxy.
Given a user message, the available tools, and optional system context, output:

1. needs_tools  — true if the response requires calling any tool (file read/write,
                  shell commands, web search, API calls, etc.)
                  false if it is a pure question, explanation, or discussion

2. complexity   — Choose carefully based on WHAT the task requires, not how many tools:
                  low    — Information retrieval, reading files, running simple shell
                           commands (ls, grep, cat), status checks, Q&A, explanations,
                           internal tool updates (Task), web lookups.
                  medium — Writing or editing a small amount of code in a single file,
                           fixing a simple bug with a clear solution, moderate research.
                  high   — Complex multi-file changes, deep debugging of hard-to-trace
                           issues, architectural decisions, research requiring synthesis
                           of many sources, writing large new features from scratch.
                  IMPORTANT: Do NOT increase complexity just because multiple tools are
                  listed. A request that reads 3 files is still LOW complexity.
                  Only use HIGH for tasks that genuinely require deep reasoning.
                  Only use MEDIUM for tasks that require actual code writing/editing.

3. relevant_tools — which specific tools from the list are DEFINITELY needed.
                    Be strict — only include tools the task cannot be done without.
                    Do NOT include tools speculatively or "just in case".
                    Empty list [] if needs_tools is false.

4. can_skip_history — true if the current request is self-contained or the immediate
                      task is clearly defined in the most recent turns.
                      Set to true if the model can fulfill the request using only
                      the system prompt, current task context, and the last 2-3 turns
                      without needing the full conversation history.
                      Set to false if the request refers to specific details,
                      code, or context discussed much earlier in the session.

Reply with JSON only, no explanation:
{"needs_tools": true/false, "complexity": "low/medium/high", "relevant_tools": ["Tool1"], "can_skip_history": true/false}"""

    prompt = f"""{system_context}Available tools:
{tools_list}

User message: {user_message}"""

    try:
        resp = httpx.post(
            f"{base_url.rstrip('/')}/v1/messages",
            headers={
                "Authorization":     f"Bearer {api_key}",
                "Content-Type":      "application/json",
                "anthropic-version": "2023-06-01",
            },
            json={
                "model":      classifier_model,
                "max_tokens": 120,
                "messages":   [{"role": "user", "content": prompt}],
                "system":     system,
            },
            verify=False,
            timeout=30,
        )
        # Surface 400 errors so caller can retry with next classifier model
        if resp.status_code == 400:
            return {
                "needs_tools":      True,
                "complexity":       "medium",
                "relevant_tools":   tool_names,
                "_clf_error_retry": True,
                "_clf_model":       classifier_model,
                "_clf_input":       0,
                "_clf_output":      0,
            }
        raw   = resp.json()
        usage = raw.get("usage", {})
        text  = raw.get("content", [{}])[0].get("text", "").strip()
        text  = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        result = json.loads(text)
        valid  = set(tool_names)
        return {
            "needs_tools":    bool(result.get("needs_tools", True)),
            "complexity":     result.get("complexity", "medium"),
            "relevant_tools": [t for t in result.get("relevant_tools", tool_names) if t in valid],
            "can_skip_history": bool(result.get("can_skip_history", False)),
            # Classifier call cost — tracked as overhead (added expense, not a saving)
            "_clf_model":  classifier_model,
            "_clf_input":  usage.get("input_tokens",  0),
            "_clf_output": usage.get("output_tokens", 0),
        }
    except Exception as e:
        # Network/parse error — safe fallback, stay on Claude with all tools
        return {
            "needs_tools":    True,
            "complexity":     "medium",
            "relevant_tools": tool_names,
            "error":          str(e),
            "_clf_model":  classifier_model,
            "_clf_input":  0,
            "_clf_output": 0,
        }


# ── Main routing decision ──────────────────────────────────────────────────────

def decide_model(
    body: dict,
    available: set,
    base_url: str,
    api_key: str,
) -> tuple[str, str, dict]:
    """
    Returns (chosen_model, reason, classification).
    classification always contains: needs_tools, complexity, relevant_tools, source.
    """
    messages      = body.get("messages", [])
    original      = body.get("model", "claude-sonnet")
    tools         = body.get("tools", [])
    tool_names    = [t.get("name", "") for t in tools if isinstance(t, dict)]
    system_prompt = body.get("system", "") if isinstance(body.get("system"), str) else ""

    # ── 1. Large prompt -> 1M context model ───────────────────────────────────
    # Split by tool complexity — complex tools need Claude 1M, simple/none use Gemini large.
    # tools_are_gemini_safe() does this without any extra LLM call.
    prompt_bytes = estimate_prompt_bytes(body)
    if prompt_bytes > LARGE_PROMPT_BYTES:
        if tools and not tools_are_gemini_safe(tool_names, tools):
            model  = best_from(CLAUDE_1M_CHAINS, available, original)
            reason = f"large-prompt:{prompt_bytes // 1000}KB->claude-1m"
        else:
            model  = best_from(GEMINI_LARGE_CHAINS, available, original)
            reason = f"large-prompt:{prompt_bytes // 1000}KB->gemini-large"
        classification = {
            "needs_tools":    bool(tools),
            "complexity":     "high",
            "relevant_tools": tool_names,
            "prompt_bytes":   prompt_bytes,
            "source":         "size-check",
        }
        return model, reason, classification

    # ── 2. Current turn mid-tool-execution ───────────────────────────────────
    # Each LLM call is stateless and gets full context, so we can freely choose
    # the best model for THIS turn regardless of which model handled prior turns.
    if is_mid_tool_execution(messages):
        active_names    = get_active_tool_names(messages)
        active_tool_objs = [t for t in tools if t.get("name") in set(active_names)]
        tool_count      = len(active_names)
        complexity      = "high" if tool_count > 10 else "medium" if tool_count > 4 else "low"

        if tools_are_gemini_safe(active_names, active_tool_objs):
            model = best_from(GEMINI_CHAINS[complexity], available, original)
            classification = {
                "needs_tools":    True,
                "complexity":     complexity,
                "relevant_tools": active_names,
                "source":         "mid-tool:gemini-safe",
            }
            return model, f"mid-tool:gemini-{complexity}", classification
        else:
            model = best_from(CLAUDE_CHAINS[complexity], available, original)
            classification = {
                "needs_tools":    True,
                "complexity":     complexity,
                "relevant_tools": active_names,
                "source":         "mid-tool:complex-tools",
            }
            return model, f"mid-tool:claude-{complexity}", classification

    # ── 3. No user message ────────────────────────────────────────────────────
    last_user = _last_user_text(messages)
    if not last_user:
        return original, "passthrough:no-user-message", {
            "needs_tools": True, "complexity": "medium", "relevant_tools": tool_names,
        }

    # ── 4. Cache check ────────────────────────────────────────────────────────
    cache_key = _cache_key(last_user, tool_names, system_prompt)
    cached    = _get_cached(cache_key)
    if cached:
        classification = {**cached, "source": "cache"}
        model, reason  = _route_from_classification(classification, tools, tool_names, available, original)
        return model, reason + "[cached]", classification

    # ── 5. Pick cheapest available classifier model ───────────────────────────
    # Try each model in preference order; retry with next if one returns a 400 error.
    available_classifiers = [m for m in CLASSIFIER_PREFERENCE if m in available] or ["gemini-flash"]

    # ── 6. Classify (1 LLM call, with fallback) ──────────────────────────────
    classification = None
    for classifier_model in available_classifiers:
        classification = classify_via_llm(
            user_message=last_user,
            tool_names=tool_names,
            classifier_model=classifier_model,
            base_url=base_url,
            api_key=api_key,
            system_prompt=system_prompt,
        )
        # If the call hit a 400/format error, try the next classifier model
        if classification.get("_clf_error_retry"):
            continue
        break
    classification["source"] = f"classifier:{classifier_model}"
    _set_cache(cache_key, classification)

    model, reason = _route_from_classification(classification, tools, tool_names, available, original)
    return model, reason, classification


def _route_from_classification(
    classification: dict,
    tools: list,
    tool_names: list[str],
    available: set,
    original: str,
) -> tuple[str, str]:
    """Pick model + reason from a classification result."""
    needs_tools = classification.get("needs_tools", True)
    complexity  = classification.get("complexity", "medium")
    relevant    = classification.get("relevant_tools", tool_names)

    if not needs_tools:
        # Pure text — cheapest Gemini, no tools needed
        chain = GEMINI_CHAINS.get(complexity, GEMINI_CHAINS["medium"])
        model = best_from(chain, available, original)
        classification["_chain"] = chain
        return model, f"no-tool:gemini-{complexity}"

    # Tools needed — check if all relevant tools are Gemini-safe
    relevant_objs = [t for t in tools if t.get("name") in set(relevant)]
    if tools_are_gemini_safe(relevant, relevant_objs):
        chain = GEMINI_CHAINS.get(complexity, GEMINI_CHAINS["medium"])
        model = best_from(chain, available, original)
        classification["_chain"] = chain
        return model, f"simple-tools:gemini-{complexity}"
    else:
        # Find which specific tools are NOT Gemini-safe so we can log them
        unsafe = [t for t in relevant if not _is_tool_gemini_safe(t)
                  and not _is_schema_simple(
                      next((x.get("input_schema", x.get("parameters", {}))
                            for x in tools if x.get("name") == t), {})
                  )]
        chain = CLAUDE_CHAINS.get(complexity, CLAUDE_CHAINS["medium"])
        model = best_from(chain, available, original)
        classification["_chain"] = chain
        return model, f"complex-tools:claude-{complexity}[unsafe:{','.join(unsafe) if unsafe else 'unknown'}]"


# ── Utility ───────────────────────────────────────────────────────────────────

def _last_user_text(messages: list) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
    return ""

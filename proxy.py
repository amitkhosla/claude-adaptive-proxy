"""
Smart Model Router Proxy
Sits between Claude CLI and LiteLLM. Intercepts /v1/messages,
picks cheapest appropriate model, forwards transparently.

Claude CLI --> localhost:8000 (this proxy) --> (LiteLLM)
"""
import asyncio

class Tictoc:
    def __init__(self):
        self.times = {}
        self.start_t = time.time()
    def mark(self, name):
        now = time.time()
        self.times[name] = now - self.start_t
        self.start_t = now
    def summary(self):
        return " | ".join([f"{k}:{v:.2f}s" for k, v in self.times.items()])
import json
import time
import logging
from logging.handlers import RotatingFileHandler
import re
import secrets
import yaml
import os
import glob
import uuid
import copy
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import httpx
from httpx import Limits, Timeout
import hashlib

# ── Performance Cache ─────────────────────────────────────────────────────────
# Caches "Cleaned" history prefixes to avoid re-processing 300+ messages on every turn.
# Key: sha256(json_of_prefix_messages)
# Value: List of cleaned message dicts
_CLEAN_CACHE = {}
_CACHE_MAX_SIZE = 500

def _get_history_hash(messages: list) -> str:
    """Generate a stable hash for a list of messages."""
    if not messages:
        return "empty"
    # We only hash the content-relevant parts to be stable
    try:
        compact = json.dumps(messages, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(compact.encode()).hexdigest()
    except Exception:
        return "error"

def _clean_history_incremental(messages: list, rlog: "ReqLog") -> list:
    """
    Cleans history by checking if a prefix of the messages is already cached.
    Uses a tiered search strategy to find the longest cached prefix.

    Tier 1: Check last 20 prefixes (high granularity)
    Tier 2: Check every 10th prefix for 10 more attempts (coarse granularity)
    """
    if not messages:
        return []

    n = len(messages)
    # Define tiered search indices (last 20, then every 10th for 10 more)
    tier1 = list(range(n, max(0, n - 20), -1))
    tier2 = list(range(n - 30, max(0, n - 130), -10))
    search_indices = tier1 + tier2

    for i in search_indices:
        if i <= 0 or i > n:
            continue

        prefix = messages[:i]
        h = _get_history_hash(prefix)
        if h in _CLEAN_CACHE:
            # Cache hit — retrieve the cleaned prefix
            cached_cleaned = _CLEAN_CACHE[h]

            # Process the "tail" (new messages since the cache point)
            # Create a shallow copy of the tail to avoid mutating the source 'messages'
            remaining = [copy.deepcopy(m) for m in messages[i:]]

            if not remaining:
                rlog.info(f"Tiered Cache Hit: index {i}, all messages already cleaned")
                return cached_cleaned

            # Clean the tail (mutates 'remaining' in-place)
            n_stripped = _strip_thinking_blocks(remaining)

            # Combine cached prefix with newly cleaned tail
            full_cleaned = cached_cleaned + remaining

            # Store the full result for next turn
            full_h = _get_history_hash(messages)
            if len(_CLEAN_CACHE) < _CACHE_MAX_SIZE:
                _CLEAN_CACHE[full_h] = full_cleaned

            rlog.info(f"Tiered Cache Hit: index {i}, skipped {i} msgs, cleaned {len(remaining)} new "
                      f"({n_stripped} thinking stripped, cache size={len(_CLEAN_CACHE)})")
            return full_cleaned

    # No cache hit — process everything
    # Create a deep copy for processing to keep the original source intact for hashing
    to_clean = [copy.deepcopy(m) for m in messages]
    n_stripped = _strip_thinking_blocks(to_clean)

    full_h = _get_history_hash(messages)
    if len(_CLEAN_CACHE) < _CACHE_MAX_SIZE:
        _CLEAN_CACHE[full_h] = to_clean

    rlog.info(f"Tiered Cache Miss: cleaned all {n} messages "
              f"({n_stripped} thinking stripped, cache size={len(_CLEAN_CACHE)})")
    return to_clean

# ── Global HTTP Client ────────────────────────────────────────────────────────
# Enhanced with higher limits and specific timeouts to prevent deadlocks with large context
http = httpx.AsyncClient(
    verify=False,
    timeout=Timeout(300.0, connect=60.0, read=300.0, pool=30.0),
    limits=Limits(
        max_connections=200,          # Increase total parallel connections
        max_keepalive_connections=50,  # Keep more warm connections
        keepalive_expiry=30.0
    )
)
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse

from classifier import decide_model, fetch_available_models, _last_user_text

# ── Config ─────────────────────────────────────────────────────────────────────

def load_config():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)

config     = load_config()
UPSTREAM   = config["api"]["base_url"].rstrip("/")
PROXY_PORT = config.get("proxy_port", 8000)

# ── Logging — always writes to C:\proxylogs\smart-claude.log ───────────────────

LOG_DIR  = Path(__file__).parent / "proxylogs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "smart-claude.log"

_fmt       = logging.Formatter("%(asctime)s [Proxy] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
_file_h    = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
_file_h.setFormatter(_fmt)
_console_h = logging.StreamHandler()
_console_h.setFormatter(_fmt)

logging.basicConfig(level=logging.INFO, handlers=[_file_h, _console_h])
log = logging.getLogger("proxy")

# Suppress noisy httpx request-level logs (we log our own status lines)
logging.getLogger("httpx").setLevel(logging.WARNING)


# ── Request ID ─────────────────────────────────────────────────────────────────
# Short 4-hex-char tag added to every log line for a given request so that
# interleaved log lines from concurrent requests can be correlated.

def _new_req_id() -> str:
    """Return a short random ID like 'req-a1b2'."""
    return "req-" + secrets.token_hex(2)


class ReqLog:
    """Thin wrapper around the module logger that prefixes every message with [req-xxxx]."""

    def __init__(self, req_id: str):
        self.req_id  = req_id
        self._prefix = f"[{req_id}] "

    def info(self, msg: str, *args, **kwargs):
        log.info(self._prefix + msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        log.warning(self._prefix + msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        log.error(self._prefix + msg, *args, **kwargs)

# ── Pricing table — loaded from config ─────────────────────────────────────────
# Baseline: the model the user was calling before smart routing (claude-sonnet).
# All savings are calculated relative to what each request would have cost on
# the baseline model with the same token counts.

BASELINE_MODEL = config.get("baseline_model", "claude-sonnet")

# (input $/1M tokens, output $/1M tokens)
_raw_pricing = config.get("pricing", {})
MODEL_PRICING: dict[str, tuple[float, float]] = {
    m: (float(p[0]), float(p[1])) for m, p in _raw_pricing.items()
}

# Fallback price for models not in our pricing table
_DEFAULT_PRICE  = (3.00, 15.00)  # Claude Sonnet pricing as safe default
_BASELINE_PRICE = MODEL_PRICING.get(BASELINE_MODEL, _DEFAULT_PRICE)

# All Claude Code built-in tools. Anything outside this set is an MCP tool.
BUILTIN_TOOLS: set[str] = {
    "Read", "Write", "Edit", "MultiEdit", "Bash",
    "Glob", "Grep", "LS",
    "WebFetch", "WebSearch",
    "Task",
    "TodoRead", "TodoWrite",
    "NotebookRead", "NotebookEdit",
    "ExitPlanMode", "AskUserQuestion",
}


def _cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Cost in USD for a single call given token counts."""
    price = MODEL_PRICING.get(model, _BASELINE_PRICE)
    return (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000


def _baseline_cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * _BASELINE_PRICE[0] + output_tokens * _BASELINE_PRICE[1]) / 1_000_000


# ── Hourly cost stats ──────────────────────────────────────────────────────────

# _cost_stats: {"YYYY-MM-DD HH": [call_record, ...]}
# call_record: {ts, chosen, reason, input_tokens, output_tokens}
_cost_stats: dict = {}
_cost_lock = asyncio.Lock()


async def _record_cost(chosen: str, reason: str, input_tokens: int, output_tokens: int,
                       extra_builtin: int = 0, extra_mcp: int = 0,
                       clf_model: str = "", clf_cost: float = 0.0) -> None:
    """Append token usage for one completed call.

    extra_builtin : tokens from stripped Claude Code built-in tool schemas
                    (Read, Write, Edit, Bash, ...) not sent to the model.
    extra_mcp     : tokens from stripped MCP tool schemas (Jira, GitHub, ...)
                    not sent to the model.
    Both are input tokens the original claude-sonnet call would have consumed.

    clf_model / clf_cost : the classifier LLM call (gemini-flash) made to
                    route this request.  This is an ADDED expense that did not
                    exist before the proxy and must be subtracted from savings.
    """
    if input_tokens == 0 and output_tokens == 0:
        return
    hour_key = datetime.now().strftime("%Y-%m-%d %H")
    async with _cost_lock:
        if hour_key not in _cost_stats:
            _cost_stats[hour_key] = []
        _cost_stats[hour_key].append({
            "ts":            datetime.now().strftime("%H:%M:%S"),
            "chosen":        chosen,
            "reason":        reason,
            "input":         input_tokens,
            "output":        output_tokens,
            "extra_builtin": extra_builtin,   # built-in tool schema tokens saved
            "extra_mcp":     extra_mcp,       # MCP tool schema tokens saved
            "clf_model":     clf_model,       # classifier model used (overhead)
            "clf_cost":      clf_cost,        # $ cost of classifier call (overhead)
        })


def _savings_breakdown(per_model: dict, calls: list | None = None) -> tuple[float, float, float, float, float, float]:
    """
    Returns (actual_cost, baseline_cost,
             model_routing_savings, builtin_tool_savings, mcp_tool_savings,
             classifier_overhead).

    Three saving heads:
      model_routing  -- cheaper model price vs claude-sonnet for the same tokens
      builtin_tool   -- Claude Code built-in tool schemas not sent (Read/Write/Bash/...)
      mcp_tool       -- MCP tool schemas not sent (Jira/GitHub/...)

    Classifier overhead is an ADDED expense (extra cost not present before proxy).
    Net savings = model_routing + builtin_tool + mcp_tool - classifier_overhead.
    """
    actual_cost           = 0.0
    model_routing_savings = 0.0
    builtin_tool_savings  = 0.0
    mcp_tool_savings      = 0.0
    classifier_overhead   = 0.0

    for m, v in per_model.items():
        in_t          = v["input"]
        out_t         = v["output"]
        extra_builtin = v["extra_builtin"]
        extra_mcp     = v["extra_mcp"]
        actual        = _cost_usd(m, in_t, out_t)

        model_routing_savings += _baseline_cost_usd(in_t, out_t) - actual
        builtin_tool_savings  += _baseline_cost_usd(extra_builtin, 0)
        mcp_tool_savings      += _baseline_cost_usd(extra_mcp, 0)
        actual_cost           += actual

    # Classifier overhead: sum clf_cost from raw call records
    if calls:
        for c in calls:
            classifier_overhead += c.get("clf_cost", 0.0)

    # actual_cost includes classifier overhead (it's a real expense)
    actual_cost   += classifier_overhead
    baseline_cost  = actual_cost - classifier_overhead + model_routing_savings + builtin_tool_savings + mcp_tool_savings
    return actual_cost, baseline_cost, model_routing_savings, builtin_tool_savings, mcp_tool_savings, classifier_overhead


def _write_savings_report(hour_key: str, calls: list) -> None:
    """Write a human-readable savings report for one completed hour."""
    date_str, hour_str = hour_key.split(" ")
    report_path = LOG_DIR / f"savings-{date_str}-{hour_str}.txt"
    if report_path.exists():
        return

    # Aggregate per model
    per_model: dict[str, dict] = {}
    for c in calls:
        m = c["chosen"]
        if m not in per_model:
            per_model[m] = {"calls": 0, "input": 0, "output": 0,
                            "extra_builtin": 0, "extra_mcp": 0}
        per_model[m]["calls"]         += 1
        per_model[m]["input"]         += c["input"]
        per_model[m]["output"]        += c["output"]
        per_model[m]["extra_builtin"] += c.get("extra_builtin", 0)
        per_model[m]["extra_mcp"]     += c.get("extra_mcp", 0)

    total_calls = len(calls)
    actual_cost, baseline_cost, model_saved, builtin_saved, mcp_saved, clf_overhead = _savings_breakdown(per_model, calls)
    total_saved = model_saved + builtin_saved + mcp_saved - clf_overhead
    pct         = (total_saved / baseline_cost * 100) if baseline_cost > 0 else 0.0

    W = 100
    lines = [
        f"Smart Claude Cost Savings Report - {date_str} {hour_str}:00",
        "=" * W,
        f"Reporting period : {date_str} {hour_str}:00 - {hour_str}:59",
        f"Total calls      : {total_calls}",
        f"Baseline model   : {BASELINE_MODEL}  (what every call would have cost without smart routing)",
        "",
        "PER-MODEL BREAKDOWN  (sorted by savings, baseline includes stripped tool schemas)",
        "-" * W,
        f"  {'Model':<28}  {'Calls':>5}  {'Input tok':>10}  {'Output tok':>10}  "
        f"{'Actual $':>9}  {'Sonnet $':>9}  {'Saved $':>9}  {'Saving':>7}  Notes",
        "-" * W,
    ]

    def _model_total_saved(item):
        m, v = item
        bl = _baseline_cost_usd(v["input"] + v["extra_builtin"] + v["extra_mcp"], v["output"])
        return bl - _cost_usd(m, v["input"], v["output"])

    for m, v in sorted(per_model.items(), key=_model_total_saved, reverse=True):
        in_t    = v["input"]
        out_t   = v["output"]
        eb      = v["extra_builtin"]
        em      = v["extra_mcp"]
        actual  = _cost_usd(m, in_t, out_t)
        sonnet  = _baseline_cost_usd(in_t + eb + em, out_t)
        saved   = sonnet - actual
        pct_m   = (saved / sonnet * 100) if sonnet > 0 else 0.0
        notes   = []
        if eb: notes.append(f"+{eb:,} builtin tok")
        if em: notes.append(f"+{em:,} MCP tok")
        lines.append(
            f"  {m:<28}  {v['calls']:>5}  {in_t:>10,}  {out_t:>10,}  "
            f"${actual:>8.4f}  ${sonnet:>8.4f}  ${saved:>8.4f}  {pct_m:>6.1f}%"
            + (f"  {', '.join(notes)}" if notes else "")
        )

    lines += [
        "-" * W,
        f"  {'TOTAL':<28}  {total_calls:>5}  {'':>10}  {'':>10}  "
        f"${actual_cost:>8.4f}  ${baseline_cost:>8.4f}  ${total_saved:>8.4f}  {pct:>6.1f}%",
        "",
        "SAVINGS BY HEAD",
        f"  1. Model routing        : ${model_saved:>8.4f}  - cheaper model used (Gemini/Haiku) vs claude-sonnet for same tokens",
        f"  2. Built-in tool filter : ${builtin_saved:>8.4f}  - Claude Code tool schemas not sent (Read/Write/Bash/Edit/...)",
        f"  3. MCP tool filter      : ${mcp_saved:>8.4f}  - MCP tool schemas not sent (Jira/GitHub/... complex schemas stripped)",
        f"  4. Classifier overhead  :-${clf_overhead:>8.4f}  - gemini-flash calls to classify each request (added expense)",
        "  " + "-" * 65,
        f"  Net savings             : ${total_saved:>8.4f}  ({pct:.1f}% vs unrouted claude-sonnet with full tool list)",
        "",
        f"Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]

    try:
        report_path.write_text("\n".join(lines), encoding="utf-8")
        log.info(f"Savings report written: {report_path.name}  saved ${total_saved:.4f} ({pct:.1f}%)")
    except Exception as e:
        log.warning(f"Could not write savings report: {e}")


async def _hourly_reporter() -> None:
    """Background task: every 60s, flush completed hours to savings reports."""
    while True:
        await asyncio.sleep(60)
        current_hour = datetime.now().strftime("%Y-%m-%d %H")
        async with _cost_lock:
            completed = [h for h in _cost_stats if h < current_hour]
        for h in completed:
            async with _cost_lock:
                calls = _cost_stats.pop(h, [])
            if calls:
                _write_savings_report(h, calls)


def _log_live_savings(calls: list) -> None:
    """Log a concise 3-head savings summary to the main log (called every 10 min)."""
    if not calls:
        return

    # Build per_model dict inline
    per_model: dict[str, dict] = {}
    for c in calls:
        m = c["chosen"]
        if m not in per_model:
            per_model[m] = {"calls": 0, "input": 0, "output": 0,
                            "extra_builtin": 0, "extra_mcp": 0}
        per_model[m]["calls"]         += 1
        per_model[m]["input"]         += c["input"]
        per_model[m]["output"]        += c["output"]
        per_model[m]["extra_builtin"] += c.get("extra_builtin", 0)
        per_model[m]["extra_mcp"]     += c.get("extra_mcp", 0)

    actual_cost, baseline_total, model_saved, builtin_saved, mcp_saved, clf_overhead = _savings_breakdown(per_model, calls)
    total_savings = model_saved + builtin_saved + mcp_saved - clf_overhead
    pct = (total_savings / baseline_total * 100) if baseline_total > 0 else 0.0
    model_summary = "  ".join(
        f"{m}x{v['calls']}" for m, v in sorted(per_model.items(), key=lambda x: -x[1]["calls"])
    )

    log.info(
        f"[10-min savings] {len(calls)} calls | "
        f"actual ${actual_cost:.4f} | baseline ${baseline_total:.4f} | "
        f"net saved ${total_savings:.4f} ({pct:.1f}%)  "
        f"[model-routing ${model_saved:.4f} | "
        f"builtin-tools ${builtin_saved:.4f} | "
        f"mcp-tools ${mcp_saved:.4f} | "
        f"clf-overhead -${clf_overhead:.4f}] | "
        f"models: {model_summary}"
    )


async def _live_savings_reporter() -> None:
    """Background task: log a savings summary every 10 minutes."""
    while True:
        await asyncio.sleep(600)   # 10 minutes
        current_hour = datetime.now().strftime("%Y-%m-%d %H")
        async with _cost_lock:
            # Snapshot current hour's calls without removing them
            calls = list(_cost_stats.get(current_hour, []))
        _log_live_savings(calls)


# ── Prompt history — rolling window, last 100 per model ───────────────────────

DEBUG_FILE     = LOG_DIR / "debug-requests.jsonl"
HISTORY_FILE   = LOG_DIR / "prompt-history.json"
HISTORY_MAX    = 100
_history: dict = {}
_history_lock  = asyncio.Lock()


def _load_history() -> dict:
    try:
        if HISTORY_FILE.exists():
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_history() -> None:
    try:
        HISTORY_FILE.write_text(json.dumps(_history, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        log.warning(f"Could not save prompt history: {e}")


async def _record(prompt: str, chosen: str, reason: str, response_preview: str, status: int) -> None:
    """Append one entry to the rolling per-model history."""
    async with _history_lock:
        if chosen not in _history:
            _history[chosen] = []
        _history[chosen].append({
            "ts":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "reason":   reason,
            "prompt":   prompt[:500],
            "status":   status,
            "response": response_preview[:300],
        })
        if len(_history[chosen]) > HISTORY_MAX:
            _history[chosen] = _history[chosen][-HISTORY_MAX:]
        _save_history()


# ── Debug request/response log ────────────────────────────────────────────────
# One JSON record per line in debug-requests.jsonl.
# Keeps last 500 entries (file is rewritten on each trim to avoid unbounded growth).

DEBUG_MAX = 500
_debug_lock = asyncio.Lock()

FULL_DEBUG_LOG_DIR = LOG_DIR / "full-requests"
FULL_DEBUG_LOG_DIR.mkdir(exist_ok=True)
VERBOSE_FULL_LOGGING = config.get("verbose_full_logging", False)
MAX_FULL_LOG_FILES = config.get("max_full_log_files", 100)

async def _full_debug_log(
    request_id: str,
    original_payload: dict,
    sent_payload: dict,
    received_response: dict
) -> None:
    """Log full request, sent payload, and received response to a file."""
    if not VERBOSE_FULL_LOGGING:
        return

    try:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{ts}-{request_id}.json"
        log_path = FULL_DEBUG_LOG_DIR / filename

        data = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "original_request": original_payload,
            "sent_to_llm": sent_payload,
            "received_from_llm": received_response
        }

        def do_write():
            log_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            # Rolling update: delete oldest files if count exceeds limit
            files = sorted(list(FULL_DEBUG_LOG_DIR.glob("*.json")), key=os.path.getmtime)
            if len(files) > MAX_FULL_LOG_FILES:
                for f in files[:-MAX_FULL_LOG_FILES]:
                    try:
                        f.unlink()
                    except Exception:
                        pass

        await asyncio.to_thread(do_write)  # non-blocking — don't freeze the event loop
    except Exception as e:
        log.error(f"Failed to write full debug log: {e}")


def _trim_debug_file() -> None:
    """Keep debug file under DEBUG_MAX lines."""
    try:
        if not DEBUG_FILE.exists():
            return
        lines = DEBUG_FILE.read_bytes().splitlines()
        if len(lines) > DEBUG_MAX:
            DEBUG_FILE.write_bytes(b"\n".join(lines[-DEBUG_MAX:]) + b"\n")
    except Exception:
        pass


async def _debug_log(
    original: str,
    chosen: str,
    reason: str,
    status: int,
    messages: list,
    request_tools_count: int,
    sent_tools_count: int,
    error_body=None,
    response_preview: str = "",
) -> None:
    """Append one debug record to debug-requests.jsonl."""
    # Summarise messages: role + content length, not the full content (can be huge)
    msg_summary = [
        {
            "role":    m.get("role", "?"),
            "content": (
                m["content"][:300] if isinstance(m.get("content"), str)
                else f"[{len(m['content'])} blocks]" if isinstance(m.get("content"), list)
                else str(m.get("content", ""))[:300]
            ),
        }
        for m in messages[-6:]   # last 6 messages is enough context
    ]
    record = {
        "ts":               datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "original_model":   original,
        "chosen_model":     chosen,
        "reason":           reason,
        "status":           status,
        "tools_total":      request_tools_count,
        "tools_sent":       sent_tools_count,
        "messages_total":   len(messages),
        "last_messages":    msg_summary,
        "error":            error_body,
        "response_preview": response_preview,
    }
    async with _debug_lock:
        try:
            with DEBUG_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            _trim_debug_file()
        except Exception as e:
            log.warning(f"Could not write debug log: {e}")


# ── Response helpers ───────────────────────────────────────────────────────────

def _response_preview(content: bytes) -> str:
    try:
        data = json.loads(content)
        for block in data.get("content", []):
            if isinstance(block, dict) and block.get("type") == "text":
                return block.get("text", "")[:300]
        return ""
    except Exception:
        return ""


def _extract_usage(content: bytes) -> tuple[int, int]:
    """Return (input_tokens, output_tokens) from a non-streaming response body."""
    try:
        data  = json.loads(content)
        usage = data.get("usage", {})
        return usage.get("input_tokens", 0), usage.get("output_tokens", 0)
    except Exception:
        return 0, 0


def _supports_thinking(model: str) -> bool:
    """True if the model supports Anthropic extended thinking."""
    m = model.lower()
    return "sonnet" in m or "opus" in m


# ── Startup ────────────────────────────────────────────────────────────────────

AVAILABLE_MODELS: set = set()
_history         = {}
_models_fetched  = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _history
    _history = _load_history()
    log.info(f"Loaded prompt history: {sum(len(v) for v in _history.values())} entries across {len(_history)} models")
    # Start background reporters
    asyncio.create_task(_hourly_reporter())
    asyncio.create_task(_live_savings_reporter())
    log.info("Savings reporters started (10-min log + hourly file)")
    yield

app = FastAPI(lifespan=lifespan)

# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}

# ── Main intercept ─────────────────────────────────────────────────────────────

def _merge_system_prompt(system: any) -> str:
    """Ensure system prompt is a single string.
    Gemini/Vertex AI fails if given a list or multiple system blocks."""
    if not system:
        return ""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts = []
        for block in system:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", ""))
        return "\n\n".join(p for p in parts if p.strip())
    return str(system)


def _inject_system_into_first_message(body: dict, rlog: "ReqLog") -> None:
    """Move the full system prompt into the first user message for Gemini requests.

    WHY: Gemini/Vertex AI has an undocumented ~8,000 character limit on the
    system_instruction field. Claude Code's system prompt (persona, tool usage
    rules, environment info) regularly exceeds this, causing HTTP 400
    INVALID_ARGUMENT errors. Gemini's contents array (messages) supports 1M+
    tokens -- by moving the system prompt there we bypass the limit entirely
    without truncating or losing any instructions.

    HOW: The original system text is wrapped in <SYSTEM_CONTEXT>...</SYSTEM_CONTEXT>
    delimiters and prepended to the first user message. Modern LLMs including
    Gemini are trained to treat such delimited blocks as meta-instructions.
    The system field is then cleared so Vertex AI does not see it at all.

    NOTE: This only runs for Gemini-bound requests. Claude requests are unaffected
    and continue to use the dedicated system field as normal.
    """
    sys_prompt = body.get("system", "")
    if not sys_prompt or not isinstance(sys_prompt, str) or not sys_prompt.strip():
        return  # Nothing to move

    messages = body.get("messages", [])
    if not messages:
        return  # No message list to inject into

    # Find the first user message
    first_user_idx = -1
    for i, m in enumerate(messages):
        if m.get("role") == "user":
            first_user_idx = i
            break

    if first_user_idx == -1:
        rlog.warning("Could not inject system prompt: no user message found in history")
        return

    first_msg = messages[first_user_idx]
    injection  = f"<SYSTEM_CONTEXT>\n{sys_prompt}\n</SYSTEM_CONTEXT>\n\n"
    content    = first_msg.get("content", "")

    if isinstance(content, list):
        # Content is a list of blocks -- prepend a new text block at the start
        first_msg["content"] = [{"type": "text", "text": injection}] + content
    elif isinstance(content, str):
        # Content is a plain string -- prepend directly
        first_msg["content"] = injection + content
    else:
        rlog.warning("Could not inject system prompt: unexpected content type in first user message")
        return

    # Clear the system field so Gemini's system_instruction receives nothing
    body.pop("system", None)
    rlog.info(
        f"Gemini system->message injection: moved {len(sys_prompt):,} chars from "
        f"system_instruction into first user message (avoids Vertex AI 8k limit)"
    )


@app.post("/v1/messages")
async def messages(request: Request):
    tt = Tictoc()
    req_id = _new_req_id()
    rlog   = ReqLog(req_id)

    body           = await request.json()
    original_request_payload = copy.deepcopy(body)
    original       = body.get("model", BASELINE_MODEL)
    original_tools = list(body.get("tools", []))
    is_streaming   = body.get("stream", False)

    # ── Extract API Key from incoming request headers ──────────────────────────
    # Support both "Authorization: Bearer <key>" and "X-API-Key: <key>" formats
    incoming_key = None
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        incoming_key = auth_header[7:]  # Strip "Bearer " prefix
    if not incoming_key:
        incoming_key = request.headers.get("X-API-Key")

    # STRICT: Only use incoming key, no fallback to config
    if not incoming_key:
        rlog.warning(f"No API key in request headers (Authorization or X-API-Key), rejecting request")
        return JSONResponse(
            status_code=401,
            content={"error": "Unauthorized: No API key provided in request headers"},
        )

    effective_key = incoming_key
    #rlog.info(f"Using API key from request headers: {incoming_key[:16]}...")

    # ── Lazy Model Discovery ───────────────────────────────────────────────────
    # Retry on every request until discovery succeeds (network may be unavailable
    # at startup). Once we have a model list, lock it in for the session.
    global AVAILABLE_MODELS, _models_fetched
    if not _models_fetched:
        rlog.info("Performing lazy model discovery (first request with API key)...")
        discovered = fetch_available_models(UPSTREAM, effective_key)
        if discovered:
            AVAILABLE_MODELS = discovered
            _models_fetched = True  # only lock in once we have a real model list
            rlog.info(f"Discovered {len(AVAILABLE_MODELS)} models: {sorted(AVAILABLE_MODELS)}")
        else:
            rlog.warning("Could not fetch models - will retry on next request (using passthrough this turn)")

    # Fix any invalid tool_use IDs already in conversation history before
    # forwarding -- Gemini-generated IDs with colons/dots cause Claude 400s.
    body["messages"] = _sanitize_request_messages(body.get("messages", []))

    # Capture STABLE RAW messages for hashing before any Gemini-specific mutations (flattening, etc.)
    # We do this AFTER sanitization/orphans removal as those are universal fixes.
    _remove_orphaned_tool_results(body.get("messages", []))
    raw_messages_for_cache = copy.deepcopy(body.get("messages", []))

    user_prompt = _last_user_text(body.get("messages", []))

    # ── Decide model (runs classifier in a thread so it doesn't block the event loop) ──
    if AVAILABLE_MODELS:
        tt.mark("pre-clf")
        chosen, reason, classification = await asyncio.get_event_loop().run_in_executor(
            None, decide_model, body, AVAILABLE_MODELS, UPSTREAM, effective_key
        )
        tt.mark("classifier")
    else:
        chosen, reason, classification = original, "passthrough:no-models-discovered", {
            "needs_tools": True, "complexity": "medium", "relevant_tools": []
        }

    body["model"] = chosen

    # ── Fix System Prompt ──
    # Vertex AI and some LiteLLM upstream configs require a plain string.
    # Claude Code often sends a list of content blocks -- merge into one string.
    # Apply to all models since LiteLLM may reject list-format system prompts
    # for Claude models too (e.g. "output_config: Extra inputs are not permitted").
    if "system" in body and isinstance(body["system"], list):
        merged = _merge_system_prompt(body["system"])
        if merged:
            body["system"] = merged
        else:
            body.pop("system", None)

    # ── Gemini system prompt fixes ─────────────────────────────────────────────
    if "gemini" in chosen.lower():
        # DISABLED: System injection into user messages was causing 400 errors
        # because it modified the structure of messages in the middle of tool
        # sequences (when history was trimmed to last N messages, the "first user
        # message" could be deep in a tool loop, and injecting text blocks there
        # violated Gemini's strict alternation rules).
        #
        # Instead, we now rely on Vertex AI's native system_instruction support.
        # If the system prompt is too large, Vertex AI will reject it with a clear
        # error message we can handle separately.

        # Guard against empty system strings (Vertex AI rejects system="" too).
        sys_val = body.get("system", "")
        if isinstance(sys_val, str) and not sys_val.strip():
            body.pop("system", None)

    # ── Strip thinking params for models that don't support extended thinking ──
    # LiteLLM translates "thinking" params into "output_config" for the Vertex API.
    # Claude Haiku on Vertex rejects "output_config" with 400 INVALID_ARGUMENT.
    # Strip all thinking-related fields to prevent this.
    if not _supports_thinking(chosen):
        removed = []
        for field in ("thinking", "budget_tokens", "interleaved_thinking"):
            if body.pop(field, None) is not None:
                removed.append(field)
        if removed:
            rlog.info(f"Stripped thinking-related fields {removed} (not supported by {chosen})")

    # ── Fix for Gemini 60-message history limit ──
    # ── Flatten history for Gemini (compress old turns into text summary) ───────
    # Long conversations accumulate many token-heavy messages (e.g. large Jira
    # responses). A hard 60-message trim discards context; flattening preserves it
    # as a concise text summary while reducing the payload Gemini must process.
    tt.mark("pre-sanitize")
    if "gemini" in chosen.lower() and len(body.get("messages", [])) > 20:
        body["messages"], n_flattened = _flatten_history_for_gemini(body.get("messages", []))
        if n_flattened:
            rlog.info(f"Gemini: Flattened {n_flattened} history messages into text context")

    # Vertex AI/Gemini fails with 400 INVALID_ARGUMENT if history gets too long
    # or complex (especially with many tool calls). Trim to last 60 messages
    # as a final safety net after flattening.
    if "gemini" in chosen.lower() and len(body.get("messages", [])) > 60:
        original_count = len(body["messages"])
        # Ensure we start with a 'user' message after trimming (Gemini requirement)
        body["messages"] = body["messages"][-60:]
        if body["messages"] and body["messages"][0].get("role") == "assistant":
            body["messages"].pop(0)
        rlog.info(f"Trimmed conversation history for Gemini: {original_count} -> {len(body['messages'])} messages")
        # Trimming can cut off the assistant turn that contained the tool_use
        # blocks referenced by tool_results near the new start of history.
        # Remove those orphaned tool_results so Gemini doesn't reject with 400.
        n_orphans = _remove_orphaned_tool_results(body.get("messages", []))
        if n_orphans:
            rlog.info(f"Removed {n_orphans} orphaned tool_result(s) after history trim (Gemini)")

    # ── Strip thinking blocks from message history for Gemini models ──
    # Claude Sonnet/Opus extended thinking leaves {"type":"thinking","signature":...}
    # blocks in conversation history. Vertex AI rejects these with HTTP 400.
    # Uses incremental cache: hashes the message prefix, skips re-processing
    # messages we already cleaned in prior turns (avoids 300-msg re-scan every turn).
    if "gemini" in chosen.lower():
        # Use raw_messages_for_cache for stable hashing (prefixes aren't mutated yet)
        body["messages"] = _clean_history_incremental(raw_messages_for_cache, rlog)
        # Stripping thinking blocks can remove entire assistant turns, which may
        # create new orphaned tool_results in the following user turn.
        # Run a second pass to clean those up.
        n_orphans = _remove_orphaned_tool_results(body.get("messages", []))
        if n_orphans:
            rlog.info(f"Removed {n_orphans} orphaned tool_result(s) after thinking-block strip (Gemini)")

    # ── Final Gemini role alternation fix ──────────────────────────────────────
    # Run one last merge pass after all cleanup (thinking strip, orphan removal,
    # history trim) to catch any remaining consecutive same-role messages.
    if "gemini" in chosen.lower():
        n_merges = _merge_consecutive_same_role(body.get("messages", []))
        if n_merges:
            rlog.info(f"Merged {n_merges} consecutive same-role message pair(s) for Gemini role alternation")

    # ── Split mixed user messages for Gemini ───────────────────────────────────
    # Gemini rejects user messages that mix tool_result and text blocks.
    # Split them into separate messages, inserting a dummy assistant turn if needed.
    if "gemini" in chosen.lower():
        n_splits = _split_mixed_user_messages_for_gemini(body.get("messages", []))
        if n_splits:
            rlog.info(f"Split {n_splits} mixed user message(s) (tool_result+text) for Gemini")

    # ── Ensure last user message is not tool_results only ────────────────────
    # Gemini rejects requests where the last message is purely tool_result blocks
    # (function responses) with no text. Append dummy turns so it ends with text.
    if "gemini" in chosen.lower():
        if _ensure_last_not_tool_result(body.get("messages", [])):
            rlog.info("Gemini fix: Appended dummy turns to satisfy 'last message cannot be tool_result' rule")

    # ── Merge/clean all multi-text-block messages for Gemini ────────────────
    # Gemini rejects user messages that contain multiple text blocks.
    # Apply _drop_empty_texts to every message content list to merge them.
    if "gemini" in chosen.lower():
        n_text_merges = 0
        for _msg in body.get("messages", []):
            _content = _msg.get("content")
            if isinstance(_content, list):
                _cleaned = _drop_empty_texts(_content)
                if len(_cleaned) != len(_content):
                    n_text_merges += 1
                _msg["content"] = _cleaned
        if n_text_merges:
            rlog.info(f"Merged/cleaned text blocks in {n_text_merges} message(s) for Gemini")

    # ── Flatten tool_result content for Gemini (must be string, not list) ────
    if "gemini" in chosen.lower():
        n_flat = _flatten_tool_results_for_gemini(body.get("messages", []))
        if n_flat:
            rlog.info(f"Flattened {n_flat} list-style tool_result block(s) for Gemini")

    # ── Normalize plain-string message content to list for Gemini ─────────────
    # LiteLLM/Vertex AI rejects messages whose content is a bare string instead
    # of a list of typed blocks. Convert any str content to [{"type":"text",...}].
    if "gemini" in chosen.lower():
        n_str_norm = 0
        for _msg in body.get("messages", []):
            if isinstance(_msg.get("content"), str):
                _msg["content"] = [{"type": "text", "text": _msg["content"]}]
                n_str_norm += 1
        if n_str_norm:
            rlog.info(f"Normalized {n_str_norm} plain-string message content(s) to block list for Gemini")

    # ── Sanitize tool schemas for Gemini ─────────────────────────────────────
    if "gemini" in chosen.lower() and body.get("tools"):
        body["tools"], n_builtin, n_mcp = _sanitize_tools_for_gemini(body["tools"])
        rlog.info(
            f"Sanitized {n_builtin + n_mcp} tool schemas for Gemini "
            f"({n_builtin} builtin fully-sanitized + {n_mcp} MCP minimal-sanitized)"
        )
        # Serialise any dict/list tool_use input values in history to JSON strings.
        # Only needed for builtin tools (full sanitization flattened their schemas);
        # MCP tools keep their original structure so no conversion needed.
        n_inp = _sanitize_tool_use_inputs_for_gemini(
            body.get("messages", []), body["tools"])
        if n_inp:
            rlog.info(f"Serialised {n_inp} complex tool_use input(s) to JSON string for Gemini")

    # ── Final Gemini Message Validation ────────────────────────────────────────
    # Strip empty text blocks from assistant messages that appear before tool_use.
    # Gemini is sensitive to empty/whitespace-only blocks in complex sequences.
    if "gemini" in chosen.lower():
        n_empty_stripped = 0
        for i, msg in enumerate(body.get("messages", [])):
            if msg.get("role") == "assistant":
                content = msg.get("content", [])
                if isinstance(content, list):
                    # Remove any text blocks that are empty/whitespace-only
                    cleaned = [
                        b for b in content
                        if not (isinstance(b, dict) and b.get("type") == "text"
                                and not b.get("text", "").strip())
                    ]
                    if len(cleaned) < len(content):
                        n_empty_stripped += 1
                        msg["content"] = cleaned if cleaned else [{"type": "text", "text": "OK"}]
        if n_empty_stripped:
            rlog.info(f"Stripped {n_empty_stripped} empty text block(s) from assistant message(s) for Gemini")

        # Final safety pass: remove orphaned tool_results and ensure no empty messages
        n_orphans = _sanitize_gemini_history(body.get("messages", []))
        if n_orphans:
            rlog.info(f"Removed {n_orphans} orphaned tool_result(s) in final Gemini sanitization pass")

    # ── Smart tool filtering ──────────────────────────────────────────────────
    needs_tools = classification.get("needs_tools", True)
    relevant    = set(classification.get("relevant_tools", []))

    # Track how many tokens were saved by not sending tool schemas.
    # Split into builtin (Read/Write/Bash/...) vs MCP (Jira/GitHub/...) for reporting.
    extra_builtin = 0   # tokens from stripped Claude Code built-in tool schemas
    extra_mcp     = 0   # tokens from stripped MCP tool schemas

    def _schema_tokens(tool_list: list) -> tuple[int, int]:
        """Returns (builtin_tokens, mcp_tokens) for a list of tool objects."""
        builtin = [t for t in tool_list if t.get("name") in BUILTIN_TOOLS]
        mcp     = [t for t in tool_list if t.get("name") not in BUILTIN_TOOLS]
        return (
            len(json.dumps(builtin).encode()) // 4 if builtin else 0,
            len(json.dumps(mcp).encode())     // 4 if mcp     else 0,
        )

    if original_tools:
        if not needs_tools:
            extra_builtin, extra_mcp = _schema_tokens(original_tools)
            body.pop("tools", None)
            body.pop("tool_choice", None)
            rlog.info(
                f"Tools: stripped all {len(original_tools)} "
                f"(~{extra_builtin:,} builtin + ~{extra_mcp:,} MCP tokens saved)"
            )
        elif relevant:
            # Filter from body["tools"] (may be Gemini-sanitized) not original_tools
            _src         = body.get("tools", original_tools)
            filtered     = [t for t in _src if t.get("name") in relevant]
            not_filtered = [t for t in original_tools if t.get("name") not in relevant]
            if len(filtered) < len(original_tools):
                extra_builtin, extra_mcp = _schema_tokens(not_filtered)
                body["tools"] = filtered
                stripped = len(original_tools) - len(filtered)
                rlog.info(
                    f"Tools: {len(original_tools)} -> {len(filtered)} "
                    f"(stripped {stripped}: ~{extra_builtin:,} builtin + ~{extra_mcp:,} MCP tokens saved)"
                )

    rlog.info(f"[{reason}] {original} -> {chosen} (Tools: {', '.join(relevant) if relevant else 'none'})")

    upstream_url = f"{UPSTREAM}/v1/messages"
    headers = {
        "Authorization":     f"Bearer {effective_key}",
        "Content-Type":      "application/json",
        "anthropic-version": request.headers.get("anthropic-version", "2023-06-01"),
    }

    # Classifier overhead: cost of the gemini-flash call used to route this request.
    # This is an ADDED expense (not present before the proxy) and must be counted
    # against savings rather than ignored.
    clf_model = classification.get("_clf_model", "")
    clf_in    = classification.get("_clf_input",  0)
    clf_out   = classification.get("_clf_output", 0)
    clf_cost  = _cost_usd(clf_model, clf_in, clf_out) if clf_model else 0.0
    if clf_cost:
        rlog.info(f"Classifier: {clf_model} ({clf_in} in / {clf_out} out tokens, overhead ${clf_cost:.6f})")

    req_tools_count  = len(original_tools)
    sent_tools_count = len(body.get("tools", []))
    orig_messages    = body.get("messages", [])

    # Build a chain of fallback models to try before giving up and using original.
    # e.g. gemini-flash fails -> try gemini-2.5-flash -> then fall back to claude-sonnet.
    fallback_chain = [m for m in classification.get("_chain", [])
                      if m in AVAILABLE_MODELS and m != chosen]

    # Ensure claude-haiku is in the chain for low/medium complexity tasks
    # to avoid falling back directly from Gemini to the expensive Sonnet.
    if classification.get("complexity") in ("low", "medium") and "claude-haiku" in AVAILABLE_MODELS:
        if "claude-haiku" not in fallback_chain and chosen != "claude-haiku":
            fallback_chain.append("claude-haiku")

    tt.mark("sanitize")
    rlog.info(f"Timing: {tt.summary()}")
    if is_streaming:
        return StreamingResponse(
            _stream(upstream_url, headers, body, chosen, original, user_prompt, reason,
                    extra_builtin, extra_mcp, clf_model, clf_cost,
                    orig_messages, req_tools_count, sent_tools_count,
                    fallback_chain=fallback_chain, rlog=rlog,
                    original_request_payload=original_request_payload),
            media_type="text/event-stream",
        )
    return await _forward(upstream_url, headers, body, chosen, original, user_prompt, reason,
                          extra_builtin, extra_mcp, clf_model, clf_cost,
                          orig_messages, req_tools_count, sent_tools_count,
                          fallback_chain=fallback_chain, rlog=rlog,
                          original_request_payload=original_request_payload)


# ── Model field rewriters ──────────────────────────────────────────────────────

# ── Tool-use ID sanitization ───────────────────────────────────────────────────
# Gemini (via LiteLLM) sometimes generates tool_use IDs containing characters
# outside [a-zA-Z0-9_-] (e.g. colons in UUIDs, dots).  Claude Code echoes
# those IDs back in the next turn as tool_result.tool_use_id, and Claude
# models then reject the whole request with a 400.  Sanitize on every response.

_BAD_TOOL_ID = re.compile(r"[^a-zA-Z0-9_-]")


def _fix_tool_id(tid: str) -> str:
    return _BAD_TOOL_ID.sub("_", tid)


def _sanitize_tool_ids_inplace(data: dict) -> None:
    """Fix tool_use.id fields in a parsed Anthropic response dict."""
    # Non-streaming: content array at top level
    for block in data.get("content", []):
        if isinstance(block, dict) and block.get("type") == "tool_use":
            if "id" in block:
                block["id"] = _fix_tool_id(block["id"])
    # Streaming message_start wrapper
    msg = data.get("message")
    if isinstance(msg, dict):
        for block in msg.get("content", []):
            if isinstance(block, dict) and block.get("type") == "tool_use":
                if "id" in block:
                    block["id"] = _fix_tool_id(block["id"])
    # Streaming content_block_start
    cb = data.get("content_block")
    if isinstance(cb, dict) and cb.get("type") == "tool_use":
        if "id" in cb:
            cb["id"] = _fix_tool_id(cb["id"])


def _sanitize_request_messages(messages: list) -> list:
    """Sanitize the INCOMING request messages array before forwarding.
    Returns a new list with empty messages/blocks removed.
    """
    cleaned_messages = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            if isinstance(content, str) and content.strip():
                cleaned_messages.append(msg)
            continue

        # 1. Drop empty text blocks
        content = [
            b for b in content
            if not (isinstance(b, dict) and b.get("type") == "text"
                    and not b.get("text", "").strip())
        ]

        # 2. Fix tool IDs
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use" and "id" in block:
                block["id"] = _fix_tool_id(block["id"])
            elif block.get("type") == "tool_result":
                if "tool_use_id" in block:
                    block["tool_use_id"] = _fix_tool_id(block["tool_use_id"])
                nested = block.get("content")
                if isinstance(nested, list):
                    block["content"] = [
                        nb for nb in nested
                        if not (isinstance(nb, dict) and nb.get("type") == "text"
                                and not nb.get("text", "").strip())
                    ]

        # 3. Only keep the message if it still has content
        if content:
            msg["content"] = content
            cleaned_messages.append(msg)

    return cleaned_messages


def _remove_orphaned_tool_results(messages: list) -> int:
    """Remove tool_result blocks (and OpenAI-style tool-role messages) that have
    no matching tool_use in any preceding assistant message.

    This fixes two failure modes that both surface after /context compaction or
    auto-compaction rewrites conversation history:

    1. HTTP 500 from LiteLLM/Gemini:
       "Missing corresponding tool call for tool response message"
       The compacted history drops the assistant turn that contained the
       tool_use block, leaving a dangling tool_result in a user message.

    2. HTTP 400 INVALID_ARGUMENT from Vertex AI / Gemini:
       Same root cause -- Gemini strictly requires every tool_result to have
       a matching tool_use in the immediately-preceding assistant turn.

    Returns the number of orphaned blocks removed.
    """
    # Collect all tool_use IDs present in assistant messages
    valid_tool_ids: set[str] = set()
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                tid = block.get("id", "")
                if tid:
                    valid_tool_ids.add(tid)

    removed = 0
    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg.get("role", "")

        # OpenAI-style: role=tool messages (LiteLLM sometimes converts these)
        if role == "tool":
            tid = msg.get("tool_call_id", "")
            if tid not in valid_tool_ids:
                log.warning(
                    f"Removing orphaned tool-role message (tool_call_id={tid!r} "
                    f"has no matching tool_use in history)"
                )
                messages.pop(i)
                removed += 1
                continue
            i += 1
            continue

        # Anthropic-style: role=user with tool_result blocks
        if role == "user":
            content = msg.get("content", [])
            if not isinstance(content, list):
                i += 1
                continue
            new_content = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tid = block.get("tool_use_id", "")
                    if tid not in valid_tool_ids:
                        log.warning(
                            f"Removing orphaned tool_result block (tool_use_id={tid!r} "
                            f"has no matching tool_use in history)"
                        )
                        removed += 1
                        continue  # drop this block
                new_content.append(block)
            if not new_content:
                # Entire user message was orphaned tool_results -- drop the message
                messages.pop(i)
                continue
            msg["content"] = new_content

        i += 1

    return removed



def _sanitize_tools_for_gemini(tools: list) -> tuple[list, int, int]:
    """Sanitize tool schemas for Gemini with two strategies:

    - Builtin Claude Code tools (Read/Write/Bash/...): FULL sanitization.
      Flattens nested objects/arrays to strings and strips all invalid keys.
      These schemas are Claude Code internals and are safe to aggressively simplify.

    - MCP tools (Jira/GitHub/Confluence/...): MINIMAL sanitization.
      Only removes Gemini-incompatible keywords while preserving structure.
      Aggressive flattening breaks MCP tool calls because the MCP server
      expects the original parameter types (objects, arrays) that Gemini
      would then return as JSON strings that the server can't parse.

    Returns (sanitized_tools, n_builtin_sanitized, n_mcp_sanitized).
    """

    def _clean_full(schema: dict) -> dict:
        """Full sanitization for builtin tools — maximum Gemini compatibility."""
        if not isinstance(schema, dict):
            return {"type": "string"}
        schema = dict(schema)

        # Flatten anyOf / oneOf / allOf
        for combiner in ("anyOf", "oneOf", "allOf"):
            if combiner in schema:
                candidates = schema.pop(combiner)
                if not isinstance(candidates, list) or not candidates:
                    continue
                chosen = next(
                    (c for c in candidates
                     if isinstance(c, dict) and c.get("type") not in (None, "null")),
                    candidates[0],
                )
                if isinstance(chosen, dict):
                    for k, v in chosen.items():
                        if k not in schema:
                            schema[k] = v

        # Normalise list-style types  e.g. ["string", "null"] -> "string"
        raw_type = schema.get("type")
        if isinstance(raw_type, list):
            schema["type"] = next((t for t in raw_type if t != "null"), "string")

        # Strip EVERY key Vertex AI does not understand
        allowed_keys = {"type", "properties", "items", "description", "required", "enum"}
        for k in list(schema.keys()):
            if k not in allowed_keys:
                schema.pop(k)

        # Ensure type is set
        if "type" not in schema:
            if "properties" in schema:
                schema["type"] = "object"
            elif "items" in schema:
                schema["type"] = "array"
            else:
                schema["type"] = "string"

        # Flatten nested properties to string (Gemini rejects nested objects/arrays)
        if "properties" in schema and isinstance(schema["properties"], dict):
            new_props = {}
            for prop_name, prop_schema in schema["properties"].items():
                cleaned = _clean_full(prop_schema)
                if cleaned.get("type") in ("object", "array"):
                    desc = cleaned.get("description", "")
                    new_props[prop_name] = {
                        "type": "string",
                        "description": ("(JSON-encoded) " + desc).strip(),
                    }
                else:
                    new_props[prop_name] = cleaned
            schema["properties"] = new_props

        # Flatten array items to string
        if "items" in schema:
            cleaned_items = _clean_full(schema["items"])
            if cleaned_items.get("type") in ("object", "array"):
                schema["items"] = {"type": "string"}
            else:
                schema["items"] = cleaned_items

        return schema

    def _clean_minimal(schema: dict) -> dict:
        """Minimal sanitization for MCP tools — fix invalid keywords, preserve structure.

        MCP tool schemas often have nested objects (e.g. Jira additional_fields).
        Flattening them to strings breaks tool calls because the MCP server expects
        the original object types. We only remove keywords that Gemini/Vertex AI
        actually rejects, leaving the structure intact.
        """
        if not isinstance(schema, dict):
            return schema
        schema = dict(schema)

        # Remove keywords Gemini/Vertex AI cannot handle
        for key in ("$schema", "$defs", "$ref", "nullable",
                    "exclusiveMinimum", "exclusiveMaximum",
                    "const", "if", "then", "else", "not",
                    "unevaluatedProperties", "additionalProperties"):
            schema.pop(key, None)

        # Flatten combiners
        for combiner in ("anyOf", "oneOf", "allOf"):
            if combiner in schema:
                candidates = schema.pop(combiner)
                if not isinstance(candidates, list) or not candidates:
                    continue
                chosen = next(
                    (c for c in candidates
                     if isinstance(c, dict) and c.get("type") not in (None, "null")),
                    candidates[0],
                )
                if isinstance(chosen, dict):
                    for k, v in chosen.items():
                        if k not in schema:
                            schema[k] = v

        # Normalise list-style types  e.g. ["string", "null"] -> "string"
        raw_type = schema.get("type")
        if isinstance(raw_type, list):
            schema["type"] = next((t for t in raw_type if t != "null"), "string")

        # Recursively clean nested properties (preserve object/array structure)
        if "properties" in schema and isinstance(schema["properties"], dict):
            schema["properties"] = {k: {"type": "string", "description": "(JSON) " + v.get("description", "")} if isinstance(v, dict) and (v.get("type") in ("object", "array") or "properties" in v) else _clean_minimal(v) for k, v in schema["properties"].items()}

        if "items" in schema and isinstance(schema["items"], dict):
            schema["items"] = _clean_minimal(schema["items"])

        return schema

    result = []
    n_builtin = 0
    n_mcp = 0
    for tool in tools:
        tool = copy.deepcopy(tool)
        name = tool.get("name", "")
        is_builtin = name in BUILTIN_TOOLS
        for schema_key in ("input_schema", "parameters"):
            if isinstance(tool.get(schema_key), dict):
                if is_builtin:
                    cleaned = _clean_full(tool[schema_key])
                    if "type" not in cleaned:
                        cleaned["type"] = "object"
                    tool[schema_key] = cleaned
                    n_builtin += 1
                else:
                    tool[schema_key] = _clean_minimal(tool[schema_key])
                    n_mcp += 1
                break
        result.append(tool)
    return result, n_builtin, n_mcp



def _flatten_tool_results_for_gemini(messages: list) -> int:
    """Convert list-style tool_result content to plain string for Gemini.

    Anthropic allows tool_result content to be a list of content blocks.
    Gemini / Vertex AI only accepts plain strings. Flatten them.
    Returns number of blocks converted.
    """
    converted = 0
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            c = block.get("content")
            if isinstance(c, list):
                parts = []
                for item in c:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        parts.append(item)
                block["content"] = "\n".join(parts)
                converted += 1
    return converted


def _sanitize_tool_use_inputs_for_gemini(messages: list, tools: list) -> int:
    """Serialise complex tool_use input values to JSON strings — BUILTIN tools only.

    _sanitize_tools_for_gemini() applies FULL sanitization to builtin Claude Code
    tools (Read/Write/Bash/...), flattening all nested object/array properties to
    'string'. Any tool_use block in history that passed those properties as real
    dicts/lists must also be serialised to JSON strings so the value type matches
    the schema type, otherwise Gemini rejects with 400.

    MCP tools (Jira/GitHub/Confluence/...) receive MINIMAL sanitization that
    preserves their object/array structure. Their inputs must NOT be serialised —
    converting them to strings would create a type mismatch with the preserved schema
    and break MCP tool calls.

    Returns number of values converted.
    """
    import json

    # Only apply to builtin tools (schema was fully flattened to strings).
    # MCP tool names start with "mcp__" — skip those.
    builtin_tool_names = {
        t.get("name", "") for t in tools
        if isinstance(t, dict) and not t.get("name", "").startswith("mcp__")
        and t.get("name", "") in BUILTIN_TOOLS
    }

    converted = 0
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            tool_name = block.get("name", "")
            if tool_name not in builtin_tool_names:
                continue  # Skip MCP tools — their inputs stay as-is
            inputs = block.get("input", {})
            if not isinstance(inputs, dict):
                continue
            for key, val in list(inputs.items()):
                if isinstance(val, (dict, list)):
                    inputs[key] = json.dumps(val, ensure_ascii=False)
                    converted += 1
    return converted



def _flatten_history_for_gemini(messages: list, keep_recent: int = 20) -> tuple[list, int]:
    """Compress old conversation history into a text summary to reduce tokens for Gemini.

    Long conversations (even after the 60-message trim) can contain many token-heavy
    tool results (e.g. large Jira API responses). Flattening old turns into a text
    summary reduces the payload while preserving context.

    Keeps the most recent `keep_recent` messages intact.
    Tries to find a clean boundary — the last user text message before `keep_recent` —
    to avoid splitting mid-tool-chain.

    Returns (new_messages, n_compressed).
    """
    if len(messages) <= keep_recent:
        return messages, 0

    # Find the best boundary: last "clean" user message (text only, no tool_results)
    # before the keep_recent window — so we don't cut mid-tool-chain.
    safe_start = len(messages) - keep_recent
    for i in range(len(messages) - keep_recent - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") != "user":
            continue
        content = msg.get("content", [])
        has_tool_results = False
        has_text = False
        if isinstance(content, str):
            has_text = bool(content.strip())
        elif isinstance(content, list):
            for b in content:
                if isinstance(b, dict):
                    if b.get("type") == "tool_result":
                        has_tool_results = True
                    elif b.get("type") == "text" and b.get("text", "").strip():
                        has_text = True
        if has_text and not has_tool_results:
            safe_start = i
            break

    if safe_start == 0:
        return messages, 0

    to_compress = messages[:safe_start]
    preserved  = messages[safe_start:]

    # Build a concise text summary of the compressed portion
    parts = ["[Conversation History Summary]"]
    for m in to_compress:
        role    = m.get("role", "?")
        content = m.get("content", "")
        if isinstance(content, str):
            if content.strip():
                parts.append(f"{role.upper()}: {content[:300]}")
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text" and block.get("text", "").strip():
                    parts.append(f"{role.upper()}: {block['text'][:300]}")
                elif btype == "tool_use":
                    parts.append(f"ASSISTANT called tool: {block.get('name', '?')}")
    parts.append("[End of history summary]")

    summary_msg = {
        "role": "user",
        "content": [{"type": "text", "text": "\n".join(parts)}],
    }

    # Ensure role alternation: if preserved starts with user, add a dummy assistant turn
    new_messages = [summary_msg]
    if preserved and preserved[0].get("role") == "user":
        new_messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": "Understood. Continuing from where we left off."}],
        })
    new_messages.extend(preserved)

    return new_messages, len(to_compress)


def _ensure_last_not_tool_result(messages: list) -> bool:
    """Fix Gemini restriction: the last user message cannot consist solely of tool_results.

    Gemini (via LiteLLM) translates Anthropic tool_result blocks into Gemini function
    responses. If the final message in the conversation is ONLY function responses with
    no text, Gemini rejects the request (it needs an explicit user text prompt to know
    what to generate next).

    Fix: append a dummy assistant acknowledgement + a dummy user "Continue" turn so the
    conversation ends with a plain text user message.

    Returns True if dummy turns were appended.
    """
    if not messages:
        return False
    last = messages[-1]
    if last.get("role") != "user":
        return False
    content = last.get("content", [])
    if not isinstance(content, list):
        return False
    has_tool_results = any(
        isinstance(b, dict) and b.get("type") == "tool_result" for b in content
    )
    has_text = any(
        isinstance(b, dict) and b.get("type") == "text" and b.get("text", "").strip()
        for b in content
    )
    if has_tool_results and not has_text:
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": "I have received the tool results."}],
        })
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": "Continue."}],
        })
        return True
    return False


def _drop_empty_texts(blocks: list) -> list:
    """Remove empty text blocks AND merge consecutive text blocks."""
    if not isinstance(blocks, list): return blocks

    # Pass 1: Remove empty
    cleaned = [
        b for b in blocks
        if not (isinstance(b, dict) and b.get("type") == "text" and not b.get("text", "").strip())
    ]

    # Pass 2: Merge consecutive text blocks
    if not cleaned: return cleaned
    merged = []
    for b in cleaned:
        if merged and isinstance(b, dict) and b.get("type") == "text" and            isinstance(merged[-1], dict) and merged[-1].get("type") == "text":
            merged[-1]["text"] = merged[-1]["text"].rstrip() + "\n\n" + b.get("text", "").lstrip()
        else:
            merged.append(b)
    return merged

def _merge_consecutive_same_role(messages: list) -> int:
    """Merge consecutive messages with the same role so Gemini is happy.

    Also strips empty/whitespace-only text blocks from every message
    as a side-effect, since Gemini rejects those too.
    """
    if len(messages) < 2:
        return 0

    # Pass 1 – strip empty text blocks from all messages
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            msg["content"] = [
                b for b in content
                if not (isinstance(b, dict)
                        and b.get("type") == "text"
                        and not b.get("text", "").strip())
            ]

    # Pass 2 – merge consecutive same-role messages
    merged_count = 0
    i = 0
    while i < len(messages) - 1:
        if messages[i]["role"] == messages[i + 1]["role"]:
            c1 = messages[i].get("content")
            c2 = messages[i + 1].get("content")

            if isinstance(c1, list) and isinstance(c2, list):
                messages[i]["content"] = c1 + c2
            elif isinstance(c1, str) and isinstance(c2, str):
                messages[i]["content"] = c1 + "\n" + c2
            else:
                # Normalise both to list then merge
                def _to_list(c):
                    if isinstance(c, list):
                        return c
                    return [{"type": "text", "text": str(c)}]
                messages[i]["content"] = _to_list(c1) + _to_list(c2)

            messages.pop(i + 1)
            merged_count += 1
        else:
            i += 1

    return merged_count


def _split_mixed_user_messages_for_gemini(messages: list) -> int:
    """Split user messages that contain both tool_result and text blocks.

    Gemini/Vertex AI rejects user turns that mix tool_result blocks with text
    blocks. When this pattern is found, the message is split into two:
      - A user message containing only the tool_result blocks
      - A user message containing only the text blocks
    A dummy assistant message is inserted between them to maintain role alternation.

    Returns the number of messages that were split.
    """
    split_count = 0
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.get("role") != "user":
            i += 1
            continue
        content = msg.get("content", [])
        if not isinstance(content, list):
            i += 1
            continue

        tool_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]
        text_blocks = [b for b in content if isinstance(b, dict) and b.get("type") != "tool_result"]

        if not tool_blocks or not text_blocks:
            # No mix -- nothing to split
            i += 1
            continue

        # Replace this message with: [tool_result msg] + [dummy assistant] + [text msg]
        tool_msg        = {"role": "user",      "content": tool_blocks}
        dummy_assistant = {"role": "assistant", "content": [{"type": "text", "text": "OK"}]}
        text_msg        = {"role": "user",      "content": text_blocks}

        messages[i:i+1] = [tool_msg, dummy_assistant, text_msg]
        split_count += 1
        i += 3  # skip past the three new messages

    return split_count


def _strip_thinking_blocks(messages: list) -> int:
    """Convert Claude extended-thinking blocks to plain text in assistant messages.
    Gemini/Vertex AI rejects the raw thinking type with HTTP 400 'Thought signature is not valid'.
    Instead of discarding them, we preserve the reasoning as a [Thought Process] text block
    so the model retains continuity across turns.

    Handles three cascading problems:
    1. Thinking blocks converted/stripped -> assistant message may become empty -> remove it.
    2. Removed assistant message -> next user message may have orphaned tool_results
       (tool_results with no matching tool_use) -> remove that user message too.
    3. Above removals may create consecutive same-role messages -> merge/drop them.
    """
    stripped = 0
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.get("role") != "assistant":
            i += 1
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            i += 1
            continue

        thoughts = []
        new_content = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "thinking":
                thought_text = b.get("thinking", "").strip()
                if thought_text:
                    thoughts.append(thought_text)
                stripped += 1
            else:
                new_content.append(b)

        if not thoughts and len(new_content) == len(content):
            # Nothing changed
            i += 1
            continue

        # Append captured thoughts as a plain text block so Gemini sees normal text
        if thoughts:
            thought_footer = "\n\n[Thought Process]\n" + "\n".join(thoughts)
            # Try to append to an existing text block first
            text_block_exists = False
            for block in new_content:
                if isinstance(block, dict) and block.get("type") == "text":
                    block["text"] += thought_footer
                    text_block_exists = True
                    break
            # No existing text block found -- create a new one
            if not text_block_exists:
                new_content.append({"type": "text", "text": thought_footer})

        if new_content:
            # Keep the assistant message with thoughts converted to text
            msg["content"] = new_content
            i += 1
        else:
            # Assistant message had ONLY thinking blocks -- remove it entirely
            messages.pop(i)
            # The next user message may now be orphaned (tool_results with no
            # matching tool_use above them). Remove it too if so.
            if i < len(messages):
                next_msg = messages[i]
                if next_msg.get("role") == "user":
                    next_content = next_msg.get("content", [])
                    if (isinstance(next_content, list) and next_content
                            and all(isinstance(b, dict) and b.get("type") == "tool_result"
                                    for b in next_content)):
                        messages.pop(i)
            # Do not increment i -- re-check same position after removal

    # Final pass: MERGE consecutive same-role messages.
    # Dropping is wrong — the second message may contain tool_use blocks that are
    # essential for the conversation (dropping them orphans the next tool_result).
    # Merging preserves all content while satisfying Gemini's strict alternation rule.
    _merge_consecutive_same_role(messages)

    return stripped


def _strip_empty_text_blocks(data: dict) -> None:
    """Remove empty/whitespace-only text blocks from response content.
    Gemini via LiteLLM sometimes emits {"type":"text","text":""} blocks.
    Claude Code stores these in history, then Claude rejects them on replay.
    Handles both top-level content and nested content inside tool_result blocks.
    """
    content = data.get("content")
    if not isinstance(content, list):
        return
    cleaned = _drop_empty_texts(content)
    # Also clean nested content inside tool_result blocks
    for block in cleaned:
        if isinstance(block, dict) and block.get("type") == "tool_result":
            nested = block.get("content")
            if isinstance(nested, list):
                block["content"] = _drop_empty_texts(nested)
    data["content"] = cleaned


def _rewrite_model(content: bytes, model: str) -> bytes:
    try:
        data = json.loads(content)
        if "model" in data:
            data["model"] = model
        _strip_empty_text_blocks(data)
        _sanitize_tool_ids_inplace(data)
        return json.dumps(data).encode()
    except Exception:
        return content


def _rewrite_sse_chunk(chunk: bytes, model: str) -> bytes:
    if b'"model"' not in chunk and b'"tool_use"' not in chunk and b'"text"' not in chunk:
        return chunk
    try:
        text   = chunk.decode()
        lines  = text.split("\n")
        result = []
        for line in lines:
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    if "model" in data:
                        data["model"] = model
                    if "message" in data and isinstance(data["message"], dict):
                        msg = data["message"]
                        if "model" in msg:
                            msg["model"] = model
                        _strip_empty_text_blocks(msg)
                    _sanitize_tool_ids_inplace(data)
                    line = "data: " + json.dumps(data)
                except Exception:
                    pass
            result.append(line)
        return "\n".join(result).encode()
    except Exception:
        return chunk


# ── SSE token extractor ────────────────────────────────────────────────────────

def _scan_sse_for_usage(chunk: bytes, state: dict) -> None:
    """
    Scan one SSE chunk for token usage events and update state in place.
    state = {"input": int, "output": int}
    Anthropic SSE format:
      message_start  -> message.usage.input_tokens
      message_delta  -> usage.output_tokens  (final output count)
    """
    try:
        for line in chunk.decode(errors="replace").split("\n"):
            if not line.startswith("data: "):
                continue
            try:
                data = json.loads(line[6:])
                if data.get("type") == "message_start":
                    u = data.get("message", {}).get("usage", {})
                    state["input"]  = u.get("input_tokens",  state["input"])
                    state["output"] = u.get("output_tokens", state["output"])
                elif data.get("type") == "message_delta":
                    u = data.get("usage", {})
                    if "output_tokens" in u:
                        state["output"] = u["output_tokens"]
            except Exception:
                pass
    except Exception:
        pass


def _sanitize_gemini_history(messages: list) -> int:
    """Final safety pass for Gemini history before sending to Vertex AI.
    Ensures:
    1. No assistant message is completely empty.
    2. No orphaned tool_results (results without matching tool_use in history).
    Returns: number of orphaned tool_results removed.
    """
    known_tool_ids = set()
    orphans_removed = 0

    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg.get("role")
        content = msg.get("content")

        if role == "assistant":
            # Track tool IDs seen so far
            if isinstance(content, list):
                has_text = False
                has_tool_use = False
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "tool_use":
                        known_tool_ids.add(block.get("id"))
                        has_tool_use = True
                    elif block.get("type") == "text" and block.get("text", "").strip():
                        has_text = True

                # Gemini often requires at least one text block in every assistant
                # message, even if it also contains tool_use.
                if not has_text:
                    content.insert(0, {"type": "text", "text": "I'll use a tool." if has_tool_use else "..."})
            elif not content or (isinstance(content, str) and not content.strip()):
                msg["content"] = "..."
            elif isinstance(content, str):
                # If it's a string, it has text.
                pass

        elif role == "user" and isinstance(content, list):
            # Remove orphaned tool results (tool_result without matching tool_use)
            new_content = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    if block.get("tool_use_id") in known_tool_ids:
                        new_content.append(block)
                    else:
                        orphans_removed += 1
                else:
                    new_content.append(block)

            if not new_content:
                # User message became empty after removing orphans, drop the message
                messages.pop(i)
                continue
            msg["content"] = new_content

        i += 1

    return orphans_removed


def _log_400_detail(body: dict, rlog) -> None:
    """Log a detailed breakdown of the request body to help diagnose Gemini 400 errors.

    Vertex AI returns a generic "Request contains an invalid argument" for many different
    problems. This function logs enough detail to identify the actual cause:
      - system field presence / length
      - tool names and whether their schemas have nested object/array types
      - per-message content block types (text, tool_use, tool_result, thinking, image, ...)
      - any suspicious patterns (empty content, consecutive same-role messages, ...)
    """
    # --- System field ---
    sys_val  = body.get("system")
    sys_len  = len(sys_val) if isinstance(sys_val, str) else (len(sys_val) if isinstance(sys_val, list) else 0)
    rlog.warning(f"  [400-diag] system={type(sys_val).__name__}(len={sys_len})")

    # --- Tools ---
    tools = body.get("tools", [])
    if tools:
        for t in tools:
            name   = t.get("name", "?")
            schema = t.get("input_schema", t.get("parameters", {}))
            props  = list(schema.get("properties", {}).keys()) if isinstance(schema, dict) else []
            nested = [p for p, v in (schema.get("properties", {}) if isinstance(schema, dict) else {}).items()
                      if isinstance(v, dict) and v.get("type") in ("object", "array")]
            rlog.warning(f"  [400-diag] tool={name!r} props={props} nested_types={nested}")
    else:
        rlog.warning("  [400-diag] tools=(none)")

    # --- Messages ---
    messages  = body.get("messages", [])
    prev_role = None
    for i, msg in enumerate(messages):
        role    = msg.get("role", "?")
        content = msg.get("content", "")
        if isinstance(content, str):
            block_summary = f"str(len={len(content)})"
        elif isinstance(content, list):
            types       = [b.get("type", "?") if isinstance(b, dict) else type(b).__name__ for b in content]
            empty_texts = sum(1 for b in content
                              if isinstance(b, dict) and b.get("type") == "text"
                              and not b.get("text", "").strip())
            block_summary = f"blocks={types}"
            if empty_texts:
                block_summary += f" EMPTY_TEXT={empty_texts}"
        else:
            block_summary = f"UNEXPECTED={type(content).__name__}"

        consecutive = " *** CONSECUTIVE_SAME_ROLE ***" if role == prev_role else ""
        rlog.warning(f"  [400-diag] msg[{i}] role={role} {block_summary}{consecutive}")
        prev_role = role



# ── Non-streaming ──────────────────────────────────────────────────────────────

async def _forward(url, headers, body, chosen, original, user_prompt: str = "",
                   reason: str = "", extra_builtin: int = 0, extra_mcp: int = 0,
                   clf_model: str = "", clf_cost: float = 0.0,
                   orig_messages: list | None = None,
                   req_tools_count: int = 0, sent_tools_count: int = 0,
                   fallback_chain: list | None = None,
                   rlog: "ReqLog | None" = None,
                   original_request_payload: dict | None = None) -> Response:
    """Forward request; log status; try chain fallbacks then original model on error."""
    rlog = rlog or ReqLog(_new_req_id())
    msgs = orig_messages or body.get("messages", [])
    # Capture original payload on first entry
    original_request_payload = original_request_payload or copy.deepcopy(body)
    try:
        resp    = await http.post(url, headers=headers, json=body)
        status  = resp.status_code
        content = resp.content

        if status >= 400:
            try:
                err_body = resp.json()
            except Exception:
                err_body = content.decode(errors="replace")[:500]
            rlog.warning(f"{chosen} -> HTTP {status}: {err_body}")
            if status == 400:
                _log_400_detail(body, rlog)
            asyncio.create_task(_debug_log(original, chosen, reason, status, msgs,
                                           req_tools_count, sent_tools_count, err_body))

            # Full debug log on error
            if VERBOSE_FULL_LOGGING:
                asyncio.create_task(_full_debug_log(
                    rlog.req_id,
                    original_request_payload,
                    body,
                    {"status": status, "error": err_body}
                ))

            # Try next model in the chain before falling back to original
            remaining = list(fallback_chain) if fallback_chain else []
            if remaining:
                next_model = remaining.pop(0)
                rlog.info(f"Trying next chain model {next_model} after HTTP {status} on {chosen}")
                body["model"] = next_model
                return await _forward(url, headers, body, next_model, original, user_prompt,
                                      reason + f"[retry:{next_model}]", extra_builtin, extra_mcp,
                                      clf_model, clf_cost, orig_messages,
                                      req_tools_count, sent_tools_count,
                                      fallback_chain=remaining, rlog=rlog,
                                      original_request_payload=original_request_payload)

            if chosen != original:
                rlog.info(f"Falling back to {original} after HTTP {status}")
                fb_body = copy.deepcopy(original_request_payload)
                fb_body["model"] = original
                resp    = await http.post(url, headers=headers, json=fb_body)
                content = _rewrite_model(resp.content, original)
                rlog.info(f"Fallback {original} -> HTTP {resp.status_code}")
                in_t, out_t = _extract_usage(content)
                asyncio.create_task(_record_cost(original, reason + "[fallback]", in_t, out_t, extra_builtin, extra_mcp, clf_model, clf_cost))
                asyncio.create_task(_record(user_prompt, original, reason + "[fallback]", _response_preview(content), resp.status_code))
                asyncio.create_task(_debug_log(original, original, reason + "[fallback]", resp.status_code, msgs,
                                               req_tools_count, sent_tools_count,
                                               None if resp.status_code < 400 else resp.text,
                                               _response_preview(content)))

                if VERBOSE_FULL_LOGGING:
                    try:
                        resp_json = json.loads(content)
                    except:
                        resp_json = {"raw": content.decode(errors="replace")[:1000]}
                    asyncio.create_task(_full_debug_log(
                        rlog.req_id + "-fallback",
                        original_request_payload,
                        body,
                        resp_json
                    ))

                return Response(content=content, status_code=resp.status_code,
                                media_type=resp.headers.get("content-type", "application/json"))

            asyncio.create_task(_record(user_prompt, chosen, reason, "", status))
            return Response(content=content, status_code=status,
                            media_type=resp.headers.get("content-type", "application/json"))

        # Success
        content = _rewrite_model(content, original)
        in_t, out_t = _extract_usage(content)
        preview = _response_preview(content)
        rlog.info(f"{chosen} -> HTTP {status} OK  ({in_t} in / {out_t} out tokens  +{extra_builtin} builtin +{extra_mcp} MCP schema tokens saved)")
        asyncio.create_task(_record_cost(chosen, reason, in_t, out_t, extra_builtin, extra_mcp, clf_model, clf_cost))
        asyncio.create_task(_record(user_prompt, chosen, reason, preview, status))
        asyncio.create_task(_debug_log(original, chosen, reason, status, msgs,
                                       req_tools_count, sent_tools_count, None, preview))

        if VERBOSE_FULL_LOGGING:
            try:
                resp_json = json.loads(content)
            except:
                resp_json = {"raw": content.decode(errors="replace")[:1000]}
            asyncio.create_task(_full_debug_log(
                rlog.req_id,
                original_request_payload,
                body,
                resp_json
            ))

        return Response(content=content, status_code=status,
                        media_type=resp.headers.get("content-type", "application/json"))

    except Exception as e:
        rlog.warning(f"{chosen} failed (exception): {e}")
        asyncio.create_task(_debug_log(original, chosen, reason, 502, msgs,
                                       req_tools_count, sent_tools_count, str(e)))
        if chosen != original:
            rlog.info(f"Falling back to {original}")
            fb_body = copy.deepcopy(original_request_payload)
            fb_body["model"] = original
            try:
                resp    = await http.post(url, headers=headers, json=fb_body)
                content = _rewrite_model(resp.content, original)
                rlog.info(f"Fallback {original} -> HTTP {resp.status_code}")
                in_t, out_t = _extract_usage(content)
                asyncio.create_task(_record_cost(original, reason + "[fallback]", in_t, out_t, extra_builtin, extra_mcp, clf_model, clf_cost))
                asyncio.create_task(_record(user_prompt, original, reason + "[fallback]", _response_preview(content), resp.status_code))
                return Response(content=content, status_code=resp.status_code,
                                media_type=resp.headers.get("content-type", "application/json"))
            except Exception as e2:
                rlog.error(f"Fallback also failed: {e2}")
        asyncio.create_task(_record(user_prompt, chosen, reason, "", 502))
        return Response(
            content=json.dumps({"error": {"message": str(e), "type": "proxy_error"}}),
            status_code=502, media_type="application/json",
        )


# ── Streaming ──────────────────────────────────────────────────────────────────

async def _stream(url: str, headers: dict, body: dict, chosen: str, original: str,
                  user_prompt: str = "", reason: str = "", extra_builtin: int = 0, extra_mcp: int = 0,
                  clf_model: str = "", clf_cost: float = 0.0,
                  orig_messages: list | None = None,
                  req_tools_count: int = 0, sent_tools_count: int = 0,
                  fallback_chain: list | None = None,
                  rlog: "ReqLog | None" = None,
                  original_request_payload: dict | None = None):
    """Stream SSE; capture token usage from events; try chain fallbacks then original on error."""
    rlog  = rlog or ReqLog(_new_req_id())
    usage = {"input": 0, "output": 0}
    msgs  = orig_messages or body.get("messages", [])
    original_request_payload = original_request_payload or copy.deepcopy(body)

    try:
        async with http.stream("POST", url, headers=headers, json=body) as resp:
            status = resp.status_code
            if status >= 400:
                err_content = await resp.aread()
                try:
                    err_body = json.loads(err_content)
                except Exception:
                    err_body = err_content.decode(errors="replace")[:500]
                rlog.warning(f"{chosen} -> HTTP {status}: {err_body}")
                if status == 400:
                    _log_400_detail(body, rlog)
                asyncio.create_task(_debug_log(original, chosen, reason, status, msgs,
                                               req_tools_count, sent_tools_count, err_body))

                if VERBOSE_FULL_LOGGING:
                    asyncio.create_task(_full_debug_log(
                        rlog.req_id,
                        original_request_payload,
                        body,
                        {"status": status, "error": err_body}
                    ))

                # Try next model in chain before falling back to original
                remaining = list(fallback_chain) if fallback_chain else []
                if remaining:
                    next_model = remaining.pop(0)
                    rlog.info(f"Trying next chain model {next_model} after HTTP {status} on {chosen}")
                    body["model"] = next_model
                    async for chunk in _stream(url, headers, body, next_model, original,
                                               user_prompt, reason + f"[retry:{next_model}]",
                                               extra_builtin, extra_mcp, clf_model, clf_cost,
                                               orig_messages, req_tools_count, sent_tools_count,
                                               fallback_chain=remaining, rlog=rlog,
                                               original_request_payload=original_request_payload):
                        yield chunk
                    return

                if chosen != original:
                    rlog.info(f"Falling back to {original} after HTTP {status}")
                    body["model"] = original
                    fallback_usage = {"input": 0, "output": 0}
                    async with http.stream("POST", url, headers=headers, json=body) as resp2:
                        rlog.info(f"Fallback {original} -> HTTP {resp2.status_code} (streaming)")
                        async for chunk in resp2.aiter_bytes():
                            _scan_sse_for_usage(chunk, fallback_usage)
                            yield _rewrite_sse_chunk(chunk, original)
                    asyncio.create_task(_record_cost(original, reason + "[fallback]", fallback_usage["input"], fallback_usage["output"], extra_builtin, extra_mcp, clf_model, clf_cost))
                    asyncio.create_task(_record(user_prompt, original, reason + "[fallback]", "(streaming)", 200))

                    if VERBOSE_FULL_LOGGING:
                        asyncio.create_task(_full_debug_log(
                            rlog.req_id + "-fallback",
                            original_request_payload,
                            body,
                            {"status": 200, "info": "streaming success"}
                        ))
                    return

                err = {"type": "error", "error": {"type": "upstream_error", "message": str(err_body)}}
                yield f"data: {json.dumps(err)}\n\n".encode()
                asyncio.create_task(_record(user_prompt, chosen, reason, "", status))
                return

            rlog.info(f"{chosen} -> HTTP {status} OK (streaming)")
            async for chunk in resp.aiter_bytes():
                _scan_sse_for_usage(chunk, usage)
                yield _rewrite_sse_chunk(chunk, original)

        _in, _out = usage["input"], usage["output"]
        rlog.info(f"{chosen} stream done  ({_in} in / {_out} out tokens  +{extra_builtin} builtin +{extra_mcp} MCP schema tokens saved)")
        asyncio.create_task(_record_cost(chosen, reason, usage["input"], usage["output"], extra_builtin, extra_mcp, clf_model, clf_cost))
        asyncio.create_task(_record(user_prompt, chosen, reason, "(streaming)", 200))
        asyncio.create_task(_debug_log(original, chosen, reason, 200, msgs,
                                       req_tools_count, sent_tools_count, None, "(streaming)"))

        if VERBOSE_FULL_LOGGING:
            asyncio.create_task(_full_debug_log(
                rlog.req_id,
                original_request_payload,
                body,
                {"status": 200, "usage": usage, "info": "streaming success"}
            ))

    except Exception as e:
        rlog.warning(f"Stream failed on {chosen}: {e}")
        if chosen != original:
            rlog.info(f"Retrying stream with original model {original}")
            body["model"] = original
            try:
                fallback_usage = {"input": 0, "output": 0}
                async with http.stream("POST", url, headers=headers, json=body) as resp:
                    rlog.info(f"Fallback {original} -> HTTP {resp.status_code} (streaming)")
                    async for chunk in resp.aiter_bytes():
                        _scan_sse_for_usage(chunk, fallback_usage)
                        yield _rewrite_sse_chunk(chunk, original)
                asyncio.create_task(_record_cost(original, reason + "[fallback]", fallback_usage["input"], fallback_usage["output"], extra_builtin, extra_mcp, clf_model, clf_cost))
                asyncio.create_task(_record(user_prompt, original, reason + "[fallback]", "(streaming)", 200))
                return
            except Exception as e2:
                rlog.error(f"Fallback stream also failed: {e2}")
        asyncio.create_task(_record(user_prompt, chosen, reason, "", 502))
        err = {"type": "error", "error": {"type": "proxy_error", "message": str(e)}}
        yield f"data: {json.dumps(err)}\n\n".encode()


# ── Passthrough ────────────────────────────────────────────────────────────────

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def passthrough(request: Request, path: str):
    url = f"{UPSTREAM}/{path}"
    # Extract the API key from the incoming request (same logic as messages() handler)
    _pt_key  = ""
    _pt_auth = request.headers.get("Authorization", "")
    if _pt_auth.startswith("Bearer "):
        _pt_key = _pt_auth[7:]
    if not _pt_key:
        _pt_key = request.headers.get("X-API-Key", "")
    headers = {"Authorization": f"Bearer {_pt_key}"} if _pt_key else {}
    body    = await request.body() if request.method in ("POST", "PUT") else None
    try:
        resp = await http.request(request.method, url, headers=headers, content=body)
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "application/json"),
        )
    except httpx.ConnectError as e:
        log.warning(f"Passthrough connect error for /{path}: upstream unreachable ({e})")
        return Response(
            content=json.dumps({"error": {"message": "Upstream unreachable", "type": "proxy_connect_error"}}),
            status_code=503, media_type="application/json",
        )
    except Exception as e:
        log.warning(f"Passthrough error for /{path}: {e}")
        return Response(
            content=json.dumps({"error": {"message": str(e), "type": "proxy_error"}}),
            status_code=502, media_type="application/json",
        )


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"[Proxy] Starting on http://localhost:{PROXY_PORT}")
    print(f"[Proxy] Upstream : {UPSTREAM}")
    print(f"[Proxy] Usage    : set ANTHROPIC_BASE_URL=http://localhost:{PROXY_PORT}")
    uvicorn.run(app, host="127.0.0.1", port=PROXY_PORT, log_level="warning")

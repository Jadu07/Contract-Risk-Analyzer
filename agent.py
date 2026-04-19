"""
Module 3: LangGraph Agentic Workflow
------------------------------------
Wires the agent loop that drives end-to-end analysis:

    ┌──────────────┐     ┌───────────────┐
    │  retrieve    │ ──> │   analyze     │ ──┐
    │  (RAG)       │     │   (Groq LLM)  │   │ loop while more clauses
    └──────────────┘     └───────────────┘ ──┘
              ^                │
              └── next clause ─┤    once all done
                               ▼
                        ┌──────────────┐
                        │  summarize   │ ──► END
                        │  (Groq LLM)  │
                        └──────────────┘

  * ``retrieve``  : pulls top-k legal guidelines for the current clause
                    from the Chroma retriever.
  * ``analyze``   : asks Groq's Llama-3 to classify the clause into
                    ``Low | Medium | High`` risk and produce a
                    structured JSON analysis. Appends to the report.
  * ``summarize`` : final node — produces the executive-level contract
                    summary, overall severity, top risks, and
                    recommended actions required by the rubric.

State is explicit (TypedDict). Loop termination is handled by a
conditional edge from ``analyze``.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.vectorstores import VectorStoreRetriever
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

logger = logging.getLogger(__name__)

# Llama 3.3 70B on Groq — strong reasoning, generous free tier, fast.
GROQ_MODEL = "llama-3.3-70b-versatile"

# The word "JSON" MUST appear in the prompt whenever we enable Groq's
# JSON response_format, or the API will reject the request.
ANALYZER_SYSTEM_PROMPT = """\
You are a senior contracts lawyer reviewing ONE clause at a time. Be
conservative: if a clause is ambiguous or missing standard protections,
flag it as risky rather than waving it through.

You will receive:
  1. The exact text of ONE contract clause.
  2. Retrieved reference guidelines from a legal knowledge base (may be
     empty or only partially relevant).

You MUST respond with ONLY a single JSON object that has EXACTLY these
three keys:
  - "Risk_Level"  : one of "Low", "Medium", "High".
  - "Explanation" : 2-3 sentences explaining WHY this clause carries
                    that risk level, grounded in the retrieved
                    guidelines. If the guidelines are not relevant,
                    say so explicitly — do not fabricate citations.
  - "Mitigation"  : one concrete redline or negotiation suggestion.
                    Use the string "None" if the clause is already
                    acceptable.

CRITICAL JSON RULES — violating any of these makes your output invalid:
  1. EVERY string value MUST be wrapped in double quotes ("..."). This
     applies to "Explanation" and "Mitigation" as well.
  2. Any literal double-quote inside a string value MUST be escaped as \".
  3. Do not use line breaks inside string values; keep each value on one
     logical line.
  4. Do not include trailing commas.
  5. Do not include any text, comments, or markdown outside the JSON.

Example of the EXACT shape required:
{"Risk_Level": "High", "Explanation": "The clause caps liability at $0, leaving the vendor exposed to unlimited damages.", "Mitigation": "Cap liability at 12 months of fees paid."}
"""

SUMMARIZER_SYSTEM_PROMPT = """\
You are a senior contracts lawyer writing an EXECUTIVE SUMMARY of a
contract risk review. You will be given a JSON array of per-clause
analyses (each with Risk_Level, Explanation, Mitigation, and the
original clause_text).

You MUST respond with ONLY a single JSON object containing EXACTLY
these keys:
  - "contract_overview"     : 2-3 sentences describing what the
                              contract appears to cover, the parties
                              (if inferable), and the primary
                              obligations. Do not fabricate party
                              names — say "Party A / Party B" if
                              unknown.
  - "overall_risk_severity" : one of "Low", "Medium", "High" — the
                              worst-case assessment across all clauses.
  - "top_risks"             : array of up to 5 objects, each with keys
                              "clause_number" (int) and
                              "risk_description" (string). Ordered
                              most-severe first.
  - "recommended_actions"   : array of 3-5 concrete negotiation or
                              redline suggestions at the CONTRACT
                              level (not per-clause).
  - "disclaimer"            : a one-sentence legal/ethical disclaimer
                              stating this is not legal advice.

Do not include any other keys. Do not wrap the JSON in markdown.
"""


# --------------------------------------------------------------------------
# Graph state
# --------------------------------------------------------------------------
class ContractState(TypedDict):
    """State object threaded through every node in the LangGraph.

    Attributes
    ----------
    clauses:
        List of ``{"clause_number": int, "text": str}`` from ingestion.
    current_index:
        Zero-based pointer to the clause being processed this turn.
    retrieved_context:
        Top-k retrieved guideline text for the current clause.
    structured_report:
        Accumulated per-clause analysis results (grows each loop).
    contract_summary:
        Final executive summary dict, populated by the summarize node.
    errors:
        Non-fatal error notes collected along the way. Optional.
    """

    clauses: List[Dict[str, Any]]
    current_index: int
    retrieved_context: str
    structured_report: List[Dict[str, Any]]
    contract_summary: Optional[Dict[str, Any]]
    errors: Optional[str]


# --------------------------------------------------------------------------
# Graph builder
# --------------------------------------------------------------------------
def build_agent(
    retriever: VectorStoreRetriever,
    groq_api_key: str,
    model_name: str = GROQ_MODEL,
):
    """Construct and compile the LangGraph workflow.

    Parameters
    ----------
    retriever:
        A LangChain retriever (see retriever.build_retriever).
    groq_api_key:
        Free-tier Groq API key supplied by the user.
    model_name:
        Groq-hosted model id. Defaults to Llama 3.3 70B.

    Returns
    -------
    Compiled LangGraph application. Invoke with ``app.invoke(state)``.
    """
    # Temperature 0 for reproducible legal analysis. JSON mode ensures
    # the output is directly parseable.
    llm = ChatGroq(
        api_key=groq_api_key,
        model=model_name,
        temperature=0.0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )

    # ---- Node 1: retrieve ------------------------------------------------
    def retrieve_node(state: ContractState) -> Dict[str, Any]:
        """Pull top-k guideline chunks for the current clause."""
        idx = state["current_index"]
        clause = state["clauses"][idx]
        try:
            docs = retriever.invoke(clause["text"])
            context = "\n\n".join(d.page_content for d in docs) if docs else ""
            return {"retrieved_context": context}
        except Exception as exc:  # fail-soft: analyzer can still run
            logger.warning("Retriever failed on clause %d: %s", idx, exc)
            return {
                "retrieved_context": "",
                "errors": f"retriever_error@{idx}: {exc}",
            }

    # ---- Node 2: analyze -------------------------------------------------
    def analyze_node(state: ContractState) -> Dict[str, Any]:
        """Ask the Groq LLM to score + explain the current clause."""
        idx = state["current_index"]
        clause = state["clauses"][idx]
        context = state.get("retrieved_context", "") or "No relevant guidelines found."

        user_msg = (
            f"CLAUSE #{clause['clause_number']}:\n"
            f"\"\"\"\n{clause['text']}\n\"\"\"\n\n"
            f"RETRIEVED GUIDELINES:\n"
            f"\"\"\"\n{context}\n\"\"\""
        )

        analysis = _invoke_with_json_repair(
            llm,
            system_prompt=ANALYZER_SYSTEM_PROMPT,
            user_msg=user_msg,
            label=f"analyzer@clause_{clause['clause_number']}",
        ) or {
            "Risk_Level": "Unknown",
            "Explanation": "Automated analysis failed after retries.",
            "Mitigation": "Manual review required.",
        }

        report_entry: Dict[str, Any] = {
            "clause_number": clause["clause_number"],
            "clause_text": clause["text"],
            "Risk_Level": analysis.get("Risk_Level", "Unknown"),
            "Explanation": analysis.get("Explanation", ""),
            "Mitigation": analysis.get("Mitigation", ""),
            "retrieved_context": context,
        }

        return {
            "structured_report": state["structured_report"] + [report_entry],
            "current_index": idx + 1,
        }

    # ---- Node 3: summarize (runs once, after the loop) ------------------
    def summarize_node(state: ContractState) -> Dict[str, Any]:
        """Generate the executive contract-level summary."""
        # Compact the per-clause report so we don't overflow the context.
        compact = [
            {
                "clause_number": e["clause_number"],
                "Risk_Level": e.get("Risk_Level", "Unknown"),
                "Explanation": e.get("Explanation", ""),
                "Mitigation": e.get("Mitigation", ""),
                # Truncate the original clause text to keep prompt tight.
                "clause_excerpt": (e.get("clause_text", "") or "")[:500],
            }
            for e in state["structured_report"]
        ]

        user_msg = (
            "Here is the per-clause analysis as JSON. "
            "Write the executive summary now.\n\n"
            f"{json.dumps(compact, ensure_ascii=False)}"
        )
        summary = _invoke_with_json_repair(
            llm,
            system_prompt=SUMMARIZER_SYSTEM_PROMPT,
            user_msg=user_msg,
            label="summarizer",
        )
        if not summary:
            summary = _fallback_summary(
                state["structured_report"],
                error="LLM returned invalid JSON after retries",
            )

        # Always ensure a disclaimer is present even if the LLM omitted it.
        summary.setdefault(
            "disclaimer",
            "This AI-generated assessment is informational only and does "
            "not constitute legal advice. Consult a qualified attorney "
            "before acting on its findings.",
        )

        return {"contract_summary": summary}

    # ---- Conditional routing --------------------------------------------
    def route_after_analyze(state: ContractState) -> str:
        """Loop back to retrieve, or advance to summarize."""
        if state["current_index"] < len(state["clauses"]):
            return "retrieve"
        return "summarize"

    # ---- Graph assembly --------------------------------------------------
    graph = StateGraph(ContractState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("summarize", summarize_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "analyze")
    graph.add_conditional_edges(
        "analyze",
        route_after_analyze,
        {"retrieve": "retrieve", "summarize": "summarize"},
    )
    graph.add_edge("summarize", END)

    return graph.compile()


# --------------------------------------------------------------------------
# Convenience runner
# --------------------------------------------------------------------------
def run_agent(
    app,
    clauses: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Invoke the compiled graph on a list of clauses.

    Returns a dict with:
      - ``structured_report``: list of per-clause analyses.
      - ``contract_summary`` : the executive summary object.
    """
    initial_state: ContractState = {
        "clauses": clauses,
        "current_index": 0,
        "retrieved_context": "",
        "structured_report": [],
        "contract_summary": None,
        "errors": None,
    }

    # LangGraph's default recursion_limit (25) is too low for long
    # contracts. Each clause consumes ~2 steps (retrieve + analyze),
    # plus one final summarize, so budget generously.
    recursion_limit = max(50, len(clauses) * 3 + 10)
    final_state = app.invoke(initial_state, config={"recursion_limit": recursion_limit})

    return {
        "structured_report": final_state["structured_report"],
        "contract_summary": final_state.get("contract_summary"),
    }


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _invoke_with_json_repair(
    llm,
    *,
    system_prompt: str,
    user_msg: str,
    label: str,
    max_retries: int = 2,
) -> Optional[Dict[str, Any]]:
    """Invoke the LLM and guarantee a dict, repairing Groq's common mistakes.

    Groq's JSON mode (``response_format={"type": "json_object"}``) occasionally
    rejects Llama-3's output with ``json_validate_failed`` when the model
    forgets to quote a string value. The malformed payload is still returned
    in the error body under ``failed_generation`` — we extract it, try to
    repair it, and fall back to a stricter retry.
    """
    last_err: Optional[str] = None
    strictness_reminder = (
        "\n\nREMINDER: return ONE JSON object. Every string value MUST be "
        "wrapped in double quotes. No line breaks inside values. No prose "
        "outside the JSON."
    )

    for attempt in range(max_retries + 1):
        msg = user_msg if attempt == 0 else user_msg + strictness_reminder
        try:
            raw = llm.invoke(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": msg},
                ]
            ).content
            if isinstance(raw, str):
                parsed = _parse_or_repair(raw)
                if parsed is not None:
                    return parsed
            last_err = "empty or non-string LLM response"
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
            failed = _extract_failed_generation(exc)
            if failed:
                parsed = _parse_or_repair(failed)
                if parsed is not None:
                    logger.info("%s: recovered from json_validate_failed on attempt %d", label, attempt + 1)
                    return parsed
        logger.warning("%s: attempt %d failed (%s)", label, attempt + 1, last_err)

    return None


def _parse_or_repair(raw: str) -> Optional[Dict[str, Any]]:
    """Try strict json.loads, then a regex-based repair, then give up."""
    cleaned = raw.strip()
    # Strip ```json fences the model sometimes sneaks in despite JSON mode.
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").lstrip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].lstrip()

    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass

    repaired = _repair_unquoted_string_values(cleaned)
    try:
        obj = json.loads(repaired)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


# Keys whose values should always be strings — Llama-3 on Groq drops the
# outer quotes on these most often. We only repair at known keys to avoid
# corrupting legitimately structured values (arrays, ints, nested objects).
_STRING_VALUED_KEYS = (
    "Risk_Level",
    "Explanation",
    "Mitigation",
    "contract_overview",
    "overall_risk_severity",
    "risk_description",
    "disclaimer",
)

_KV_RE = re.compile(
    r'("(?:' + "|".join(_STRING_VALUED_KEYS) + r')"\s*:\s*)'
    r'([^\n\r]+?)'
    # Stop at: optional-comma + newline + next "key": | end-of-object.
    r'(?=(?:\s*,)?\s*(?:\r?\n)+\s*"[A-Za-z_][A-Za-z0-9_]*"\s*:|\s*(?:\r?\n)*\s*\}\s*$)',
    re.DOTALL,
)

# Inserts a missing comma when a closing quote is followed by whitespace
# and another "key": — Groq sometimes drops the comma along with the
# outer quotes.
_MISSING_COMMA_RE = re.compile(r'(")(\s*(?:\r?\n)\s*)(")([A-Za-z_][A-Za-z0-9_]*"\s*:)')


def _repair_unquoted_string_values(raw: str) -> str:
    """Wrap unquoted string values in double quotes for known keys.

    Converts:  "Explanation": The clause is risky,
    Into:      "Explanation": "The clause is risky",

    Values that are already valid JSON (quoted strings, arrays, numbers,
    etc.) are left untouched — the callback inspects each captured value
    and only rewrites it if `json.loads` rejects it as-is.
    """
    def repl(m: "re.Match[str]") -> str:
        key_prefix, value = m.group(1), m.group(2)
        stripped = value.strip().rstrip(",").strip()
        if not stripped:
            return m.group(0)
        # If the captured value is already a valid JSON literal
        # (quoted string, array, object, number, bool, null), keep it.
        try:
            json.loads(stripped)
            return m.group(0)
        except json.JSONDecodeError:
            pass
        # Otherwise wrap the raw text as a JSON string.
        escaped = stripped.replace("\\", "\\\\").replace('"', '\\"')
        return f'{key_prefix}"{escaped}"'

    out = _KV_RE.sub(repl, raw)
    # Second pass: insert commas missing between adjacent string-valued
    # key/value pairs (e.g. '"..."\n    "NextKey": ...').
    out = _MISSING_COMMA_RE.sub(r'\1,\2\3\4', out)
    return out


def _extract_failed_generation(exc: Exception) -> Optional[str]:
    """Pull the ``failed_generation`` payload out of a Groq BadRequestError."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error") or {}
        if isinstance(err, dict):
            val = err.get("failed_generation")
            if isinstance(val, str) and val.strip():
                return val
    # Fallback: scrape it from the stringified exception.
    text = str(exc)
    marker = "'failed_generation': '"
    i = text.find(marker)
    if i != -1:
        j = text.find("'}", i + len(marker))
        if j != -1:
            # Decode the single-quoted python-repr payload back to a raw string.
            snippet = text[i + len(marker):j].encode("utf-8").decode("unicode_escape", errors="replace")
            return snippet
    return None


def _fallback_summary(
    report: List[Dict[str, Any]],
    error: str = "",
) -> Dict[str, Any]:
    """Deterministic backup summary when the LLM summarizer fails.

    Required by the rubric: the app must handle LLM/retrieval failures
    gracefully, not crash the user flow.
    """
    counts = {"High": 0, "Medium": 0, "Low": 0, "Unknown": 0}
    for e in report:
        counts[e.get("Risk_Level", "Unknown")] = counts.get(e.get("Risk_Level", "Unknown"), 0) + 1

    if counts["High"]:
        severity = "High"
    elif counts["Medium"]:
        severity = "Medium"
    elif counts["Low"]:
        severity = "Low"
    else:
        severity = "Unknown"

    top = sorted(
        [e for e in report if e.get("Risk_Level") in {"High", "Medium"}],
        key=lambda e: 0 if e.get("Risk_Level") == "High" else 1,
    )[:5]

    return {
        "contract_overview": (
            "Automated summary unavailable — showing deterministic fallback. "
            f"Analyzed {len(report)} clause(s)."
            + (f" (LLM error: {error})" if error else "")
        ),
        "overall_risk_severity": severity,
        "top_risks": [
            {
                "clause_number": e["clause_number"],
                "risk_description": (e.get("Explanation") or "")[:200],
            }
            for e in top
        ],
        "recommended_actions": [
            "Review each High and Medium risk clause with legal counsel.",
            "Request redlines for any uncapped liability or indemnity clauses.",
            "Confirm governing law and dispute-resolution terms are acceptable.",
        ],
    }

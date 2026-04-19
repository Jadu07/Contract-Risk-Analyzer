"""
Module 4: Streamlit Interface
-----------------------------
Ties the back-end modules together into a public-facing UI:

    user PDF  ──►  ingestion.py (Gemini 2.5 Flash)
                        │
                        ▼
                   list[Clause]  ──►  agent.py (Groq + Chroma RAG)
                                             │
                                             ▼
                                {summary, per-clause report}
                                             │
                        ┌────────────────────┴────────────────────┐
                        ▼                                         ▼
                  Streamlit view                         pdf_export.py
"""

from __future__ import annotations

import base64
import json
import os
import tempfile
from typing import Any, Dict, List

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from agent import build_agent, run_agent
from ingestion import ingest_and_segment_contract
from pdf_export import build_pdf_report
from pdf_highlighter import highlight_risky_clauses
from retriever import build_retriever

# Load GEMINI_API_KEY / GROQ_API_KEY from a local .env file, if present.
# Real env vars always take precedence (so production hosting can inject
# them without a .env file on disk).
load_dotenv(override=False)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

# Risk levels we actually show in the UI and highlight in the PDF.
# "Low" clauses are clean findings and "Unknown" clauses are analyzer
# failures — both add noise for reviewers, so we surface only the real
# findings here. Counts for Low / Unknown still appear in the metric
# cards so nothing is silently dropped.
RISKY_LEVELS = {"High", "Medium"}

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Agentic Contract Risk Analyzer",
    page_icon="📄",
    layout="wide",
)

RISK_ICON = {"High": "🔴", "Medium": "🟠", "Low": "🟢", "Unknown": "⚪"}
RISK_ORDER = {"High": 0, "Medium": 1, "Low": 2, "Unknown": 3}


# --------------------------------------------------------------------------
# Cached retriever — embeddings are heavy, so only build once per process.
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embeddings & vector store...")
def get_retriever():
    """Build (or load) the Chroma retriever. Cached across reruns."""
    return build_retriever()


# --------------------------------------------------------------------------
# Core pipeline driver
# --------------------------------------------------------------------------
def _run_pipeline(pdf_bytes: bytes) -> Dict[str, Any]:
    """Write the PDF to a temp file, then run ingestion → agent.

    Returns a dict with keys ``structured_report`` and ``contract_summary``.
    API keys are read from process env (see ``GEMINI_API_KEY`` /
    ``GROQ_API_KEY`` at module load time) — never from the UI.
    """
    # Gemini's SDK expects a filesystem path, so we stage the upload.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        clauses = ingest_and_segment_contract(tmp_path, GEMINI_API_KEY)
    finally:
        # Best-effort local cleanup; OS will also clean /tmp eventually.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    retriever = get_retriever()
    compiled_graph = build_agent(retriever, GROQ_API_KEY)
    return run_agent(compiled_graph, clauses)


# --------------------------------------------------------------------------
# UI helpers
# --------------------------------------------------------------------------
def _render_summary_cards(report: List[Dict[str, Any]]) -> None:
    """Top-of-page numeric cards + alert banner."""
    counts = {"High": 0, "Medium": 0, "Low": 0, "Unknown": 0}
    for entry in report:
        level = entry.get("Risk_Level", "Unknown")
        counts[level] = counts.get(level, 0) + 1

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Clauses", len(report))
    c2.metric("🔴 High", counts["High"])
    c3.metric("🟠 Medium", counts["Medium"])
    c4.metric("🟢 Low", counts["Low"])
    c5.metric("⚪ Unknown", counts["Unknown"])

    if counts["High"] > 0:
        st.error(f"⚠️  {counts['High']} high-risk clause(s) detected — review carefully.")
    elif counts["Medium"] > 0:
        st.warning(f"{counts['Medium']} medium-risk clause(s) detected.")
    else:
        st.success("No high- or medium-risk clauses detected.")


def _render_executive_summary(summary: Dict[str, Any]) -> None:
    """Render the contract-level executive summary produced by the agent."""
    if not summary:
        return

    severity = summary.get("overall_risk_severity", "Unknown")
    icon = RISK_ICON.get(severity, "⚪")

    st.markdown(f"### {icon} Overall Risk Severity: **{severity}**")

    overview = summary.get("contract_overview")
    if overview:
        st.markdown("**Contract Overview**")
        st.write(overview)

    top_risks = summary.get("top_risks") or []
    if top_risks:
        st.markdown("**Top Risks**")
        for r in top_risks:
            cn = r.get("clause_number", "?")
            desc = r.get("risk_description", "")
            st.markdown(f"- **Clause {cn}** — {desc}")

    actions = summary.get("recommended_actions") or []
    if actions:
        st.markdown("**Recommended Actions**")
        for a in actions:
            st.markdown(f"- {a}")

    disclaimer = summary.get("disclaimer")
    if disclaimer:
        st.caption(f"⚖️  {disclaimer}")


def _embed_pdf(pdf_bytes: bytes, height: int = 900) -> None:
    """Render PDF bytes inline inside the Streamlit page using direct DOM injection."""
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    
    # Using <embed> or <object> often works better for PDFs than <iframe>
    pdf_display = (
        f'<embed src="data:application/pdf;base64,{b64}" '
        f'width="100%" height="{height}" '
        f'style="border:1px solid #444;border-radius:6px;" '
        f'type="application/pdf">'
    )
    
    # Inject directly into the Streamlit DOM to avoid the components.html sandbox
    st.markdown(pdf_display, unsafe_allow_html=True)


def _render_clause_details(report: List[Dict[str, Any]]) -> None:
    """Render only High / Medium risk clauses — these are real findings.

    Low-risk and Unknown-risk clauses are deliberately omitted from this
    list to keep the reviewer focused on actionable items. Counts for
    both still appear in the top-of-page metric cards.
    """
    risky = [e for e in report if e.get("Risk_Level") in RISKY_LEVELS]

    if not risky:
        st.success("✅ No high or medium risk clauses detected.")
        return

    sorted_risky = sorted(
        risky,
        key=lambda e: (
            RISK_ORDER.get(e.get("Risk_Level", "Unknown"), 99),
            e.get("clause_number", 0),
        ),
    )
    for entry in sorted_risky:
        level = entry.get("Risk_Level", "Unknown")
        icon = RISK_ICON.get(level, "⚪")
        header = f"{icon} Clause {entry['clause_number']} — {level} Risk"
        with st.expander(header, expanded=(level == "High")):
            st.markdown("**Clause Text**")
            st.write(entry.get("clause_text", ""))

            st.markdown("**Explanation**")
            st.write(entry.get("Explanation") or "—")

            st.markdown("**Suggested Mitigation**")
            st.write(entry.get("Mitigation") or "—")

            if entry.get("retrieved_context"):
                with st.popover("📚 Retrieved legal context"):
                    st.caption(entry["retrieved_context"])


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    st.title("📄 Agentic Contract Risk Analyzer")
    st.caption(
        "Gemini 2.5 Flash segments the PDF · "
        "Llama-3 (Groq) reasons about each clause · "
        "Chroma + MiniLM provides legal context"
    )

    # ---- Sidebar: upload only (keys live in .env on the backend) --------
    with st.sidebar:
        st.header("📤  Upload")
        pdf_file = st.file_uploader("Contract (PDF)", type=["pdf"])
        st.markdown("---")
        st.caption(
            "⚖️  **Disclaimer**: This tool is for educational and "
            "informational use only. It does **not** constitute legal "
            "advice. Always consult a qualified attorney before acting "
            "on any of these findings."
        )

    # ---- Gate: API keys must be configured on the backend ---------------
    missing_keys = [
        name
        for name, value in (
            ("GEMINI_API_KEY", GEMINI_API_KEY),
            ("GROQ_API_KEY", GROQ_API_KEY),
        )
        if not value
    ]
    if missing_keys:
        st.error(
            "Missing backend configuration: "
            + ", ".join(missing_keys)
            + ". Set these in a local `.env` file or the hosting "
            "environment before running the app."
        )
        return

    # ---- Main panel -----------------------------------------------------
    if not pdf_file:
        st.info("👈 Upload a contract PDF to begin.")
        return

    analyze_clicked = st.button(
        "🔍 Analyze Contract",
        type="primary",
        use_container_width=True,
    )

    if analyze_clicked:
        # Read once, cache bytes so the highlighter can reuse them later.
        pdf_bytes = pdf_file.read()
        try:
            with st.spinner("🧠 Segmenting with Gemini → reasoning with Llama-3..."):
                result = _run_pipeline(pdf_bytes)
            st.session_state["result"] = result
            st.session_state["original_pdf_bytes"] = pdf_bytes
            st.success(
                f"Analyzed {len(result['structured_report'])} clause(s)."
            )
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")
            return

    # Render whatever is in session (so expanders survive reruns).
    result = st.session_state.get("result")
    if not result:
        return

    report: List[Dict[str, Any]] = result.get("structured_report", [])
    summary: Dict[str, Any] = result.get("contract_summary") or {}

    st.markdown("## 📊 Risk Summary")
    _render_summary_cards(report)

    st.markdown("## 📝 Executive Summary")
    _render_executive_summary(summary)

    st.markdown("## 🚩 Risky Clauses")
    _render_clause_details(report)

    # Only the risky subset goes into the generated PDF report, matching
    # what the reviewer sees on-screen.
    risky_report = [e for e in report if e.get("Risk_Level") in RISKY_LEVELS]

    # ---- Highlighted contract preview (rendered inline) ----------------
    st.markdown("## 🖍️ Highlighted Contract")
    original_bytes = st.session_state.get("original_pdf_bytes")
    if not original_bytes:
        st.info("Upload and re-analyze a contract to see the highlighted preview.")
    elif not risky_report:
        st.info("No high or medium risk clauses to highlight.")
    else:
        try:
            highlighted = highlight_risky_clauses(original_bytes, risky_report)
            _embed_pdf(highlighted)
        except Exception as exc:
            st.error(f"Highlighting failed: {exc}")

    # ---- Export ---------------------------------------------------------
    st.markdown("## ⬇️ Export")
    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            "Download JSON report",
            data=json.dumps(result, indent=2, ensure_ascii=False),
            file_name="contract_risk_report.json",
            mime="application/json",
            use_container_width=True,
            help="Full analysis including low-risk clauses.",
        )

    with col2:
        try:
            report_pdf = build_pdf_report(risky_report, summary)
            st.download_button(
                "📄 Risk report PDF",
                data=report_pdf,
                file_name="contract_risk_report.pdf",
                mime="application/pdf",
                use_container_width=True,
                help="Generated analysis (risky clauses only).",
            )
        except Exception as exc:
            # PDF should never block the rest of the UI.
            st.error(f"Risk report PDF failed: {exc}")


if __name__ == "__main__":
    main()

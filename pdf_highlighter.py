"""
Module 6: Original-PDF Highlighter (Milestone 2 extension)
----------------------------------------------------------
Takes the user's uploaded contract PDF and overlays colored highlight
annotations on every clause the agent flagged as risky. Returns a new
PDF byte string suitable for ``st.download_button``.

Implementation notes:
  * Uses PyMuPDF (``pymupdf``) because fpdf2 cannot round-trip an
    existing PDF — only PyMuPDF exposes text-search + annotation APIs.
  * Clause text returned by Gemini is verbatim, but PDFs line-wrap
    internally, so a direct ``search_for(full_clause_text)`` almost
    always misses. We therefore split each clause into short snippets
    (sentences, then fixed-width chunks) and highlight whatever matches.
  * Short snippets (< 15 chars) are skipped to avoid highlighting
    common words like "and" or "the".
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import pymupdf  # PyMuPDF; the import was previously spelled "fitz".

logger = logging.getLogger(__name__)

# Light pastel fills (RGB in [0..1]) so the underlying text stays readable.
_HIGHLIGHT_COLORS: Dict[str, Tuple[float, float, float]] = {
    "High":    (1.00, 0.70, 0.70),  # light red
    "Medium":  (1.00, 0.86, 0.60),  # light amber
    "Unknown": (0.95, 0.95, 0.55),  # light yellow
}

_MIN_SNIPPET_CHARS = 15
_MAX_SNIPPET_CHARS = 80
_MAX_SNIPPETS_PER_CLAUSE = 8


def highlight_risky_clauses(
    original_pdf_bytes: bytes,
    report: List[Dict[str, Any]],
    levels_to_highlight: Sequence[str] = ("High", "Medium"),
) -> bytes:
    """Return the uploaded PDF with risky clauses highlighted.

    Parameters
    ----------
    original_pdf_bytes:
        Raw bytes of the user's uploaded PDF.
    report:
        Per-clause analysis list produced by ``run_agent``.
    levels_to_highlight:
        Risk levels to annotate. Low-risk clauses are excluded by
        default — they are the "all clear" signal, not findings.

    Returns
    -------
    bytes
        The annotated PDF. If the source cannot be parsed, the original
        bytes are returned unchanged so the download never fails silently.
    """
    try:
        doc = pymupdf.open(stream=original_pdf_bytes, filetype="pdf")
    except Exception as exc:  # noqa: BLE001 — broad because pymupdf raises many types
        logger.warning("Could not open PDF for highlighting: %s", exc)
        return original_pdf_bytes

    try:
        risky = [e for e in report if e.get("Risk_Level") in levels_to_highlight]
        for entry in risky:
            level = entry.get("Risk_Level", "Unknown")
            color = _HIGHLIGHT_COLORS.get(level, _HIGHLIGHT_COLORS["Unknown"])
            clause_text = entry.get("clause_text", "") or ""
            annotation_title = (
                f"Clause {entry.get('clause_number', '?')} — {level} Risk"
            )
            annotation_body = (entry.get("Explanation") or "")[:400]

            _highlight_clause(
                doc,
                snippets=_clause_snippets(clause_text),
                color=color,
                title=annotation_title,
                body=annotation_body,
            )

        # ``incremental=False`` rewrites the whole file, which is needed
        # because we are mutating pages (adding annotations).
        return doc.tobytes()
    finally:
        doc.close()


# --------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------
def _highlight_clause(
    doc: "pymupdf.Document",
    *,
    snippets: Iterable[str],
    color: Tuple[float, float, float],
    title: str,
    body: str,
) -> None:
    """Search each page for each snippet and add a highlight annotation."""
    for snippet in snippets:
        if len(snippet) < _MIN_SNIPPET_CHARS:
            continue
        found_any = False
        for page in doc:
            # quads=True returns per-line quadrilaterals so multi-line
            # matches still highlight correctly.
            try:
                quads = page.search_for(snippet, quads=True)
            except Exception as exc:  # noqa: BLE001
                logger.debug("search_for failed on a page: %s", exc)
                continue
            if not quads:
                continue
            found_any = True
            annot = page.add_highlight_annot(quads)
            annot.set_colors(stroke=color)
            annot.set_info(title=title, content=body)
            annot.update()
        if found_any:
            # First hit per snippet is enough; no need to keep searching
            # the same snippet on other pages for the same clause.
            continue


def _clause_snippets(text: str) -> List[str]:
    """Break a clause into short, searchable chunks.

    Gemini preserves wording verbatim, but PDF text often has line
    breaks inside sentences. We therefore flatten whitespace and split
    into sentence-ish pieces, further chunking long sentences so each
    snippet fits on one PDF line where possible.
    """
    if not text:
        return []

    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    pieces: List[str] = []
    # Split at sentence terminators while keeping chunks meaningful.
    for sentence in re.split(r"(?<=[.;:!?])\s+", normalized):
        sentence = sentence.strip()
        if not sentence:
            continue
        while len(sentence) > _MAX_SNIPPET_CHARS:
            # Prefer breaking on a word boundary.
            cut = sentence.rfind(" ", 0, _MAX_SNIPPET_CHARS)
            if cut < _MIN_SNIPPET_CHARS:
                cut = _MAX_SNIPPET_CHARS
            pieces.append(sentence[:cut])
            sentence = sentence[cut:].lstrip()
        if sentence:
            pieces.append(sentence)

    # Cap to avoid flooding the PDF with annotations on long clauses.
    return pieces[:_MAX_SNIPPETS_PER_CLAUSE]

"""
Module 1: Multimodal Ingestion
------------------------------
Uses Gemini 2.5 Flash (via the modern ``google-genai`` SDK) STRICTLY as
a document segmenter: reads the uploaded PDF and returns a JSON array
of clause objects.

Design constraints enforced here:
  * Gemini is NEVER used for legal reasoning or risk assessment — only
    for OCR + structural segmentation. Legal analysis happens in
    agent.py via Groq (Llama-3).
  * The uploaded file is always deleted from Gemini's servers after
    processing, regardless of success or failure.
  * Output is a well-typed list of dicts that downstream modules can
    consume directly.

SDK note:
  We use ``google-genai`` (new, unified SDK). The older
  ``google-generativeai`` package has been retired and no longer
  receives updates.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# Narrow prompt: segment only, do not opine. Defense-in-depth against
# the ingestion model accidentally offering "legal advice".
SEGMENTER_PROMPT = """\
You are a contract document SEGMENTER. You must NOT provide legal
advice, opinions, risk assessments, summaries, or commentary.

Task: read the attached PDF contract and split its body into logical
clauses / sections.

Return ONLY a valid JSON array. Each element MUST be an object with:
  - "clause_number": integer (1-indexed, in document order)
  - "text": string (verbatim text of the clause, no edits)

Rules:
  1. Preserve the original wording exactly. Do not summarize, paraphrase,
     translate, or "clean up" the text.
  2. A clause may be a numbered section (e.g. "3.1 Termination"), a
     labeled paragraph, or a self-contained block of text.
  3. Skip page headers, page footers, page numbers, and signature
     blocks. They are not clauses.
  4. Output MUST be valid JSON. No markdown fences. No prose before or
     after the array.
"""


def ingest_and_segment_contract(
    pdf_path: str,
    gemini_api_key: str,
    model_name: str = "gemini-2.5-flash",
) -> List[Dict[str, Any]]:
    """Upload a PDF to Gemini, segment it into clauses, and clean up.

    Parameters
    ----------
    pdf_path:
        Path to a local ``.pdf`` file. Must already exist on disk.
    gemini_api_key:
        A free-tier Google AI Studio API key supplied by the user.
    model_name:
        Gemini model id. Defaults to ``gemini-2.5-flash`` for speed.

    Returns
    -------
    List[Dict[str, Any]]
        A list of ``{"clause_number": int, "text": str}`` dicts in
        document order.

    Raises
    ------
    RuntimeError
        If the upload or the Gemini generation call fails.
    ValueError
        If Gemini's output cannot be parsed as a non-empty JSON array.
    """
    client = genai.Client(api_key=gemini_api_key)

    # --- 1. Upload the PDF to Gemini ---------------------------------------
    # MIME type is auto-detected from the .pdf extension by the SDK.
    try:
        uploaded = client.files.upload(file=pdf_path)
    except Exception as exc:  # network, auth, quota, etc.
        raise RuntimeError(f"Failed to upload PDF to Gemini: {exc}") from exc

    uploaded_name = uploaded.name  # remember for cleanup even if refresh fails

    try:
        # Gemini files briefly sit in PROCESSING before becoming ACTIVE.
        # Poll with a short timeout so we don't hang the UI forever.
        for _ in range(15):
            info = client.files.get(name=uploaded_name)
            state_name = info.state.name if info.state else "UNKNOWN"
            if state_name == "ACTIVE":
                uploaded = info
                break
            if state_name == "FAILED":
                raise RuntimeError("Gemini reported file state FAILED.")
            time.sleep(1)
        else:
            raise RuntimeError("Timed out waiting for Gemini file to become ACTIVE.")

        # --- 2. Ask Gemini to segment (JSON mode) --------------------------
        response = client.models.generate_content(
            model=model_name,
            contents=[SEGMENTER_PROMPT, uploaded],
            config=types.GenerateContentConfig(
                # Force structured JSON output — no markdown wrapping.
                response_mime_type="application/json",
                # Deterministic segmentation.
                temperature=0.0,
            ),
        )
        raw_text: str = response.text or ""
    finally:
        # --- 3. Always clean up the remote file ----------------------------
        _safe_delete_gemini_file(client, uploaded_name)

    # --- 4. Parse the JSON defensively -----------------------------------
    cleaned = raw_text.strip()
    # Defensive strip of any accidental ```json fences.
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").lstrip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].lstrip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Gemini did not return valid JSON. First 500 chars:\n"
            f"{raw_text[:500]}"
        ) from exc

    if not isinstance(parsed, list):
        raise ValueError("Expected a top-level JSON array of clauses.")

    # Normalize every element to a trusted shape.
    normalized: List[Dict[str, Any]] = []
    for i, item in enumerate(parsed, start=1):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        try:
            clause_number = int(item.get("clause_number", i))
        except (TypeError, ValueError):
            clause_number = i
        normalized.append({"clause_number": clause_number, "text": text})

    if not normalized:
        raise ValueError("Gemini returned no parseable clauses.")

    logger.info("Segmented contract into %d clauses.", len(normalized))
    return normalized


def _safe_delete_gemini_file(client: "genai.Client", file_name: str) -> None:
    """Best-effort delete of a Gemini-hosted file; swallows errors."""
    try:
        client.files.delete(name=file_name)
        logger.debug("Deleted Gemini file %s", file_name)
    except Exception as exc:  # noqa: BLE001 — cleanup must never raise
        logger.warning("Failed to delete Gemini file %s: %s", file_name, exc)

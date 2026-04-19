"""
Module 2: RAG Setup
-------------------
Initializes a local Chroma vector store seeded with basic legal
guidelines and returns a LangChain retriever for use inside the
LangGraph agent.

Why this exists:
  * Grounding the legal analyzer in a small, curated knowledge base
    measurably reduces hallucinated citations.
  * The store is persisted to disk, so re-running the app is fast.

Notes on embeddings:
  * The user spec asked for HuggingFace open-source embeddings using
    ``all-MiniLM-L6-v2``. That model is a vanilla sentence-transformer,
    NOT a BGE model — so we use ``HuggingFaceEmbeddings`` rather than
    ``HuggingFaceBgeEmbeddings`` (BGE expects model-specific query/
    document prefixes that would hurt retrieval quality for MiniLM).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "contract_legal_guidelines"

# A small, inline, illustrative knowledge base. For a production system
# this would be replaced with curated excerpts from Indian Kanoon,
# GDPR articles, ICC model clauses, SEBI / MCA circulars, etc.
# Keeping it inline means the MVP runs with zero external data files.
SAMPLE_LEGAL_GUIDELINES = """\
Guideline 1 — Indemnification:
An indemnification clause shifts risk by requiring one party to
compensate the other for specified losses. Uncapped or one-sided
indemnities are HIGH risk. Best practice is to cap indemnity liability
at a multiple of contract value and to exclude indirect and
consequential damages.

Guideline 2 — Limitation of Liability:
Missing or one-sided limitation-of-liability clauses are a classic
red flag. Industry standard caps direct damages at twelve months of
fees paid and excludes consequential, special, and punitive damages.

Guideline 3 — Termination:
Termination-for-convenience with no notice period favors the
terminating party and is MEDIUM to HIGH risk for the counterparty.
A mutual 30-to-60 day notice period is standard.

Guideline 4 — Confidentiality:
NDAs should define "Confidential Information" precisely, exclude
publicly known or independently developed information, and cap
duration at three to five years post-termination. Perpetual
confidentiality over broad categories is risky and often unenforceable.

Guideline 5 — Governing Law and Jurisdiction:
Foreign jurisdiction clauses create enforcement risk. A neutral seat
of arbitration such as Singapore or London is preferable to a
home-court clause that favors only one party.

Guideline 6 — Auto-Renewal:
Automatic renewal clauses that lack a clear opt-out window are HIGH
risk for the buyer. Best practice requires written notice of
non-renewal 30 to 60 days before the renewal date.

Guideline 7 — Intellectual Property Assignment:
Broad IP assignment clauses that sweep in pre-existing "background"
IP or IP created outside the engagement are risky. Explicit
background-IP carve-outs are standard.

Guideline 8 — Payment Terms:
Net-60 or longer payment terms favor the buyer and are risky for the
vendor. Net-30 is standard. Unilateral rights of set-off should be
flagged.

Indian Contract Act, 1872 — Section 23:
An agreement whose object or consideration is unlawful is void. Any
clause that defeats a provision of law is unenforceable.

Indian Contract Act, 1872 — Section 27:
An agreement that restrains a person from exercising a lawful
profession, trade, or business is void to that extent. Broad
post-termination non-compete clauses are typically unenforceable in
India.
"""


def build_retriever(
    persist_dir: str = "./chroma_db",
    k: int = 3,
) -> VectorStoreRetriever:
    """Initialize or load a Chroma store and return a top-k retriever.

    Parameters
    ----------
    persist_dir:
        Filesystem path where the Chroma collection is stored. Created
        if it does not exist.
    k:
        Number of top similarity matches to return per query.

    Returns
    -------
    VectorStoreRetriever
        A LangChain retriever ready to be ``.invoke()``-d inside the
        agent graph.
    """
    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)

    # MiniLM on CPU is fast enough for interactive demo use and avoids
    # any GPU-dependency headaches when hosted on HF Spaces.
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(persist_path),
    )

    # Only seed the first time — make this idempotent across restarts.
    try:
        existing = store._collection.count()
    except Exception:  # defensive: API differences across chroma versions
        existing = 0

    if existing == 0:
        logger.info("Seeding Chroma collection with sample legal guidelines.")
        store.add_documents(_load_sample_documents())
    else:
        logger.info("Reusing existing Chroma collection (%d chunks).", existing)

    return store.as_retriever(search_kwargs={"k": k})


def _load_sample_documents() -> List[Document]:
    """Chunk the in-memory sample text into LangChain Documents."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=60,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_text(SAMPLE_LEGAL_GUIDELINES)
    return [
        Document(page_content=c, metadata={"source": "sample_guidelines"})
        for c in chunks
    ]

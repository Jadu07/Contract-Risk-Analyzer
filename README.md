# Contract Risk Analyzer

An agentic legal-assistance system that reads a contract PDF, identifies risky
clauses, explains why each one is risky, suggests mitigations, and highlights
the findings directly on the source PDF.

Built in two milestones:

| Milestone | Scope | Entry point |
| --- | --- | --- |
| **M1** — binary risk classifier | TF-IDF + Logistic Regression on cleaned legal text. Paste a clause, get `Risky` / `Not Risky`. | `main.py` |
| **M2** — agentic stack | Multimodal PDF ingestion, RAG-backed clause analysis, executive summary, highlighted-PDF preview, structured exports. | `app.py` |

---

## Table of contents

1. [Architecture](#architecture)
2. [Features](#features)
3. [Tech stack](#tech-stack)
4. [Setup](#setup)
5. [Configuration](#configuration)
6. [Running the app](#running-the-app)
7. [How it works (module-by-module)](#how-it-works-module-by-module)
8. [Project structure](#project-structure)
9. [Design decisions](#design-decisions)
10. [Troubleshooting](#troubleshooting)
11. [Disclaimer](#disclaimer)

---

## Architecture

```
                          ┌──────────────────────────────┐
       user PDF  ───────► │ ingestion.py                 │
                          │ Gemini 2.5 Flash (genai SDK) │
                          │  → segment into JSON clauses │
                          └──────────────┬───────────────┘
                                         │  list[Clause]
                                         ▼
                          ┌──────────────────────────────┐
                          │ retriever.py                 │
                          │ Chroma + MiniLM embeddings   │
                          │  seeded with legal guidelines│
                          └──────────────┬───────────────┘
                                         │  retriever
                                         ▼
                          ┌──────────────────────────────┐
                          │ agent.py  (LangGraph)        │
                          │  ┌───────────┐               │
                          │  │ retrieve  │◄──loop──┐     │
                          │  └─────┬─────┘         │     │
                          │        ▼               │     │
                          │  ┌───────────┐   more? │     │
                          │  │ analyze   │─────────┘     │
                          │  └─────┬─────┘  Groq Llama-3 │
                          │        ▼                     │
                          │  ┌───────────┐               │
                          │  │ summarize │  exec. summary│
                          │  └───────────┘               │
                          └──────────────┬───────────────┘
                                         │
             ┌───────────────────────────┼───────────────────────────┐
             ▼                           ▼                           ▼
      app.py (Streamlit)         pdf_export.py              pdf_highlighter.py
      risk summary + clauses     generated risk-report      annotate source PDF
      inline highlighted PDF     PDF (fpdf2)                with colored highlights
                                                            (PyMuPDF)
```

The graph state is an explicit `TypedDict` (`ContractState`) threaded through every
node. Loop termination is governed by a conditional edge from `analyze`.

---

## Features

- **PDF upload → clause-level risk report.** High and Medium risk clauses are
  surfaced individually with an explanation and a suggested mitigation.
- **Executive summary.** Contract overview, overall severity, top risks, and
  recommended negotiation actions.
- **Inline highlighted contract preview.** The uploaded PDF is rendered in the
  browser with colored annotations over risky passages (red = High,
  amber = Medium).
- **Low-risk and Unknown clauses are excluded from the findings list** but still
  counted in the top-of-page metric cards.
- **Downloadable exports.** Full JSON report + a generated risk-report PDF
  (risky clauses only).
- **API keys stay on the backend.** Loaded from `.env` (or environment
  variables) via `python-dotenv` — never exposed in the UI.
- **Graceful degradation.** Retriever and LLM failures never crash the
  pipeline; Groq's `json_validate_failed` errors are automatically repaired and
  retried.

---

## Tech stack

| Area | Choice | Why |
| --- | --- | --- |
| Ingestion | Gemini 2.5 Flash (`google-genai`) | Free tier; strong PDF OCR + structural segmentation. |
| Reasoning | Groq Llama 3.3 70B Versatile (`langchain-groq`) | Free tier; JSON mode; fast. |
| Retrieval | Chroma + `sentence-transformers/all-MiniLM-L6-v2` | Local, free, CPU-friendly. |
| Orchestration | LangGraph | Explicit state, conditional edges, clean loop semantics. |
| UI | Streamlit | Minimal boilerplate for a document-review dashboard. |
| Report PDF | fpdf2 | Pure-Python, no native deps. |
| Highlighting | PyMuPDF | The only library that can round-trip an existing PDF while adding annotations. |
| Config | `python-dotenv` | Keep secrets out of the UI and out of git. |

---

## Setup

**Requirements**
- Python 3.11 – 3.13 (pinned in `pyproject.toml`)
- A free Google AI Studio API key — <https://aistudio.google.com/apikey>
- A free Groq API key — <https://console.groq.com/keys>

**Install**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first run will download the MiniLM embedding weights (~90 MB) and build
the local Chroma collection under `./chroma_db/`. Subsequent runs reuse both.

---

## Configuration

Copy the example file and fill in your keys:

```bash
cp .env.example .env
```

`.env`:

```env
GEMINI_API_KEY=your-gemini-key
GROQ_API_KEY=your-groq-key
```

`.env` is already listed in `.gitignore`. Real environment variables always
take precedence over the file, so hosted deployments can inject keys via the
platform's secrets UI without needing a `.env` on disk.

---

## Running the app

**Milestone 2 (agentic, recommended):**

```bash
streamlit run app.py
```

Upload a contract PDF in the sidebar and click **Analyze Contract**. After
analysis you will see:

1. Risk-summary metric cards
2. Executive summary (contract overview, overall severity, top risks, actions)
3. Risky Clauses list — High and Medium only, each with explanation + mitigation
4. Inline highlighted PDF preview
5. Export buttons — JSON report + generated risk-report PDF

**Milestone 1 (legacy binary classifier):**

```bash
streamlit run main.py
```

Paste any clause text and get a `Risky` / `Not Risky` prediction from the
TF-IDF + Logistic Regression model trained in `model.ipynb`.

---

## How it works (module-by-module)

### `ingestion.py` — Gemini segmentation (no reasoning)
Uploads the PDF to Gemini, prompts it **strictly** as a segmenter (no legal
advice, no summaries), and parses the returned JSON array of
`{clause_number, text}` objects. The uploaded file is deleted from Gemini's
servers after segmentation — success or failure. Response is forced to JSON
via `response_mime_type="application/json"`.

### `retriever.py` — RAG setup
Builds (or reloads) a local Chroma collection seeded with a small curated set
of legal guidelines covering Indemnification, Limitation of Liability,
Termination, Confidentiality, Governing Law, Auto-Renewal, IP Assignment,
Payment Terms, and Indian Contract Act §23/§27. Guidelines are chunked with
`RecursiveCharacterTextSplitter` (400 chars, 60 overlap). Seeding is
idempotent.

### `agent.py` — LangGraph workflow
Three nodes:

- **retrieve** — top-k guideline chunks for the current clause
- **analyze** — Groq Llama-3 classifies the clause (`Low` / `Medium` / `High`)
  and returns a JSON object with `Risk_Level`, `Explanation`, `Mitigation`.
- **summarize** — runs once, at the end, producing the executive summary

JSON-mode errors from Groq (unquoted string values, missing commas) are caught
by `_invoke_with_json_repair`, which extracts the `failed_generation` payload
from the error, repairs the known patterns with a regex, and retries with a
stricter reminder before falling back to `Unknown`.

### `app.py` — Streamlit UI
Loads API keys from `.env` at startup, caches the retriever with
`@st.cache_resource`, runs the ingestion + agent pipeline, and renders:

- metric cards for all four severities (High / Medium / Low / Unknown)
- executive summary
- risky clause list (High + Medium only)
- inline highlighted PDF via base64 `<embed>`
- JSON + generated-PDF download buttons

### `pdf_export.py` — Risk report PDF
Renders the report as a fresh PDF using fpdf2 with built-in Helvetica. Text
is sanitized to Latin-1 so exotic Unicode in the contract never crashes the
renderer. Risky clauses are color-coded by severity.

### `pdf_highlighter.py` — Annotate the source PDF
Opens the user's uploaded PDF with PyMuPDF, splits each risky clause into
short snippets (~80 chars, sentence-aware), searches each page for those
snippets, and adds colored highlight annotations. The annotation tooltip
contains the clause number and the LLM's explanation. Returns the annotated
PDF bytes, which `app.py` base64-embeds into the page.

---

## Project structure

```
Contract-Risk-Analyzer/
├── app.py                    # Streamlit UI (Milestone 2)
├── agent.py                  # LangGraph workflow
├── ingestion.py              # Gemini PDF → clause JSON
├── retriever.py              # Chroma + MiniLM
├── pdf_export.py             # Generated risk report (fpdf2)
├── pdf_highlighter.py        # Annotate source PDF (PyMuPDF)
│
├── main.py                   # Milestone 1 classifier UI
├── legal_preprocessing_py.py # Milestone 1 text cleaning
├── model.ipynb               # Milestone 1 training notebook
│
├── requirements.txt
├── pyproject.toml
├── .env.example
├── .streamlit/config.toml    # fileWatcherType = "none"
├── chroma_db/                # (generated) persistent vector store
└── start.sh                  # Milestone 1 quick-start script
```

---

## Design decisions

- **Gemini only segments, Groq only reasons.** Legal analysis is kept off the
  ingestion model by an explicit "you are a segmenter, not a lawyer" prompt
  and a `temperature=0.0` JSON-mode call. This is defense-in-depth against the
  ingestion path drifting into "legal advice".
- **Free tier everywhere.** Gemini, Groq, local Chroma, HuggingFace CPU
  embeddings. No paid APIs on the critical path.
- **Explicit LangGraph state.** `ContractState` is a `TypedDict` with six
  fields; the whole state flows through every node, which makes the loop easy
  to reason about and test.
- **Fail-soft pipeline.** A retriever exception yields empty context; an
  analyzer exception yields an `Unknown` entry; a summarizer exception falls
  back to a deterministic count-based summary. The UI filter hides `Unknown`
  entries from the findings list so analyzer drop-outs don't pollute the
  report.
- **Structured JSON output + a repair pass.** Groq's JSON mode occasionally
  emits unquoted string values; rather than silently losing those clauses, the
  agent repairs and retries.
- **API keys live on the backend only.** The sidebar never exposes a key
  input — they come from `.env` or the hosting environment.

---

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `Missing backend configuration: GEMINI_API_KEY, GROQ_API_KEY` | `.env` not created or keys blank. | Copy `.env.example` to `.env` and fill in both keys. |
| `Warning: You are sending unauthenticated requests to the HF Hub.` | First-time MiniLM download. | Harmless — you can optionally set `HF_TOKEN` to raise rate limits. |
| `BertModel LOAD REPORT ... UNEXPECTED ... embeddings.position_ids` | MiniLM has one extra buffer the Bert loader doesn't use. | Harmless — ignore. |
| `ModuleNotFoundError: No module named 'torchvision'` during Streamlit reload | Streamlit's file watcher introspects transformers submodules. | Already fixed via `.streamlit/config.toml` → `fileWatcherType = "none"`. Restart Streamlit to pick up code changes. |
| Groq `json_validate_failed` in logs | Llama-3 produced malformed JSON. | Already handled by `_invoke_with_json_repair` — the clause is recovered on retry. |
| Inline PDF preview is blank | Browser blocked the data-URI embed (strict privacy mode / extensions). | Disable the blocker for this site, or open the generated risk-report PDF download instead. |

---

## Disclaimer

This tool is a student project built for educational and informational use
only. It does **not** constitute legal advice. The AI-generated risk
assessments may be incomplete, incorrect, or miss clauses entirely. Always
consult a qualified attorney before acting on any of its findings.

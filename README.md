# 🪨 Stratum

> RAPTOR-powered document intelligence — chat with your knowledge base using Google Gemini.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![Gemini](https://img.shields.io/badge/LLM-Google%20Gemini-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## What Is Stratum?

**Stratum** is a Retrieval-Augmented Generation (RAG) application that uses the **RAPTOR** technique to build hierarchical layers of understanding from your documents, then lets you query them through a clean chat interface powered by Google Gemini.

Unlike flat RAG systems that only retrieve raw chunks, Stratum builds a **tree of knowledge** — each layer summarizes the one below it — so it handles both fine-grained detail questions and big-picture architecture questions with equal precision.

---

## Architecture

```
Raw Document
    │
    ▼
[ Chunker ]                  ← Layer 0: raw text chunks (800 chars, 100 overlap)
    │
    ▼
[ Gemini Embeddings ]        ← Dense vector representations (768-dim)
    │
    ▼
[ K-Means Clustering ]       ← Groups similar chunks (dynamic K ≤ MAX_CLUSTERS)
    │
    ▼
[ Gemini Summarizer ]        ← Layer 1: cluster summaries
    │
    ▼ (recurse until MIN_CLUSTER_SIZE)
[ Summary of Summaries ]     ← Layer N: high-level overview
    │
    ▼
[ In-Memory Vector Store ]   ← All layers indexed together (NumPy)
    │
    ▼
[ Cosine Similarity Search ] ← Retrieves top-4 nodes on query
    │
    ▼
[ Gemini LLM Answer ]        ← Grounded, context-aware response
```

---

## Stack

| Component         | Technology                                           |
|-------------------|------------------------------------------------------|
| LLM + Embeddings  | Google Gemini (`gemini-1.5-flash`, `text-embedding-004`) |
| Clustering        | scikit-learn K-Means                                 |
| Vector Math       | NumPy                                                |
| Configuration     | pydantic-settings                                    |
| UI                | Streamlit                                            |

No LangChain. No LlamaIndex. Raw orchestration.

---

## Quickstart

### 1. Clone & Install

```bash
git clone https://github.com/1SHAMAY1/Stratum
cd stratum
pip install -r requirements.txt
```

### 2. Set Up Environment

```bash
cp .env.example .env
# Open .env and add your GEMINI_API_KEY
```

Get your **free** API key at [Google AI Studio](https://aistudio.google.com).

### 3. Run

```bash
streamlit run app.py
```

---

## Configuration

All settings live in `config/settings.py` and can be overridden via `.env`:

| Variable           | Default                      | Description                             |
|--------------------|------------------------------|-----------------------------------------|
| `GEMINI_API_KEY`   | *(required)*                 | Your Gemini API key                     |
| `EMBEDDING_MODEL`  | `models/text-embedding-004`  | Embedding model                         |
| `LLM_MODEL`        | `gemini-1.5-flash`           | Chat/summarisation model                |
| `MAX_CLUSTERS`     | `5`                          | Max K-Means clusters per layer          |
| `MIN_CLUSTER_SIZE` | `2`                          | Stop recursion below this               |
| `CHUNK_SIZE`       | `800`                        | Characters per document chunk           |
| `CHUNK_OVERLAP`    | `100`                        | Overlap between consecutive chunks      |

---

## Project Structure

```
stratum/
├── app.py               ← Streamlit entry point
├── requirements.txt
├── .env.example
├── .cursorrules
├── config/
│   ├── __init__.py
│   └── settings.py      ← Pydantic settings (single source of truth)
└── src/
    ├── __init__.py
    ├── clustering.py    ← RAPTOR tree builder
    └── pipeline.py      ← Orchestrator (ingest → retrieve → answer)
```

---

## License

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

# Agentic RAG Demo

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete agentic RAG (Retrieval-Augmented Generation) system built with modern AI technologies.

## Tech Stack

| Component | Technology |
|-----------|------------|
| Deployment/LLM | Google Cloud Gemini 2.5 Flash |
| Evaluation | LangSmith |
| Framework | LangChain |
| Vector DB | Pinecone |
| Embeddings | Ollama (nomic-embed-text) |
| Data Extraction | Firecrawl |
| Memory | Zep |
| Alignment/Observability | Guardrails AI |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG Agent                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Gemini     │  │  Guardrails  │  │   LangSmith  │          │
│  │   2.5 Flash  │  │      AI      │  │  (Tracing)   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
├─────────────────────────────────────────────────────────────────┤
│                     Memory & Retrieval                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │     Zep      │  │   Pinecone   │  │    Ollama    │          │
│  │   (Memory)   │  │ (Vector DB)  │  │ (Embeddings) │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
├─────────────────────────────────────────────────────────────────┤
│                     Data Extraction                              │
│  ┌──────────────┐  ┌──────────────┐                             │
│  │  Firecrawl   │  │    PyPDF     │                             │
│  │    (Web)     │  │   (PDFs)     │                             │
│  └──────────────┘  └──────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
```

## Setup

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/) running locally
- API keys for: Google Cloud, Pinecone, Firecrawl, Zep, LangSmith

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/agentic-rag-demo.git
cd agentic-rag-demo
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e ".[dev]"
# Or using requirements.txt
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

5. Pull the Ollama embedding model:
```bash
ollama pull nomic-embed-text
```

### Running Tests

```bash
pytest tests/
```

### Running the Demo

```bash
python -m demo.run_demo
```

## Project Structure

```
agentic-rag-demo/
├── src/
│   ├── config.py              # Configuration management
│   ├── agents/
│   │   └── rag_agent.py       # Main agentic RAG implementation
│   ├── data/
│   │   ├── firecrawl_loader.py # Web scraping
│   │   └── pdf_loader.py       # PDF extraction
│   ├── embeddings/
│   │   └── ollama_embeddings.py # Embedding model
│   ├── memory/
│   │   └── zep_memory.py       # Conversation memory
│   ├── retrieval/
│   │   └── pinecone_store.py   # Vector store
│   ├── guardrails/
│   │   └── validators.py       # Input/output validation
│   └── llm/
│       └── gemini.py           # LLM wrapper
├── tests/                      # Test suite
└── demo/
    └── run_demo.py             # Interactive demo
```

## Features

- **Multi-source Data Ingestion**: Load data from web pages and PDFs
- **Semantic Search**: Find relevant documents using vector similarity
- **Conversation Memory**: Maintain context across interactions
- **Input/Output Validation**: Ensure safe and appropriate responses
- **Observability**: Full tracing with LangSmith

## License

MIT

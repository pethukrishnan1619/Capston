# Multi‑Agent RAG System Package

This project contains a Python package (`my_code_package`) that implements a multi‑agent
retrieval‑augmented generation (RAG) system with ReAct‑style reasoning, tool integration
(weather and calculator), and orchestration via LangGraph.

The package is derived from the user's Jupyter notebook and has been refactored into
Python modules for easier maintenance and deployment in a Git repository. The
functionality remains identical to the original notebook.

## Structure

- `my_code_package/`
  - `rag.py` – Functions for loading PDFs, creating a FAISS vector store, and
    generating answers from retrieved context.
  - `tools.py` – Implementation of the weather and calculator tools with
    pydantic input validation.
  - `agents.py` – Definitions of the planner, retriever, tool executor and
    synthesiser agents, plus global setters for the vector store and LLM.
  - `orchestrator.py` – Initialises the system, constructs the LangGraph
    workflow, and provides a `handle_user_query` function to run a query end‑to‑end.
  - `interface.py` – Gradio UI for interacting with the system.
  - `__init__.py` – Marks the directory as a package.
- `main.py` – Command‑line entry point to run the system with a given PDF.
- `README.md` – This file.
- `requirements.txt` – List of Python dependencies.

## Installation

1. Create and activate a virtual environment (optional but recommended).
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Download or place your PDF document in a known location.

## Usage

### Command‑Line Interface

Run the package from the repository root using `python main.py`. You must specify the
path to your PDF file. The script will build or load a FAISS index and start an
interactive prompt for queries:

```bash
python main.py --pdf /path/to/your.pdf --faiss-dir ./faiss_store --no-ui
```

Entering queries such as `weather in Chennai` or `calculate (10+20)/2` will
invoke the appropriate tool or retrieval function. Type `quit` or `exit` to leave
CLI mode.

### Gradio UI

Omit the `--no-ui` flag to launch a Gradio web interface instead:

```bash
python main.py --pdf /path/to/your.pdf
```

This will open a local browser window where you can type questions, see the
answer, citations (if any), and the ReAct reasoning trace.

## Notes

- The weather tool uses the [Open‑Meteo](https://open-meteo.com/) free API and
  therefore requires an internet connection. If your network has SSL issues,
  consider replacing the implementation in `tools.py` with another service.
- The RAG functions use a HuggingFace model (`google/flan-t5-base`) via
  transformers, which may download model weights the first time you run the
  system.
- The FAISS index is stored in the directory specified via `--faiss-dir` and
  will be reused across runs to speed up start‑up time.

"""
Entry point for running the multi‑agent RAG system.

Usage:
    python main.py --pdf /path/to/your.pdf [--no-ui]

The script initialises the system with the specified PDF, then either launches a
Gradio UI or runs a simple command‑line loop for asking questions.
"""
import argparse

from my_code_package.orchestrator import initialise_system, handle_user_query
from my_code_package.interface import launch_demo


def main():
    parser = argparse.ArgumentParser(description="Run the multi-agent RAG system.")
    parser.add_argument(
        "--pdf",
        type=str,
        required=True,
        help="Path to the PDF document to ingest (for RAG retrieval).",
    )
    parser.add_argument(
        "--faiss-dir",
        type=str,
        default="./faiss_store",
        help="Directory for storing/loading the FAISS index.",
    )
    parser.add_argument(
        "--no-ui",
        action="store_true",
        help="Run in command-line mode instead of launching the Gradio UI.",
    )
    args = parser.parse_args()
    # Initialise vector store and LLM
    initialise_system(pdf_path=args.pdf, faiss_directory=args.faiss_dir)
    if args.no_ui:
        # Simple CLI loop
        print("Multi-agent RAG system ready. Type 'quit' to exit.")
        while True:
            try:
                user_query = input("\nAsk a question: ").strip()
            except EOFError:
                break
            if not user_query or user_query.lower() in {"quit", "exit"}:
                break
            response = handle_user_query(user_query)
            print("\nAnswer:\n", response.get("answer", ""))
            citations = response.get("citations", [])
            if citations:
                print("\nCitations:")
                for c in citations:
                    print(f"- {c['source']} (page {c['page']})")
            else:
                print("\nNo citations available.")
    else:
        # Launch Gradio UI
        launch_demo()


if __name__ == "__main__":
    main()

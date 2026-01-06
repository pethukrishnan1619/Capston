"""
RAG functions: loading PDF, chunking, embedding with HuggingFace, building FAISS vector stores, and generating answers.
This module replicates the logic from the user's Jupyter notebook but in a Python module.
"""
from typing import List, Tuple, Dict, Any
import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Public API
__all__ = [
    "extract_pdf_pages",
    "initialize_vector_store",
    "retrieve_rag_chunks",
    "create_local_llm",
    "generate_answer_from_context",
]

def extract_pdf_pages(path: str) -> List[Document]:
    """Extracts text pages from a PDF into Document objects with metadata."""
    reader = PdfReader(path)
    docs: List[Document] = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            docs.append(Document(page_content=text, metadata={"source": os.path.basename(path), "page": i}))
    return docs


def initialize_vector_store(
    pdf_path: str,
    faiss_directory: str = "./faiss_store",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> FAISS:
    """Build or load a FAISS vector store from a PDF. If a FAISS index exists, it is loaded.

    Args:
        pdf_path: Path to the PDF document.
        faiss_directory: Directory where the FAISS index and metadata will be stored.
        embedding_model: Name of the HuggingFace embedding model.
        chunk_size: Number of characters per text chunk.
        chunk_overlap: Overlap between chunks to preserve context.

    Returns:
        A FAISS vector store loaded with embeddings from the document.
    """
    os.makedirs(faiss_directory, exist_ok=True)
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    # If index exists, load it
    if os.path.exists(os.path.join(faiss_directory, "index.faiss")):
        return FAISS.load_local(faiss_directory, embeddings, allow_dangerous_deserialization=True)

    # Otherwise, create new index
    docs = extract_pdf_pages(pdf_path)
    if not docs:
        raise ValueError(
            "No text extracted from PDF. If it is scanned (image-only), you need OCR or another extraction method."
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(faiss_directory)
    return vs


def retrieve_rag_chunks(vector_store: FAISS, query: str, k: int = 3) -> Tuple[str, List[Dict[str, Any]]]:
    """Retrieve top-k similar chunks from the vector store for a given query.

    Returns a tuple of concatenated context string and a list of citation dicts (source, page).
    """
    retrieved_docs = vector_store.similarity_search(query, k=k)
    context = "\n\n".join(
        f"(source={d.metadata.get('source')}, page={d.metadata.get('page')})\n{d.page_content}" for d in retrieved_docs
    )
    citations = [
        {"source": d.metadata.get("source"), "page": d.metadata.get("page")} for d in retrieved_docs
    ]
    return context, citations


def create_local_llm(model_id: str = "google/flan-t5-base", max_new_tokens: int = 220) -> HuggingFacePipeline:
    """Initialise a local HuggingFace text-to-text pipeline for answer generation."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    gen_pipe = pipeline(
        "text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=max_new_tokens
    )
    return HuggingFacePipeline(pipeline=gen_pipe)


def generate_answer_from_context(llm: HuggingFacePipeline, context: str, query: str) -> str:
    """Generate an answer from provided context and query using the given LLM.

    If the answer is not present in the context, the model should return a fallback message.
    """
    prompt = f"""
Answer the question strictly using the context below.
If the answer is not present in the context, say: \"I don't know from the provided documents.\"

Context:
{context}

Question: {query}
Answer:
""".strip()
    return llm.invoke(prompt)
